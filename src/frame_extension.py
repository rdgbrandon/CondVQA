"""Conditional VQA module - LLM-based question parsing and two-stage video analysis with Integrated Gradients"""

import os
import re
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from captum.attr import visualization
from tqdm import tqdm

from .utils import setup_colormap
from .image_interpreter import compute_single_image_attribution
from .text_interpreter import compute_text_attributions, visualize_text_attributions


def parse_question_with_llm(question, text_model, text_tokenizer):
    """
    Use dedicated text LLM to parse complex conditional questions into frame-finding and answering stages.

    Args:
        question: Natural language question (e.g., "When there is a dog in the frame, what color is the dog?")
        text_model: Text-only LLM for parsing (e.g., Qwen)
        text_tokenizer: Tokenizer for the text model

    Returns:
        dict with:
            - 'type': 'conditional' or 'simple' or 'timestamp'
            - 'frame_condition': Question to identify relevant frames (e.g., "Is there a dog?")
            - 'answer_question': Question to answer on identified frames (e.g., "What color is the dog?")
            - 'original_question': Original question
            - 'timestamp': For timestamp type, the time in seconds

    Examples:
        "When there is a dog in the frame, what color is the dog?"
        -> {'type': 'conditional',
            'frame_condition': 'Is there a dog in the frame?',
            'answer_question': 'What color is the dog?'}

        "What is happening at 4 seconds?"
        -> {'type': 'timestamp',
            'timestamp': 4.0,
            'answer_question': 'What is happening?'}
    """

    # Create parsing prompt for the LLM using chat template
    messages = [
        {
            "role": "system",
            "content": "You parse video questions. If a question has 'when X, Y' or 'where X, Y' structure, output CONDITIONAL with both CONDITION and QUESTION fields. Otherwise output SIMPLE or TIMESTAMP."
        },
        {
            "role": "user",
            "content": f"""Parse: "{question}"

TASK: Identify if this is CONDITIONAL, TIMESTAMP, or SIMPLE, then extract the appropriate fields.

TYPE RULES:
- CONDITIONAL: Has "when X, Y" or "where X, Y" or "when you see X, Y" pattern → Split into CONDITION (yes/no) and QUESTION
- TIMESTAMP: Mentions specific time like "at 4 seconds" → Extract TIME and QUESTION
- SIMPLE: No condition, no timestamp → Just output QUESTION

EXAMPLES:

"When there is a dog in the frame, what color is the dog?"
→ CONDITIONAL
CONDITION: Is there a dog in the frame?
QUESTION: What color is the dog?

"When you see a number one, what color is it?"
→ CONDITIONAL
CONDITION: Is there a number one?
QUESTION: What color is it?

"When there is a cat, what color is the cat?"
→ CONDITIONAL
CONDITION: Is there a cat?
QUESTION: What color is the cat?

"Where the person is running, what are they wearing?"
→ CONDITIONAL
CONDITION: Is the person running?
QUESTION: What are they wearing?

"What happens at 4 seconds?"
→ TIMESTAMP
TIME: 4
QUESTION: What happens?

"What is in this video?"
→ SIMPLE
QUESTION: What is in this video?

Now parse: "{question}"

Output (choose ONE type only - CONDITIONAL, TIMESTAMP, or SIMPLE):"""
        }
    ]

    # Format using chat template
    text = text_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = text_tokenizer([text], return_tensors="pt").to(text_model.device)

    # Generate response with strict settings for consistent output
    with torch.no_grad():
        output_ids = text_model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            pad_token_id=text_tokenizer.pad_token_id,
            eos_token_id=text_tokenizer.eos_token_id
        )

    # Decode response (only the generated part, not the input)
    response = text_tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    # Extract the assistant's response (after the prompt)
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    # Debug: Print raw LLM response for troubleshooting
    print(f"\n[DEBUG] Raw LLM Response:\n{response}\n")

    # Parse the LLM response
    result = {
        'original_question': question,
        'type': 'simple',
        'frame_condition': None,
        'answer_question': question,
        'timestamp': None
    }

    lines = [line.strip() for line in response.split('\n') if line.strip()]

    if not lines:
        return result

    # Extract type from first line (handle both "CONDITIONAL" and "TYPE: CONDITIONAL")
    response_type = lines[0].upper()
    if ':' in response_type:
        # Format: "TYPE: CONDITIONAL"
        response_type = response_type.split(':', 1)[1].strip()

    if 'CONDITIONAL' in response_type:
        result['type'] = 'conditional'
        for line in lines[1:]:
            if line.upper().startswith('CONDITION:'):
                result['frame_condition'] = line.split(':', 1)[1].strip()
            elif line.upper().startswith('QUESTION:'):
                result['answer_question'] = line.split(':', 1)[1].strip()

        # Validation: If conditional but no condition found, this is an error
        if result['frame_condition'] is None or result['frame_condition'] == '':
            print(f"⚠ ERROR: LLM classified as CONDITIONAL but failed to extract condition!")
            print(f"⚠ LLM Response: {response}")
            print(f"⚠ This indicates the LLM is not following instructions properly.")
            raise ValueError(f"LLM parsing failed: classified as CONDITIONAL but no condition extracted. Response: {response}")

    elif 'TIMESTAMP' in response_type:
        result['type'] = 'timestamp'
        for line in lines[1:]:
            if line.upper().startswith('TIME:'):
                try:
                    time_str = line.split(':', 1)[1].strip()
                    # Extract first number found
                    match = re.search(r'\d+(?:\.\d+)?', time_str)
                    if match:
                        result['timestamp'] = float(match.group())
                except:
                    pass
            elif line.upper().startswith('QUESTION:'):
                result['answer_question'] = line.split(':', 1)[1].strip()

    elif 'SIMPLE' in response_type:
        for line in lines[1:]:
            if line.upper().startswith('QUESTION:'):
                result['answer_question'] = line.split(':', 1)[1].strip()

    return result


def extract_frames(video_path, output_dir="frames", fps_sample=1, max_frames=None):
    """Extract frames from video with sampling."""
    os.makedirs(output_dir, exist_ok=True)

    vid = cv2.VideoCapture(video_path)
    video_fps = vid.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        print("Warning: Could not get video FPS, defaulting to 30")
        video_fps = 30

    frame_interval = int(video_fps / fps_sample) if fps_sample > 0 else 1

    print(f"Video FPS: {video_fps:.2f}")
    print(f"Sampling every {frame_interval} frames ({fps_sample} FPS)")

    frame_paths = []
    count = 0
    frame_number = 0

    with tqdm(desc="Extracting frames") as pbar:
        while True:
            success, image = vid.read()
            if not success:
                break

            if frame_number % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"frame_{count:04d}.jpg")
                cv2.imwrite(frame_path, image)
                frame_paths.append(frame_path)
                count += 1
                pbar.update(1)

                if max_frames and count >= max_frames:
                    break

            frame_number += 1

    vid.release()
    print(f"✓ Extracted {count} frames to {output_dir}/")

    return frame_paths, video_fps, count


def identify_frames_by_condition(frame_paths, condition_question, model, processor, confidence_threshold=0.5):
    """
    Identify which frames match a given condition using the VLM.

    Args:
        frame_paths: List of all frame paths
        condition_question: Yes/no question to identify frames (e.g., "Is there a dog in the frame?")
        model: Vision-language model
        processor: Model's processor
        confidence_threshold: Minimum confidence to consider frame as matching (default: 0.5)

    Returns:
        list of tuples: [(frame_idx, frame_path, confidence_score), ...]
            Only frames where the model answers "yes" with confidence above threshold
    """
    from PIL import Image

    matching_frames = []

    print(f"Identifying frames matching: '{condition_question}'")

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Scanning frames")):
        # Load frame
        image = Image.open(frame_path).convert("RGB")

        # Prepare inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": condition_question}
                ]
            }
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_sizes = inputs["image_sizes"]

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )

        # Decode only the generated tokens (not the prompt)
        prompt_length = input_ids.shape[1]
        generated_tokens = output_ids[0, prompt_length:]
        answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

        # Debug: Print frame responses (first 10 frames only to avoid spam)
        if frame_idx < 10:
            print(f"  Frame {frame_idx}: {answer[:100]}")

        # Check if answer indicates "yes" (positive match)
        # Look for affirmative words at the START of the response (more reliable)
        affirmative_words = ['yes', 'yeah', 'yep', 'sure', 'absolutely', 'definitely', 'correct', 'true']
        negative_words = ['no', 'not', 'none', 'nope', 'negative', 'false']

        # Check for negative words first (higher priority)
        has_negative = any(answer.startswith(word) for word in negative_words)
        has_affirmative = any(answer.startswith(word) for word in affirmative_words)

        is_match = has_affirmative and not has_negative

        # Confidence based on answer clarity
        if is_match:
            if answer.startswith(tuple(affirmative_words)):
                confidence = 0.9
            else:
                confidence = 0.6

            if confidence >= confidence_threshold:
                matching_frames.append((frame_idx, frame_path, confidence))

        torch.cuda.empty_cache()

    print(f"✓ Found {len(matching_frames)} matching frames out of {len(frame_paths)}")

    return matching_frames


def get_frames_for_timerange(frame_paths, fps_sample, start_sec, end_sec):
    """
    Get frame paths corresponding to a specific time range.

    Args:
        frame_paths: List of all extracted frame paths
        fps_sample: Frames per second sampling rate
        start_sec: Start time in seconds
        end_sec: End time in seconds

    Returns:
        list: Frame paths within the specified time range
    """
    start_idx = int(start_sec * fps_sample)
    end_idx = int(end_sec * fps_sample)

    # Clamp to valid range
    start_idx = max(0, min(start_idx, len(frame_paths) - 1))
    end_idx = max(0, min(end_idx, len(frame_paths) - 1))

    # Ensure at least one frame
    if start_idx == end_idx:
        return [frame_paths[start_idx]]

    return frame_paths[start_idx:end_idx + 1]


def aggregate_temporal_results(results, method='most_confident'):
    """
    Aggregate results from multiple frames into a single result.

    Args:
        results: List of result dicts from compute_single_image_attribution
        method: Aggregation method - 'most_confident', 'consensus', or 'average'

    Returns:
        dict: Aggregated result with prediction, confidence, and attribution
    """
    if len(results) == 0:
        return None

    if len(results) == 1:
        return results[0]

    if method == 'most_confident':
        # Return the frame with highest confidence
        best_result = max(results, key=lambda r: r['confidence'])
        best_result['aggregation_method'] = 'most_confident'
        best_result['num_frames_aggregated'] = len(results)
        return best_result

    elif method == 'consensus':
        # Return the most common prediction
        predictions = [r['prediction'] for r in results]
        prediction_counts = Counter(predictions)
        most_common_pred, count = prediction_counts.most_common(1)[0]

        # Find a representative frame with this prediction (preferably high confidence)
        matching_results = [r for r in results if r['prediction'] == most_common_pred]
        representative = max(matching_results, key=lambda r: r['confidence'])

        representative['aggregation_method'] = 'consensus'
        representative['num_frames_aggregated'] = len(results)
        representative['consensus_count'] = count
        representative['consensus_percentage'] = (count / len(results)) * 100
        return representative

    elif method == 'average':
        # Average the attributions and confidence scores
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_attributions = np.mean([r['attributions'] for r in results], axis=0)

        # Use the most common prediction
        predictions = [r['prediction'] for r in results]
        most_common_pred = Counter(predictions).most_common(1)[0][0]

        # Use the first result as template
        aggregated = results[0].copy()
        aggregated['prediction'] = most_common_pred
        aggregated['confidence'] = avg_confidence
        aggregated['attributions'] = avg_attributions
        aggregated['aggregation_method'] = 'average'
        aggregated['num_frames_aggregated'] = len(results)

        return aggregated

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def process_frame_batch(frame_paths, question, model, processor, show_visualizations=False, include_text_attribution=False):
    """Process a batch of frames for a single question."""
    default_cmap = setup_colormap()
    results = []

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc="Processing frames")):
        torch.cuda.empty_cache()

        result = compute_single_image_attribution(
            frame_path, question, model, processor,
            n_steps=5, top_k=5
        )

        result['frame_idx'] = frame_idx
        result['frame_path'] = frame_path

        if include_text_attribution:
            text_attr_result = compute_text_attributions(
                frame_path, question, model, processor, n_steps=5
            )
            result['text_attr_result'] = text_attr_result

            if show_visualizations:
                visualize_text_attributions(text_attr_result)

        results.append(result)

        if show_visualizations:
            fig, _ = visualization.visualize_image_attr_multiple(
                result['attributions'],
                result['original_image'],
                ["original_image", "heat_map"],
                ["all", "absolute_value"],
                titles=[f"Frame {frame_idx}", f"Attribution: '{result['prediction']}'"],
                cmap=default_cmap,
                show_colorbar=True,
                fig_size=(12, 6)
            )
            plt.tight_layout()
            plt.show()

        torch.cuda.empty_cache()

    return results


def conditional_query_vqa_interpret(
    video_path,
    questions,
    model,
    processor,
    text_model=None,
    text_tokenizer=None,
    fps_sample=1,
    output_dir="conditional_query_analysis",
    aggregation_method='most_confident',
    confidence_threshold=0.5,
    show_visualizations=True,
    save_results=True,
    include_text_attribution=False
):
    """
    Answer complex conditional questions using LLM-based parsing and two-stage analysis.

    This function handles questions like:
    - "When there is a dog in the frame, what color is the dog?"
    - "Where the person is running, what are they wearing?"
    - "At frames with a car, is it red or blue?"

    The process:
    1. LLM parses question into: condition + question
    2. Scan all frames to find those matching the condition
    3. Answer the question only on matching frames
    4. Aggregate results

    Args:
        video_path: Path to video file
        questions: List of questions (can include conditional clauses or timestamps)
        model: Vision-language model
        processor: Model's processor
        text_model: Text-only LLM for question parsing (optional, will be loaded if not provided)
        text_tokenizer: Tokenizer for text model (optional, will be loaded if not provided)
        fps_sample: Frames per second to sample from video (default: 1)
        output_dir: Directory to save results
        aggregation_method: How to combine multi-frame results
            - 'most_confident': Use frame with highest confidence
            - 'consensus': Use most common prediction
            - 'average': Average attributions and confidences
        confidence_threshold: Minimum confidence for frame condition matching (default: 0.5)
        show_visualizations: Whether to display visualizations
        save_results: Whether to save results to disk
        include_text_attribution: Whether to compute text attributions

    Returns:
        dict: Results for each question with conditional analysis

    Example:
        >>> results = conditional_query_vqa_interpret(
        ...     "video.mp4",
        ...     ["When there is a dog in the frame, what color is the dog?"],
        ...     model, processor
        ... )
    """
    print(f"\n{'='*70}")
    print("CONDITIONAL QUERY VIDEO ANALYSIS")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Questions: {len(questions)}")
    print(f"Sample Rate: {fps_sample} FPS")
    print(f"Aggregation Method: {aggregation_method}")
    print(f"Confidence Threshold: {confidence_threshold}")
    print(f"{'='*70}\n")

    # Load text model if not provided
    if text_model is None or text_tokenizer is None:
        print("Loading text model for question parsing...")
        from .model_loader import load_text_model
        text_model, text_tokenizer = load_text_model()
        print()

    if save_results:
        os.makedirs(output_dir, exist_ok=True)

    # Extract all frames first
    print("Extracting frames from video...")
    frames_dir = os.path.join(output_dir, "frames")
    frame_paths, video_fps, num_frames = extract_frames(
        video_path,
        output_dir=frames_dir,
        fps_sample=fps_sample
    )

    if len(frame_paths) == 0:
        print("Error: No frames extracted from video")
        return

    video_duration = len(frame_paths) / fps_sample
    print(f"✓ Video duration: {video_duration:.2f} seconds")
    print(f"✓ Total frames extracted: {num_frames}")

    all_results = {}
    default_cmap = setup_colormap()

    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {q_idx}/{len(questions)}: {question}")
        print(f"{'='*70}\n")

        # Step 1: Parse question with text LLM
        print("Parsing question with text LLM...")
        parsed = parse_question_with_llm(question, text_model, text_tokenizer)

        print(f"Question Type: {parsed['type']}")
        if parsed['type'] == 'conditional':
            print(f"Frame Condition: {parsed['frame_condition']}")
            print(f"Answer Question: {parsed['answer_question']}")
        elif parsed['type'] == 'timestamp':
            print(f"Timestamp: {parsed['timestamp']}s")
            print(f"Answer Question: {parsed['answer_question']}")
        else:
            print(f"Answer Question: {parsed['answer_question']}")

        # Step 2: Identify relevant frames
        if parsed['type'] == 'conditional':
            # Find frames matching the condition
            matching_frames = identify_frames_by_condition(
                frame_paths,
                parsed['frame_condition'],
                model,
                processor,
                confidence_threshold=confidence_threshold
            )

            if len(matching_frames) == 0:
                print("⚠ No frames matched the condition. Skipping this question.")
                all_results[question] = {
                    'error': 'No matching frames found',
                    'parsed': parsed
                }
                continue

            # Extract just the frame paths
            relevant_frames = [fp for _, fp, _ in matching_frames]
            frame_indices = [idx for idx, _, _ in matching_frames]
            frame_confidences = [conf for _, _, conf in matching_frames]

            print(f"\nMatching frame indices: {frame_indices}")
            print(f"Average condition confidence: {np.mean(frame_confidences):.2%}")

        elif parsed['type'] == 'timestamp':
            # Use timestamp to select frame
            if parsed['timestamp'] is not None:
                relevant_frames = get_frames_for_timerange(
                    frame_paths, fps_sample, parsed['timestamp'], parsed['timestamp']
                )
                frame_indices = [int(parsed['timestamp'] * fps_sample)]
            else:
                print("⚠ Could not parse timestamp. Using all frames.")
                relevant_frames = frame_paths
                frame_indices = list(range(len(frame_paths)))

        else:  # simple
            # Use all frames
            relevant_frames = frame_paths
            frame_indices = list(range(len(frame_paths)))

        # Step 3: Answer the question on identified frames
        print(f"\nAnalyzing {len(relevant_frames)} frame(s) for: '{parsed['answer_question']}'")

        frame_results = process_frame_batch(
            relevant_frames,
            parsed['answer_question'],
            model,
            processor,
            show_visualizations=False,
            include_text_attribution=include_text_attribution
        )

        # Step 4: Aggregate results
        aggregated_result = aggregate_temporal_results(frame_results, method=aggregation_method)

        if aggregated_result:
            print(f"\n{'─'*70}")
            print(f"AGGREGATED RESULT ({aggregation_method})")
            print(f"{'─'*70}")
            print(f"Prediction: '{aggregated_result['prediction']}'")
            print(f"Confidence: {aggregated_result['confidence']:.4f} ({aggregated_result['confidence']*100:.2f}%)")

            if 'consensus_percentage' in aggregated_result:
                print(f"Consensus: {aggregated_result['consensus_count']}/{len(frame_results)} frames ({aggregated_result['consensus_percentage']:.1f}%)")

            print(f"\nTop predictions:")
            for i, (token, prob) in enumerate(aggregated_result['top_predictions'], 1):
                print(f"  {i}. '{token}' - {prob:.4f} ({prob*100:.2f}%)")

            # Visualize aggregated result
            if show_visualizations:
                fig, _ = visualization.visualize_image_attr_multiple(
                    aggregated_result['attributions'],
                    aggregated_result['original_image'],
                    ["original_image", "heat_map"],
                    ["all", "absolute_value"],
                    titles=[
                        f"Matching Frame(s)",
                        f"Attribution: '{aggregated_result['prediction']}'"
                    ],
                    cmap=default_cmap,
                    show_colorbar=True,
                    fig_size=(12, 6)
                )
                plt.suptitle(f"Q{q_idx}: {question}", fontsize=12, y=1.02)
                plt.tight_layout()

                if save_results:
                    save_path = os.path.join(output_dir, f"attribution_q{q_idx}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"✓ Saved visualization to {save_path}")

                plt.show()

            # Save detailed report
            if save_results:
                report_path = os.path.join(output_dir, f"report_q{q_idx}.txt")
                with open(report_path, 'w') as f:
                    f.write("=" * 70 + "\n")
                    f.write("CONDITIONAL QUERY ANALYSIS REPORT\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"Original Question: {question}\n")
                    f.write(f"Question Type: {parsed['type']}\n\n")

                    if parsed['type'] == 'conditional':
                        f.write(f"Frame Condition: {parsed['frame_condition']}\n")
                        f.write(f"Matching Frames: {len(relevant_frames)} out of {len(frame_paths)}\n")
                        f.write(f"Frame Indices: {frame_indices}\n")
                        if 'frame_confidences' in locals():
                            f.write(f"Avg Condition Confidence: {np.mean(frame_confidences):.2%}\n")
                    elif parsed['type'] == 'timestamp':
                        f.write(f"Timestamp: {parsed['timestamp']}s\n")

                    f.write(f"Answer Question: {parsed['answer_question']}\n")
                    f.write(f"Aggregation Method: {aggregation_method}\n\n")

                    f.write("-" * 70 + "\n")
                    f.write("AGGREGATED RESULT\n")
                    f.write("-" * 70 + "\n\n")
                    f.write(f"Prediction: '{aggregated_result['prediction']}'\n")
                    f.write(f"Confidence: {aggregated_result['confidence']:.4f} ({aggregated_result['confidence']*100:.2f}%)\n\n")

                    if 'consensus_percentage' in aggregated_result:
                        f.write(f"Consensus: {aggregated_result['consensus_count']}/{len(frame_results)} frames ")
                        f.write(f"({aggregated_result['consensus_percentage']:.1f}%)\n\n")

                    f.write("Top Predictions:\n")
                    for i, (token, prob) in enumerate(aggregated_result['top_predictions'], 1):
                        f.write(f"  {i}. '{token}' - {prob:.4f} ({prob*100:.2f}%)\n")

                    f.write("\n" + "-" * 70 + "\n")
                    f.write("INDIVIDUAL FRAME RESULTS\n")
                    f.write("-" * 70 + "\n\n")

                    for i, r in enumerate(frame_results):
                        if parsed['type'] == 'conditional' and i < len(frame_confidences):
                            f.write(f"Frame {frame_indices[i]} (condition conf: {frame_confidences[i]:.2%}):\n")
                        else:
                            f.write(f"Frame {i}:\n")
                        f.write(f"  Prediction: '{r['prediction']}'\n")
                        f.write(f"  Confidence: {r['confidence']:.4f}\n\n")

                print(f"✓ Saved report to {report_path}")

            # Add metadata
            aggregated_result['query_info'] = {
                'parsed': parsed,
                'frame_indices': frame_indices,
                'num_matching_frames': len(relevant_frames),
                'total_frames': len(frame_paths),
                'individual_results': frame_results
            }

            all_results[question] = aggregated_result

        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("✓ CONDITIONAL QUERY ANALYSIS COMPLETE")
    if save_results:
        print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")

    return all_results
