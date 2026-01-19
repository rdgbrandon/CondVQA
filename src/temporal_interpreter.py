"""Temporal video interpretation module for Visual Question Answering with Integrated Gradients"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization
from tqdm import tqdm

from .utils import load_image_from_file, replace_with_padding, setup_colormap


def extract_frames(video_path, output_dir="frames", fps_sample=1, max_frames=None):
    """
    Extract frames from video with sampling

    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps_sample: Sample rate (1 = 1 frame per second, 0.5 = 1 frame per 2 seconds)
        max_frames: Maximum number of frames to extract (None = all)

    Returns:
        tuple: (list of frame paths, video fps, total frames extracted)
    """
    os.makedirs(output_dir, exist_ok=True)

    vid = cv2.VideoCapture(video_path)
    video_fps = vid.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        print("Warning: Could not get video FPS, defaulting to 30")
        video_fps = 30

    # Calculate frame interval
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

            # Sample frames based on interval
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


def process_frame_batch(frame_paths, question, model, processor, show_visualizations=False):
    """
    Process a batch of frames for a single question

    Args:
        frame_paths: List of paths to frame images
        question: Question to ask about each frame
        model: The vision-language model
        processor: The model's processor
        show_visualizations: Whether to show heatmaps (False for efficiency)

    Returns:
        list: List of results for each frame containing predictions and attributions
    """
    device = next(model.parameters()).device
    default_cmap = setup_colormap()
    results = []

    for frame_idx, frame_path in enumerate(tqdm(frame_paths, desc=f"Processing frames")):
        # Clear cache before each frame
        torch.cuda.empty_cache()

        # Load image
        img = load_image_from_file(frame_path)

        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Answer briefly."},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_sizes = inputs['image_sizes']

        # Get model prediction - generate complete answer
        with torch.no_grad():
            # Generate complete answer
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
            )

            # Decode the generated answer
            prompt_length = input_ids.shape[1]
            generated_tokens = generated_ids[0, prompt_length:]
            predicted_answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Get logits for first token to compute attributions and alternatives
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=attention_mask
            )

        # Get logits and predictions for first token
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Get first token ID and confidence
        first_token_id = generated_tokens[0] if len(generated_tokens) > 0 else torch.argmax(logits, dim=-1).item()
        confidence_score = probs[0, first_token_id].item()

        # Get top-5 alternative first tokens
        top_probs, top_indices = torch.topk(probs[0], k=5)
        top_predictions = [
            (processor.tokenizer.decode(token_id).strip(), prob.item())
            for prob, token_id in zip(top_probs, top_indices)
        ]

        # Compute attributions using Integrated Gradients
        pixel_values_baseline = torch.zeros_like(pixel_values).to(device)
        q_tokens = processor.tokenizer(question, return_tensors="pt").to(device)
        input_ids_baseline = input_ids.clone().to(device, torch.int64)
        pad_token_id = processor.tokenizer.pad_token_id
        input_ids_baseline = replace_with_padding(
            input_ids_baseline,
            q_tokens["input_ids"],
            pad_token_id
        ).to(device)

        def custom_forward(pixel_values, input_ids, image_sizes=None, attention_mask=None):
            input_ids = input_ids.to(torch.int64)
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs.logits[:, -1, :]
            return next_token_logits

        ig = IntegratedGradients(custom_forward)
        attributions = ig.attribute(
            inputs=pixel_values,
            baselines=pixel_values_baseline,
            additional_forward_args=(input_ids, image_sizes, attention_mask),
            target=first_token_id,
            n_steps=5,
            internal_batch_size=1,
        )

        # Process attributions
        attributions_squeezed = attributions.squeeze(0)
        combined_patches = attributions_squeezed[0] + attributions_squeezed[1]
        attributions_img = combined_patches.permute(1, 2, 0).cpu().detach().numpy()

        # Store results
        result = {
            'frame_idx': frame_idx,
            'frame_path': frame_path,
            'prediction': predicted_answer,
            'confidence': confidence_score,
            'top_predictions': top_predictions,
            'attributions': attributions_img,
            'original_image': np.asarray(img)
        }
        results.append(result)

        # Optionally show visualization
        if show_visualizations:
            fig, _ = visualization.visualize_image_attr_multiple(
                attributions_img,
                result['original_image'],
                ["original_image", "heat_map"],
                ["all", "absolute_value"],
                titles=[f"Frame {frame_idx}", f"Attribution: '{predicted_answer}'"],
                cmap=default_cmap,
                show_colorbar=True,
                fig_size=(12, 6)
            )
            plt.tight_layout()
            plt.show()

        # Clear cache
        torch.cuda.empty_cache()

    return results


def create_temporal_timeline(results, question, save_path=None):
    """
    Create a timeline visualization showing predictions over time

    Args:
        results: List of result dictionaries from process_frame_batch
        question: The question that was asked
        save_path: Optional path to save the figure
    """
    frame_indices = [r['frame_idx'] for r in results]
    predictions = [r['prediction'] for r in results]
    confidences = [r['confidence'] for r in results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Predictions over time
    ax1.scatter(frame_indices, predictions, c=confidences, cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('Prediction', fontsize=12)
    ax1.set_title(f'Predictions Over Time: "{question}"', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Rotate y-axis labels if needed
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0, ha='right')

    # Plot 2: Confidence over time
    ax2.plot(frame_indices, confidences, marker='o', linewidth=2, markersize=6, color='steelblue')
    ax2.fill_between(frame_indices, confidences, alpha=0.3, color='steelblue')
    ax2.set_xlabel('Frame Index', fontsize=12)
    ax2.set_ylabel('Confidence Score', fontsize=12)
    ax2.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Timeline saved to {save_path}")

    plt.show()


def create_summary_report(results, question, video_fps, save_path=None):
    """
    Create a text summary report of the temporal analysis

    Args:
        results: List of result dictionaries from process_frame_batch
        question: The question that was asked
        video_fps: Original video FPS
        save_path: Optional path to save the report
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("TEMPORAL VIDEO ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append(f"\nQuestion: {question}")
    report_lines.append(f"Total Frames Analyzed: {len(results)}")
    report_lines.append(f"Video FPS: {video_fps:.2f}")
    report_lines.append("\n" + "-" * 70)
    report_lines.append("FRAME-BY-FRAME RESULTS")
    report_lines.append("-" * 70)

    for result in results:
        frame_idx = result['frame_idx']
        prediction = result['prediction']
        confidence = result['confidence']

        report_lines.append(f"\nFrame {frame_idx:04d}:")
        report_lines.append(f"  Prediction: '{prediction}' (Confidence: {confidence:.4f})")
        report_lines.append(f"  Top 5 predictions:")
        for i, (token, prob) in enumerate(result['top_predictions'], 1):
            report_lines.append(f"    {i}. '{token}' - {prob:.4f} ({prob*100:.2f}%)")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("=" * 70)

    # Calculate statistics
    all_predictions = [r['prediction'] for r in results]
    unique_predictions = list(set(all_predictions))
    avg_confidence = np.mean([r['confidence'] for r in results])

    report_lines.append(f"\nUnique Predictions: {len(unique_predictions)}")
    report_lines.append(f"Predictions: {', '.join(unique_predictions)}")
    report_lines.append(f"Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")

    # Most common prediction
    from collections import Counter
    prediction_counts = Counter(all_predictions)
    most_common = prediction_counts.most_common(1)[0]
    report_lines.append(f"\nMost Common Prediction: '{most_common[0]}' ({most_common[1]}/{len(results)} frames)")

    report = "\n".join(report_lines)
    print(report)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to {save_path}")

    return report


def temporal_vqa_interpret(
    video_path,
    questions,
    model,
    processor,
    fps_sample=1,
    max_frames=None,
    output_dir="video_analysis",
    show_frame_visualizations=False,
    show_timeline=True,
    save_results=True
):
    """
    Temporal Visual Question Answering with Integrated Gradients attribution

    Args:
        video_path: Path to the video file
        questions: List of questions to ask about the video
        model: The vision-language model
        processor: The model's processor
        fps_sample: Sample rate (1 = 1 frame per second, 0.5 = 1 frame per 2 seconds)
        max_frames: Maximum number of frames to process (None = all sampled frames)
        output_dir: Directory to save results
        show_frame_visualizations: Whether to show individual frame heatmaps (slow)
        show_timeline: Whether to show timeline visualizations
        save_results: Whether to save reports and visualizations
    """
    print(f"\n{'='*70}")
    print("TEMPORAL VIDEO ANALYSIS")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Questions: {len(questions)}")
    print(f"Sample Rate: {fps_sample} FPS")
    print(f"{'='*70}\n")

    # Create output directory
    if save_results:
        os.makedirs(output_dir, exist_ok=True)

    # Extract frames
    frames_dir = os.path.join(output_dir, "frames")
    frame_paths, video_fps, num_frames = extract_frames(
        video_path,
        output_dir=frames_dir,
        fps_sample=fps_sample,
        max_frames=max_frames
    )

    if len(frame_paths) == 0:
        print("Error: No frames extracted from video")
        return

    # Process each question
    all_results = {}

    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {q_idx}/{len(questions)}: {question}")
        print(f"{'='*70}\n")

        # Process frames
        results = process_frame_batch(
            frame_paths,
            question,
            model,
            processor,
            show_visualizations=show_frame_visualizations
        )

        all_results[question] = results

        # Create timeline visualization
        if show_timeline:
            timeline_path = os.path.join(output_dir, f"timeline_q{q_idx}.png") if save_results else None
            create_temporal_timeline(results, question, save_path=timeline_path)

        # Create summary report
        if save_results:
            report_path = os.path.join(output_dir, f"report_q{q_idx}.txt")
            create_summary_report(results, question, video_fps, save_path=report_path)
        else:
            create_summary_report(results, question, video_fps)

        # Clear cache between questions
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("✓ TEMPORAL ANALYSIS COMPLETE")
    if save_results:
        print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")

    return all_results
