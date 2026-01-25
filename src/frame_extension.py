"""Frame extension module for temporal video analysis with Integrated Gradients"""

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


def parse_temporal_reference(question, video_duration):
    """
    Parse temporal references from natural language questions.

    Args:
        question: Question string potentially containing temporal references
        video_duration: Total video duration in seconds

    Returns:
        tuple: (start_sec, end_sec, core_question)
            - start_sec: Start time in seconds
            - end_sec: End time in seconds
            - core_question: Question with temporal reference removed

    Examples:
        "What's at 4 seconds?" -> (4.0, 4.0, "What's happening?")
        "What happens in the first 5 seconds?" -> (0.0, 5.0, "What happens?")
        "What's between 2 and 6 seconds?" -> (2.0, 6.0, "What's happening?")
    """
    original_question = question.lower()

    # Pattern 1: "at X second(s)" or "at the Xth second"
    match = re.search(r'at (?:the )?(\d+(?:\.\d+)?)\s*(?:th|st|nd|rd)?\s*second', original_question)
    if match:
        time = float(match.group(1))
        core_q = re.sub(r'at (?:the )?(\d+(?:\.\d+)?)\s*(?:th|st|nd|rd)?\s*second', '', original_question).strip()
        core_q = core_q if core_q else "What is happening?"
        return (time, time, core_q)

    # Pattern 2: "in the Xth second"
    match = re.search(r'in the (\d+)(?:th|st|nd|rd) second', original_question)
    if match:
        time = float(match.group(1))
        core_q = re.sub(r'in the (\d+)(?:th|st|nd|rd) second', '', original_question).strip()
        core_q = core_q if core_q else "What is happening?"
        return (time, time, core_q)

    # Pattern 3: "first X second(s)" or "last X second(s)"
    match = re.search(r'(?:first|initial)\s+(\d+(?:\.\d+)?)\s*second', original_question)
    if match:
        duration = float(match.group(1))
        core_q = re.sub(r'(?:in the |during the )?(?:first|initial)\s+(\d+(?:\.\d+)?)\s*second(?:s)?', '', original_question).strip()
        core_q = core_q if core_q else "What happens?"
        return (0.0, duration, core_q)

    match = re.search(r'last\s+(\d+(?:\.\d+)?)\s*second', original_question)
    if match:
        duration = float(match.group(1))
        start = max(0, video_duration - duration)
        core_q = re.sub(r'(?:in the |during the )?last\s+(\d+(?:\.\d+)?)\s*second(?:s)?', '', original_question).strip()
        core_q = core_q if core_q else "What happens?"
        return (start, video_duration, core_q)

    # Pattern 4: "between X and Y seconds"
    match = re.search(r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*second', original_question)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        core_q = re.sub(r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s*second(?:s)?', '', original_question).strip()
        core_q = core_q if core_q else "What is happening?"
        return (start, end, core_q)

    # Pattern 5: "from X to Y seconds"
    match = re.search(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*second', original_question)
    if match:
        start = float(match.group(1))
        end = float(match.group(2))
        core_q = re.sub(r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*second(?:s)?', '', original_question).strip()
        core_q = core_q if core_q else "What is happening?"
        return (start, end, core_q)

    # Pattern 6: "middle", "beginning", "end"
    if 'beginning' in original_question or 'start' in original_question:
        duration = min(5.0, video_duration * 0.2)  # First 5s or 20% of video
        core_q = re.sub(r'(?:at the |in the |during the )?(?:beginning|start)(?: of the video)?', '', original_question).strip()
        core_q = core_q if core_q else "What is happening?"
        return (0.0, duration, core_q)

    if 'middle' in original_question:
        mid = video_duration / 2
        window = min(5.0, video_duration * 0.1)  # 5s window or 10% of video
        core_q = re.sub(r'(?:at the |in the |during the )?middle(?: of the video)?', '', original_question).strip()
        core_q = core_q if core_q else "What is happening?"
        return (mid - window/2, mid + window/2, core_q)

    if 'end' in original_question or 'ending' in original_question:
        duration = min(5.0, video_duration * 0.2)  # Last 5s or 20% of video
        start = max(0, video_duration - duration)
        core_q = re.sub(r'(?:at the |in the |during the )?(?:end|ending)(?: of the video)?', '', original_question).strip()
        core_q = core_q if core_q else "What is happening?"
        return (start, video_duration, core_q)

    # No temporal reference found - analyze entire video
    return (0.0, video_duration, question)


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


def create_temporal_timeline(results, question, save_path=None):
    """Create a timeline visualization showing predictions over time."""
    frame_indices = [r['frame_idx'] for r in results]
    predictions = [r['prediction'] for r in results]
    confidences = [r['confidence'] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ax1.scatter(frame_indices, predictions, c=confidences, cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('Frame Index', fontsize=12)
    ax1.set_ylabel('Prediction', fontsize=12)
    ax1.set_title(f'Predictions Over Time: "{question}"', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0, ha='right')

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
    """Create a text summary report of the temporal analysis."""
    lines = [
        "=" * 70,
        "TEMPORAL VIDEO ANALYSIS REPORT",
        "=" * 70,
        f"\nQuestion: {question}",
        f"Total Frames Analyzed: {len(results)}",
        f"Video FPS: {video_fps:.2f}",
        "\n" + "-" * 70,
        "FRAME-BY-FRAME RESULTS",
        "-" * 70
    ]

    for r in results:
        lines.append(f"\nFrame {r['frame_idx']:04d}:")
        lines.append(f"  Prediction: '{r['prediction']}' (Confidence: {r['confidence']:.4f})")
        lines.append("  Top 5 predictions:")
        for i, (token, prob) in enumerate(r['top_predictions'], 1):
            lines.append(f"    {i}. '{token}' - {prob:.4f} ({prob*100:.2f}%)")

    all_predictions = [r['prediction'] for r in results]
    unique_predictions = list(set(all_predictions))
    avg_confidence = np.mean([r['confidence'] for r in results])
    prediction_counts = Counter(all_predictions)
    most_common = prediction_counts.most_common(1)[0]

    lines.extend([
        "\n" + "=" * 70,
        "SUMMARY STATISTICS",
        "=" * 70,
        f"\nUnique Predictions: {len(unique_predictions)}",
        f"Predictions: {', '.join(unique_predictions)}",
        f"Average Confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)",
        f"\nMost Common Prediction: '{most_common[0]}' ({most_common[1]}/{len(results)} frames)"
    ])

    report = "\n".join(lines)
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
    save_results=True,
    include_text_attribution=False
):
    """Temporal Visual Question Answering with Integrated Gradients attribution."""
    print(f"\n{'='*70}")
    print("TEMPORAL VIDEO ANALYSIS")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Questions: {len(questions)}")
    print(f"Sample Rate: {fps_sample} FPS")
    print(f"Text Attribution: {'Enabled' if include_text_attribution else 'Disabled'}")
    print(f"{'='*70}\n")

    if save_results:
        os.makedirs(output_dir, exist_ok=True)

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

    all_results = {}

    for q_idx, question in enumerate(questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {q_idx}/{len(questions)}: {question}")
        print(f"{'='*70}\n")

        results = process_frame_batch(
            frame_paths,
            question,
            model,
            processor,
            show_visualizations=show_frame_visualizations,
            include_text_attribution=include_text_attribution
        )

        all_results[question] = results

        if show_timeline:
            timeline_path = os.path.join(output_dir, f"timeline_q{q_idx}.png") if save_results else None
            create_temporal_timeline(results, question, save_path=timeline_path)

        if save_results:
            report_path = os.path.join(output_dir, f"report_q{q_idx}.txt")
            create_summary_report(results, question, video_fps, save_path=report_path)
        else:
            create_summary_report(results, question, video_fps)

        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("✓ TEMPORAL ANALYSIS COMPLETE")
    if save_results:
        print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")

    return all_results


def temporal_query_vqa_interpret(
    video_path,
    temporal_questions,
    model,
    processor,
    fps_sample=1,
    output_dir="temporal_query_analysis",
    aggregation_method='most_confident',
    show_visualizations=True,
    save_results=True,
    include_text_attribution=False
):
    """
    Answer temporal queries by extracting relevant frames and computing attributions.

    This function parses questions containing temporal references (e.g., "What's at 4 seconds?")
    and analyzes only the relevant frames for each question.

    Args:
        video_path: Path to video file
        temporal_questions: List of questions with temporal references
            Examples:
                - "What is happening at 4 seconds?"
                - "What's in the first 5 seconds?"
                - "What occurs between 2 and 6 seconds?"
                - "What's at the beginning of the video?"
        model: Vision-language model
        processor: Model's processor
        fps_sample: Frames per second to sample from video (default: 1)
        output_dir: Directory to save results
        aggregation_method: How to combine multi-frame results
            - 'most_confident': Use frame with highest confidence
            - 'consensus': Use most common prediction
            - 'average': Average attributions and confidences
        show_visualizations: Whether to display visualizations
        save_results: Whether to save results to disk
        include_text_attribution: Whether to compute text attributions

    Returns:
        dict: Results for each question with temporal analysis

    Example:
        >>> results = temporal_query_vqa_interpret(
        ...     "video.mp4",
        ...     ["What's at 4 seconds?", "What happens in the first 10 seconds?"],
        ...     model, processor
        ... )
    """
    print(f"\n{'='*70}")
    print("TEMPORAL QUERY VIDEO ANALYSIS")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Questions: {len(temporal_questions)}")
    print(f"Sample Rate: {fps_sample} FPS")
    print(f"Aggregation Method: {aggregation_method}")
    print(f"Text Attribution: {'Enabled' if include_text_attribution else 'Disabled'}")
    print(f"{'='*70}\n")

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

    for q_idx, question in enumerate(temporal_questions, 1):
        print(f"\n{'='*70}")
        print(f"Question {q_idx}/{len(temporal_questions)}: {question}")
        print(f"{'='*70}\n")

        # Parse temporal reference
        start_sec, end_sec, core_question = parse_temporal_reference(question, video_duration)

        print(f"Temporal Range: {start_sec:.2f}s - {end_sec:.2f}s")
        print(f"Core Question: {core_question}")

        # Get relevant frames
        relevant_frames = get_frames_for_timerange(frame_paths, fps_sample, start_sec, end_sec)

        print(f"Analyzing {len(relevant_frames)} frame(s)...")

        # Process frames
        frame_results = process_frame_batch(
            relevant_frames,
            core_question,
            model,
            processor,
            show_visualizations=False,  # We'll show aggregated result
            include_text_attribution=include_text_attribution
        )

        # Aggregate results
        aggregated_result = aggregate_temporal_results(frame_results, method=aggregation_method)

        if aggregated_result:
            print(f"\n{'─'*70}")
            print(f"AGGREGATED RESULT ({aggregation_method})")
            print(f"{'─'*70}")
            print(f"Prediction: '{aggregated_result['prediction']}'")
            print(f"Confidence: {aggregated_result['confidence']:.4f} ({aggregated_result['confidence']*100:.2f}%)")

            if 'consensus_percentage' in aggregated_result:
                print(f"Consensus: {aggregated_result['consensus_count']}/{len(frame_results)} frames ({aggregated_result['consensus_percentage']:.1f}%)")

            print(f"\nTop predictions from aggregated result:")
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
                        f"Frame at {start_sec:.1f}s-{end_sec:.1f}s",
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
                    f.write("TEMPORAL QUERY ANALYSIS REPORT\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"Question: {question}\n")
                    f.write(f"Temporal Range: {start_sec:.2f}s - {end_sec:.2f}s\n")
                    f.write(f"Core Question: {core_question}\n")
                    f.write(f"Frames Analyzed: {len(frame_results)}\n")
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
                        frame_time = (start_sec + i * (1.0 / fps_sample))
                        f.write(f"Frame {i+1} (at {frame_time:.2f}s):\n")
                        f.write(f"  Prediction: '{r['prediction']}'\n")
                        f.write(f"  Confidence: {r['confidence']:.4f}\n\n")

                print(f"✓ Saved report to {report_path}")

            aggregated_result['temporal_info'] = {
                'start_sec': start_sec,
                'end_sec': end_sec,
                'core_question': core_question,
                'original_question': question,
                'num_frames': len(frame_results),
                'individual_results': frame_results
            }

            all_results[question] = aggregated_result

        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print("✓ TEMPORAL QUERY ANALYSIS COMPLETE")
    if save_results:
        print(f"Results saved to: {output_dir}/")
    print(f"{'='*70}\n")

    return all_results
