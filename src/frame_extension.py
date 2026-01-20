"""Frame extension module for temporal video analysis with Integrated Gradients"""

import os
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
