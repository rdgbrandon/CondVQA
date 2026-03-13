"""Batch test runner for CondVQA - runs general question test cases (tc201-tc300)."""

import os
import csv
import torch
import gc
import sys
from datetime import datetime

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


def load_test_cases(csv_path):
    """Load test cases from CSV file."""
    test_cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                'video_id': row['Video ID'].strip(),
                'question': row['GQ'].strip(),
                'expected': row['Expected Answer'].strip()
            })
    return test_cases


def run_batch_test_general(
    csv_path=None,
    video_dir=None,
    model=None,
    processor=None,
    text_model=None,
    text_tokenizer=None,
    fps_sample=1,
    confidence_threshold=0.5,
    aggregation_method='most_confident',
    show_visualizations=False,
    save_results=True,
    output_dir=None
):
    """
    Run all general question test cases from CSV and compare predictions to expected answers.

    Args:
        csv_path: Path to CondVQA_3cols_general.csv
        video_dir: Path to folder containing tc*.mp4 files
        model: Vision-language model (loaded if None)
        processor: Model processor (loaded if None)
        text_model: Text LLM for parsing (loaded if None)
        text_tokenizer: Text tokenizer (loaded if None)
        fps_sample: Frames per second to sample
        confidence_threshold: Confidence threshold for frame matching
        aggregation_method: How to aggregate multi-frame results
        show_visualizations: Whether to show plots (disable for batch)
        save_results: Whether to save per-test-case results
        output_dir: Directory for all output

    Returns:
        list of result dicts with predictions and comparisons
    """
    # Default paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        csv_path = os.path.join(script_dir, 'CondVQA_3cols_general.csv')
    if video_dir is None:
        video_dir = os.path.join(script_dir, 'testcases')
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'general_results')

    os.makedirs(output_dir, exist_ok=True)

    # Load models if not provided
    if model is None or processor is None:
        print("Loading vision-language model...")
        model, processor = load_model()

    if text_model is None or text_tokenizer is None:
        print("Loading text model for question parsing...")
        text_model, text_tokenizer = load_text_model()

    # Load test cases
    test_cases = load_test_cases(csv_path)
    print(f"\nLoaded {len(test_cases)} test cases from {csv_path}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}\n")

    results = []

    for i, tc in enumerate(test_cases):
        video_file = f"{tc['video_id']}.mp4"
        video_path = os.path.join(video_dir, video_file)

        print(f"\n{'#'*70}")
        print(f"TEST CASE {i+1}/{len(test_cases)}: {tc['video_id']}")
        print(f"Question: {tc['question']}")
        print(f"Expected: {tc['expected']}")
        print(f"{'#'*70}")

        if not os.path.exists(video_path):
            print(f"  SKIPPED - video not found: {video_path}")
            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': 'N/A',
                'confidence': 0.0,
                'status': 'SKIPPED',
                'error': 'Video file not found'
            })
            continue

        gc.collect()
        torch.cuda.empty_cache()

        tc_output_dir = os.path.join(output_dir, tc['video_id']) if save_results else None
        if tc_output_dir:
            os.makedirs(tc_output_dir, exist_ok=True)

        try:
            general_results = conditional_query_vqa_interpret(
                video_path=video_path,
                questions=[tc['question']],
                model=model,
                processor=processor,
                text_model=text_model,
                text_tokenizer=text_tokenizer,
                fps_sample=fps_sample,
                confidence_threshold=confidence_threshold,
                aggregation_method=aggregation_method,
                show_visualizations=show_visualizations,
                save_results=save_results,
                output_dir=tc_output_dir if tc_output_dir else 'general_query_analysis',
                include_text_attribution=False
            )

            # Extract result for this question
            question_key = tc['question']
            if question_key in general_results:
                result = general_results[question_key]
                if 'error' in result:
                    prediction = 'ERROR'
                    confidence = 0.0
                    error_msg = result['error']
                else:
                    prediction = result.get('prediction', 'N/A')
                    confidence = result.get('confidence', 0.0)
                    error_msg = None
            else:
                prediction = 'NO_RESULT'
                confidence = 0.0
                error_msg = 'Question key not found in results'

            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': prediction,
                'confidence': confidence,
                'status': 'OK' if error_msg is None else 'ERROR',
                'error': error_msg
            })

            print(f"\n  >> Prediction: {prediction}")
            print(f"  >> Expected:   {tc['expected']}")
            print(f"  >> Confidence: {confidence:.2%}")

        except Exception as e:
            print(f"  FAILED: {str(e)}")
            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': 'FAILED',
                'confidence': 0.0,
                'status': 'FAILED',
                'error': str(e)
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary Report ──
    print(f"\n{'='*70}")
    print("GENERAL BATCH TEST RESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Video ID':<10} {'Status':<10} {'Confidence':<12} {'Prediction':<30} {'Expected':<30}")
    print("-" * 92)

    ok_count = 0
    error_count = 0
    skip_count = 0
    fail_count = 0

    for r in results:
        status = r['status']
        if status == 'OK':
            ok_count += 1
        elif status == 'ERROR':
            error_count += 1
        elif status == 'SKIPPED':
            skip_count += 1
        else:
            fail_count += 1

        pred_display = str(r['prediction'])[:28]
        exp_display = str(r['expected'])[:28]
        print(f"{r['video_id']:<10} {status:<10} {r['confidence']:<12.2%} {pred_display:<30} {exp_display:<30}")

    print(f"\n{'─'*70}")
    print(f"Total: {len(results)} | OK: {ok_count} | Errors: {error_count} | Skipped: {skip_count} | Failed: {fail_count}")
    print(f"{'─'*70}\n")

    # Save results to CSV
    results_csv_path = os.path.join(output_dir, 'general_results.csv')
    with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'question', 'expected', 'prediction', 'confidence', 'status', 'error'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {results_csv_path}")

    return results


if __name__ == '__main__':
    run_batch_test_general()
