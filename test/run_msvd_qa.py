"""Batch test runner for MSVD-QA benchmark dataset using CondVLM."""

import os
import csv
import json
import torch
import gc
import sys
from datetime import datetime

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


def load_msvd_qa(json_path, split='test'):
    """
    Load MSVD-QA test cases from the official JSON annotation file.

    MSVD-QA JSON format:
        [ {"id": 0, "video_id": "vid1", "question": "what is ...", "answer": "..."}, ... ]

    Args:
        json_path: Path to test_qa.json (or val_qa.json)
        split: 'test' or 'val'

    Returns:
        list of dicts with keys: video_id, question, expected
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    test_cases = []
    for item in data:
        test_cases.append({
            'video_id': item['video_id'],          # e.g. "vid1"
            'question': item['question'].strip(),
            'expected': str(item['answer']).strip()
        })
    return test_cases


def normalize_answer(answer):
    """Lowercase and strip punctuation for loose matching."""
    if answer is None:
        return ''
    return answer.lower().strip().rstrip('.')


def run_msvd_qa(
    json_path,
    video_dir,
    model=None,
    processor=None,
    text_model=None,
    text_tokenizer=None,
    fps_sample=1,
    confidence_threshold=0.5,
    aggregation_method='most_confident',
    show_visualizations=False,
    save_results=True,
    output_dir=None,
    max_samples=None
):
    """
    Run CondVLM on the MSVD-QA benchmark and compare against ground truth.

    Args:
        json_path:             Path to MSVD-QA test_qa.json
        video_dir:             Path to folder containing *.avi video files
        model:                 Vision-language model (loaded if None)
        processor:             Model processor (loaded if None)
        text_model:            Text LLM for parsing (loaded if None)
        text_tokenizer:        Text tokenizer (loaded if None)
        fps_sample:            Frames per second to sample from each video
        confidence_threshold:  Confidence threshold for frame matching
        aggregation_method:    How to aggregate multi-frame results
        show_visualizations:   Whether to show plots (disable for batch runs)
        save_results:          Whether to save per-sample output files
        output_dir:            Directory for all output files
        max_samples:           Limit number of samples (None = run all)

    Returns:
        list of result dicts, and prints accuracy summary
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'msvd_qa_results')
    os.makedirs(output_dir, exist_ok=True)

    # Load models if not provided
    if model is None or processor is None:
        print("Loading vision-language model...")
        model, processor = load_model()

    if text_model is None or text_tokenizer is None:
        print("Loading text model for question parsing...")
        text_model, text_tokenizer = load_text_model()

    test_cases = load_msvd_qa(json_path)
    if max_samples is not None:
        test_cases = test_cases[:max_samples]

    print(f"\nLoaded {len(test_cases)} MSVD-QA test cases")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}\n")

    results = []
    exact_matches = 0
    skipped = 0
    errors = 0

    for i, tc in enumerate(test_cases):
        # MSVD videos are .avi files named by video_id (e.g. vid1.avi)
        video_file = f"{tc['video_id']}.avi"
        video_path = os.path.join(video_dir, video_file)

        print(f"\n{'#'*70}")
        print(f"[{i+1}/{len(test_cases)}] {tc['video_id']}")
        print(f"Q: {tc['question']}")
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
                'exact_match': False,
                'status': 'SKIPPED',
                'error': 'Video file not found'
            })
            skipped += 1
            continue

        gc.collect()
        torch.cuda.empty_cache()

        tc_output_dir = os.path.join(output_dir, tc['video_id']) if save_results else None
        if tc_output_dir:
            os.makedirs(tc_output_dir, exist_ok=True)

        try:
            raw_results = conditional_query_vqa_interpret(
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
                output_dir=tc_output_dir if tc_output_dir else 'msvd_qa_analysis',
                include_text_attribution=False
            )

            question_key = tc['question']
            if question_key in raw_results:
                result = raw_results[question_key]
                if 'error' in result:
                    prediction = 'ERROR'
                    confidence = 0.0
                    error_msg = result['error']
                    errors += 1
                else:
                    prediction = result.get('prediction', 'N/A')
                    confidence = result.get('confidence', 0.0)
                    error_msg = None
            else:
                prediction = 'NO_RESULT'
                confidence = 0.0
                error_msg = 'Question key not found in results'
                errors += 1

            # Exact match (case-insensitive, stripped)
            match = normalize_answer(prediction) == normalize_answer(tc['expected'])
            if match:
                exact_matches += 1

            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': prediction,
                'confidence': confidence,
                'exact_match': match,
                'status': 'OK' if error_msg is None else 'ERROR',
                'error': error_msg
            })

            print(f"\n  >> Prediction: {prediction}")
            print(f"  >> Expected:   {tc['expected']}")
            print(f"  >> Match:      {'YES' if match else 'NO'}")
            print(f"  >> Confidence: {confidence:.2%}")

        except Exception as e:
            print(f"  FAILED: {str(e)}")
            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': 'FAILED',
                'confidence': 0.0,
                'exact_match': False,
                'status': 'FAILED',
                'error': str(e)
            })
            errors += 1

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──
    total = len(results)
    ok_count = total - skipped - errors
    accuracy = (exact_matches / ok_count * 100) if ok_count > 0 else 0.0

    print(f"\n{'='*70}")
    print("MSVD-QA BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Total samples : {total}")
    print(f"Ran           : {ok_count}")
    print(f"Skipped       : {skipped}")
    print(f"Errors        : {errors}")
    print(f"Exact matches : {exact_matches}")
    print(f"Accuracy      : {accuracy:.2f}%")
    print(f"{'='*70}\n")

    # Save CSV
    results_csv = os.path.join(output_dir, 'msvd_qa_results.csv')
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'video_id', 'question', 'expected', 'prediction',
            'confidence', 'exact_match', 'status', 'error'
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {results_csv}")

    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total': total,
        'ran': ok_count,
        'skipped': skipped,
        'errors': errors,
        'exact_matches': exact_matches,
        'accuracy': round(accuracy, 4)
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return results


if __name__ == '__main__':
    run_msvd_qa(
        json_path='test_qa.json',
        video_dir='YouTubeClips',
    )
