"""
MSRVTT-QA Benchmark Runner for CondVLM
- Loads QA annotations from official JSON (xudejing/video-question-answering format)
- Loads videos from a local directory
- Randomly samples 200 test QA pairs (seed=42)
- Per-question-type accuracy breakdown: what / who / how / when / where / other
"""

import os
import sys
import csv
import json
import random
import gc
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QTYPES = ['what', 'who', 'how', 'when', 'where']


def get_question_type(question):
    first = question.lower().strip().split()[0].rstrip('?')
    return first if first in QTYPES else 'other'


def is_correct(prediction, expected):
    """Normalised exact-match with substring fallback — standard open VQA metric."""
    pred = prediction.lower().strip().rstrip('.,!?').strip()
    exp  = expected.lower().strip().rstrip('.,!?').strip()
    if not exp:
        return False
    return pred == exp or exp in pred or pred in exp


def run_msrvtt_benchmark(
    qa_json_path,
    video_dir,
    model=None,
    processor=None,
    text_model=None,
    text_tokenizer=None,
    output_dir=None,
    max_samples=200,
    seed=42,
    fps_sample=1,
    confidence_threshold=0.5,
    aggregation_method='most_confident',
    save_results=True,
    show_visualizations=False,
):
    """
    Run CondVLM against a random 200-sample subset of MSRVTT-QA test split.

    Args:
        qa_json_path:  Path to test_qa.json (list of {video_id, question, answer})
        video_dir:     Directory containing videoXXXX.mp4 files
        model/processor/text_model/text_tokenizer: pre-loaded models (auto-loaded if None)
        output_dir:    Where to save results
        max_samples:   Number of QA pairs to evaluate (default 200)
        seed:          Random seed for sampling (default 42)
        fps_sample:    Frames per second to extract
        confidence_threshold: Frame-condition confidence cutoff
        aggregation_method:   'most_confident' | 'consensus' | 'average'
        save_results:  Write per-sample and summary CSVs
        show_visualizations: Show IG plots (disable for batch)

    Returns:
        list of result dicts
    """
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(SCRIPT_DIR, f'msrvtt_results_{ts}')
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load & sample annotations ──────────────────────────────────────────
    print(f"\n[1/3] Loading MSRVTT-QA annotations from {qa_json_path} ...")
    with open(qa_json_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"  Loaded {len(qa_data)} total test QA pairs")

    random.seed(seed)
    sample = random.sample(qa_data, min(max_samples, len(qa_data)))
    print(f"  Sampled {len(sample)} pairs (seed={seed})")

    # ── 2. Load models ─────────────────────────────────────────────────────────
    print("\n[2/3] Loading models...")
    if model is None or processor is None:
        print("  Loading vision-language model...")
        model, processor = load_model()
    if text_model is None or text_tokenizer is None:
        print("  Loading text model...")
        text_model, text_tokenizer = load_text_model()

    # ── 3. Run inference ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RUNNING MSRVTT-QA BENCHMARK")
    print(f"{'='*70}\n")

    results = []

    for i, qa in enumerate(sample):
        video_id = str(qa['video_id'])
        question = qa['question']
        expected = str(qa['answer'])
        qtype    = get_question_type(question)

        # MSRVTT videos are named video7010.mp4, video7011.mp4, etc.
        video_path = None
        for name in [f"video{video_id}.mp4", f"{video_id}.mp4"]:
            p = os.path.join(video_dir, name)
            if os.path.exists(p):
                video_path = p
                break

        print(f"\n{'#'*70}")
        print(f"[{i+1}/{len(sample)}] video_id={video_id}  type={qtype}")
        print(f"  Q:        {question}")
        print(f"  Expected: {expected}")
        print(f"{'#'*70}")

        if video_path is None:
            print(f"  SKIPPED — video not found in {video_dir}")
            results.append({
                'video_id': video_id, 'question_type': qtype,
                'question': question, 'expected': expected,
                'prediction': 'N/A', 'confidence': 0.0,
                'correct': False, 'status': 'SKIPPED', 'error': 'Video not found',
            })
            continue

        gc.collect()
        torch.cuda.empty_cache()

        tc_out = os.path.join(output_dir, video_id) if save_results else 'msrvtt_analysis'
        os.makedirs(tc_out, exist_ok=True)

        try:
            raw = conditional_query_vqa_interpret(
                video_path=video_path,
                questions=[question],
                model=model,
                processor=processor,
                text_model=text_model,
                text_tokenizer=text_tokenizer,
                fps_sample=fps_sample,
                confidence_threshold=confidence_threshold,
                aggregation_method=aggregation_method,
                show_visualizations=show_visualizations,
                save_results=save_results,
                output_dir=tc_out,
                include_text_attribution=False,
            )

            if question in raw:
                r = raw[question]
                if 'error' in r:
                    prediction, confidence, err = 'ERROR', 0.0, r['error']
                else:
                    prediction = r.get('prediction', 'N/A')
                    confidence = r.get('confidence', 0.0)
                    err = None
            else:
                prediction, confidence, err = 'NO_RESULT', 0.0, 'Key not found in results'

            correct = is_correct(prediction, expected) if err is None else False

            results.append({
                'video_id': video_id, 'question_type': qtype,
                'question': question, 'expected': expected,
                'prediction': prediction, 'confidence': confidence,
                'correct': correct,
                'status': 'OK' if err is None else 'ERROR',
                'error': err,
            })
            print(f"  >> Prediction : {prediction}  |  Correct: {correct}")
            print(f"  >> Confidence : {confidence:.2%}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'video_id': video_id, 'question_type': qtype,
                'question': question, 'expected': expected,
                'prediction': 'FAILED', 'confidence': 0.0,
                'correct': False, 'status': 'FAILED', 'error': str(e),
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("MSRVTT-QA BENCHMARK — RESULTS BY QUESTION TYPE")
    print(f"{'='*70}")
    print(f"{'Type':<10} {'Total':<8} {'OK':<6} {'Correct':<10} {'Accuracy'}")
    print(f"{'-'*10} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")

    for qt in QTYPES + ['other']:
        subset    = [r for r in results if r['question_type'] == qt]
        if not subset:
            continue
        ok_sub    = [r for r in subset if r['status'] == 'OK']
        correct   = sum(1 for r in ok_sub if r['correct'])
        acc       = correct / len(ok_sub) if ok_sub else 0.0
        print(f"{qt:<10} {len(subset):<8} {len(ok_sub):<6} {correct:<10} {acc:.2%}")

    ok_all      = [r for r in results if r['status'] == 'OK']
    correct_all = sum(1 for r in ok_all if r['correct'])
    overall_acc = correct_all / len(ok_all) if ok_all else 0.0
    ok_cnt   = len(ok_all)
    err_cnt  = sum(1 for r in results if r['status'] == 'ERROR')
    skip_cnt = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail_cnt = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"{'─'*54}")
    print(f"{'OVERALL':<10} {len(results):<8} {ok_cnt:<6} {correct_all:<10} {overall_acc:.2%}")
    print(f"\nTotal: {len(results)} | OK: {ok_cnt} | ERROR: {err_cnt} | SKIPPED: {skip_cnt} | FAILED: {fail_cnt}")

    if save_results:
        csv_path = os.path.join(output_dir, f'msrvtt_results_{datetime.now():%Y%m%d_%H%M%S}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'video_id', 'question_type', 'question', 'expected',
                'prediction', 'confidence', 'correct', 'status', 'error',
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")
        print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--qa_json',    required=True, help='Path to test_qa.json')
    p.add_argument('--video_dir',  required=True, help='Directory with videoXXXX.mp4 files')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--max_samples', type=int, default=200)
    args = p.parse_args()
    run_msrvtt_benchmark(qa_json_path=args.qa_json, video_dir=args.video_dir,
                         output_dir=args.output_dir, max_samples=args.max_samples)
