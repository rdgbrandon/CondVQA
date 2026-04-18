"""
Ablation test: run 200 samples (tc1-tc200) with general-only pipeline (force_simple=True).
All frames are used — no LLM condition parsing.
Prints a clean per-sample results table at the end for easy copying.
"""

import os
import csv
import torch
import gc
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_test_cases(csv_path, max_samples=200):
    test_cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                'video_id': row['Video ID'].strip(),
                'question': row['CQ'].strip(),
                'expected': row['Expected Answer'].strip()
            })
            if max_samples and len(test_cases) >= max_samples:
                break
    return test_cases


def run_ablation_test(
    csv_path=None,
    video_dir=None,
    output_dir=None,
    max_samples=200,
    fps_sample=1,
    confidence_threshold=0.5,
    aggregation_method='most_confident',
    model=None,
    processor=None,
    text_model=None,
    text_tokenizer=None,
):
    if csv_path is None:
        csv_path = os.path.join(SCRIPT_DIR, 'CondVQA_3cols.csv')
    if video_dir is None:
        video_dir = os.path.join(SCRIPT_DIR, 'testcases')
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(SCRIPT_DIR, f'ablation_results_{ts}')

    os.makedirs(output_dir, exist_ok=True)

    if model is None or processor is None:
        print("Loading vision-language model...")
        model, processor = load_model()
    if text_model is None or text_tokenizer is None:
        print("Loading text model...")
        text_model, text_tokenizer = load_text_model()

    test_cases = load_test_cases(csv_path, max_samples=max_samples)
    print(f"\nLoaded {len(test_cases)} test cases — GENERAL ONLY (force_simple=True)\n")

    results = []

    for i, tc in enumerate(test_cases):
        video_path = os.path.join(video_dir, f"{tc['video_id']}.mp4")

        print(f"\n{'#'*70}")
        print(f"{i+1}/{len(test_cases)}: {tc['video_id']}")
        print(f"Q: {tc['question']}")
        print(f"Expected: {tc['expected']}")
        print(f"{'#'*70}")

        if not os.path.exists(video_path):
            print(f"  SKIPPED — video not found: {video_path}")
            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': 'N/A',
                'confidence': 0.0,
                'status': 'SKIPPED',
                'error': 'Video file not found',
            })
            continue

        gc.collect()
        torch.cuda.empty_cache()

        tc_out = os.path.join(output_dir, tc['video_id'])
        os.makedirs(tc_out, exist_ok=True)

        try:
            raw = conditional_query_vqa_interpret(
                video_path=video_path,
                questions=[tc['question']],
                model=model,
                processor=processor,
                text_model=text_model,
                text_tokenizer=text_tokenizer,
                fps_sample=fps_sample,
                confidence_threshold=confidence_threshold,
                aggregation_method=aggregation_method,
                show_visualizations=False,
                save_results=True,
                output_dir=tc_out,
                include_text_attribution=False,
                force_simple=True,
            )

            qkey = tc['question']
            if qkey in raw:
                r = raw[qkey]
                if 'error' in r:
                    prediction, confidence, err = 'ERROR', 0.0, r['error']
                else:
                    prediction = r.get('prediction', 'N/A')
                    confidence = r.get('confidence', 0.0)
                    err = None
            else:
                prediction, confidence, err = 'NO_RESULT', 0.0, 'Key not found'

            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': prediction,
                'confidence': confidence,
                'status': 'OK' if err is None else 'ERROR',
                'error': err,
            })
            print(f"  >> Prediction: {prediction}")
            print(f"  >> Expected:   {tc['expected']}")
            print(f"  >> Confidence: {confidence:.2%}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': 'FAILED',
                'confidence': 0.0,
                'status': 'FAILED',
                'error': str(e),
            })

        gc.collect()
        torch.cuda.empty_cache()

    # Save CSV
    results_csv = os.path.join(output_dir, 'ablation_general_results.csv')
    with open(results_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['video_id', 'question', 'expected', 'prediction', 'confidence', 'status', 'error']
        )
        writer.writeheader()
        writer.writerows(results)

    # ── Final results table ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION TEST — GENERAL ONLY — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'#':<6} {'Video ID':<10} {'Status':<10} {'Conf':<8} {'Prediction':<30} {'Expected'}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*30} {'-'*30}")
    for i, r in enumerate(results, 1):
        print(
            f"{i:<6} {r['video_id']:<10} {r['status']:<10} "
            f"{r['confidence']:<8.2%} {str(r['prediction']):<30} {r['expected']}"
        )

    ok = sum(1 for r in results if r['status'] == 'OK')
    err = sum(1 for r in results if r['status'] == 'ERROR')
    skip = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"\n{'─'*70}")
    print(f"Total: {len(results)} | OK: {ok} | ERROR: {err} | SKIPPED: {skip} | FAILED: {fail}")
    print(f"Results saved to: {results_csv}")
    print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    run_ablation_test()
