"""
Ablation test: compare general-only vs conditional/general (normal) pipeline
on the first 200 samples (tc1-tc200) from CondVQA_3cols.csv.

Arm A — general:      force_simple=True  → all frames used, no LLM condition parsing
Arm B — conditional:  force_simple=False → normal pipeline (LLM decides type + filters frames)
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


def run_arm(
    arm_name,
    test_cases,
    video_dir,
    model,
    processor,
    text_model,
    text_tokenizer,
    output_dir,
    force_simple,
    fps_sample=1,
    confidence_threshold=0.5,
    aggregation_method='most_confident',
):
    arm_dir = os.path.join(output_dir, arm_name)
    os.makedirs(arm_dir, exist_ok=True)

    results = []

    for i, tc in enumerate(test_cases):
        video_path = os.path.join(video_dir, f"{tc['video_id']}.mp4")

        print(f"\n{'#'*70}")
        print(f"[{arm_name.upper()}] {i+1}/{len(test_cases)}: {tc['video_id']}")
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

        tc_out = os.path.join(arm_dir, tc['video_id'])
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
                force_simple=force_simple,
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
            print(f"\n  >> Prediction: {prediction}")
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

    # Per-arm CSV
    arm_csv = os.path.join(arm_dir, f'{arm_name}_results.csv')
    with open(arm_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['video_id', 'question', 'expected', 'prediction', 'confidence', 'status', 'error']
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"\nArm '{arm_name}' results saved to: {arm_csv}")

    return results


def print_summary(arm_name, results):
    ok = sum(1 for r in results if r['status'] == 'OK')
    err = sum(1 for r in results if r['status'] == 'ERROR')
    skip = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail = sum(1 for r in results if r['status'] == 'FAILED')
    print(f"  {arm_name:<20} total={len(results)} | OK={ok} | ERROR={err} | SKIPPED={skip} | FAILED={fail}")


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

    # Load models once, share across both arms
    if model is None or processor is None:
        print("Loading vision-language model...")
        model, processor = load_model()
    if text_model is None or text_tokenizer is None:
        print("Loading text model...")
        text_model, text_tokenizer = load_text_model()

    test_cases = load_test_cases(csv_path, max_samples=max_samples)
    print(f"\nLoaded {len(test_cases)} test cases (max_samples={max_samples})")

    shared_kwargs = dict(
        video_dir=video_dir,
        model=model,
        processor=processor,
        text_model=text_model,
        text_tokenizer=text_tokenizer,
        output_dir=output_dir,
        fps_sample=fps_sample,
        confidence_threshold=confidence_threshold,
        aggregation_method=aggregation_method,
    )

    # ── Arm A: general only ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ARM A: GENERAL ONLY (force_simple=True)")
    print(f"{'='*70}")
    general_results = run_arm('general', test_cases, force_simple=True, **shared_kwargs)

    # ── Arm B: normal (conditional/general) ─────────────────────────────────
    print(f"\n{'='*70}")
    print("ARM B: CONDITIONAL/GENERAL — NORMAL PIPELINE (force_simple=False)")
    print(f"{'='*70}")
    conditional_results = run_arm('conditional', test_cases, force_simple=False, **shared_kwargs)

    # ── Side-by-side comparison CSV ─────────────────────────────────────────
    by_id_general = {r['video_id']: r for r in general_results}
    by_id_cond = {r['video_id']: r for r in conditional_results}

    comparison_rows = []
    for tc in test_cases:
        vid = tc['video_id']
        g = by_id_general.get(vid, {})
        c = by_id_cond.get(vid, {})
        comparison_rows.append({
            'video_id': vid,
            'question': tc['question'],
            'expected': tc['expected'],
            'general_prediction': g.get('prediction', 'N/A'),
            'general_confidence': g.get('confidence', 0.0),
            'general_status': g.get('status', 'N/A'),
            'conditional_prediction': c.get('prediction', 'N/A'),
            'conditional_confidence': c.get('confidence', 0.0),
            'conditional_status': c.get('status', 'N/A'),
        })

    comparison_csv = os.path.join(output_dir, 'ablation_comparison.csv')
    with open(comparison_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ABLATION TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  Samples tested: {len(test_cases)}")
    print(f"  CSV: {csv_path}")
    print()
    print_summary("general (arm A)", general_results)
    print_summary("conditional (arm B)", conditional_results)
    print(f"\n  Comparison CSV: {comparison_csv}")
    print(f"{'='*70}\n")

    return general_results, conditional_results, comparison_rows


if __name__ == '__main__':
    run_ablation_test()
