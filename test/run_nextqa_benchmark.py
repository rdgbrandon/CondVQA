"""
NExT-QA Benchmark Runner for CondVLM  — Descriptive questions only
- Loads QA annotations from official CSV (doc-doc/NExT-QA format)
- Filters to descriptive question types (type starts with 'D': DC, DL, DO)
- Randomly samples 200 QA pairs from the filtered set (seed=42)
- Loads videos from a local directory
- Frames the task as 5-way multiple choice (A–E)
- Per-subtype accuracy breakdown: DC / DL / DO
"""

import os
import sys
import csv
import random
import gc
import re
import torch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTION_LETTERS = ['A', 'B', 'C', 'D', 'E']


def build_mc_question(question, options):
    """Format question + 5 options as a multiple-choice prompt."""
    opts = '\n'.join(f'{OPTION_LETTERS[i]}) {opt}' for i, opt in enumerate(options))
    return f"{question}\n{opts}\nAnswer with A, B, C, D, or E."


def parse_mc_answer(prediction, options):
    """
    Map model output to 0-based option index.
    Tries: leading letter match → option text containment → first option found.
    Returns -1 if unable to determine.
    """
    pred = prediction.strip()

    # Direct letter at start of output
    if pred and pred[0].upper() in OPTION_LETTERS:
        return OPTION_LETTERS.index(pred[0].upper())

    # Letter followed by ) or .
    m = re.match(r'^([A-Ea-e])[).:]', pred)
    if m:
        return OPTION_LETTERS.index(m.group(1).upper())

    # Substring match against option texts
    pred_l = pred.lower()
    for idx, opt in enumerate(options):
        if opt.lower().strip() in pred_l or pred_l in opt.lower().strip():
            return idx

    return -1


def load_nextqa_csv(csv_path, descriptive_only=True):
    """
    Load NExT-QA CSV.
    Expected columns (official format):
        video, frame_count, width, height, qid, type, a0, a1, a2, a3, a4, answer, question
    Also handles simpler variant without frame_count/width/height.
    """
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            qtype = row.get('type', '').strip()
            if descriptive_only and not qtype.upper().startswith('D'):
                continue
            options = [row.get(f'a{i}', '').strip() for i in range(5)]
            if not any(options):
                continue
            try:
                answer_idx = int(row['answer'])
            except (KeyError, ValueError):
                continue
            rows.append({
                'video_id':   str(row.get('video', row.get('video_id', ''))).strip(),
                'qid':        row.get('qid', '').strip(),
                'qtype':      qtype.upper(),
                'question':   row.get('question', '').strip(),
                'options':    options,
                'answer_idx': answer_idx,
                'answer_text': options[answer_idx] if 0 <= answer_idx < 5 else '',
            })
    return rows


def run_nextqa_benchmark(
    qa_csv_path,
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
    Run CondVLM against 200 randomly sampled descriptive NExT-QA pairs.

    Args:
        qa_csv_path:  Path to val.csv or test.csv (official NExT-QA format)
        video_dir:    Directory containing <video_id>.mp4 files
        model/processor/text_model/text_tokenizer: pre-loaded (auto-loaded if None)
        output_dir:   Where to save results
        max_samples:  Number of QA pairs to evaluate (default 200)
        seed:         Random seed for sampling (default 42)
        fps_sample:   Frames per second to extract
        confidence_threshold: Frame-condition confidence cutoff
        aggregation_method:   'most_confident' | 'consensus' | 'average'
        save_results: Write CSV
        show_visualizations: Show IG plots

    Returns:
        list of result dicts
    """
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(SCRIPT_DIR, f'nextqa_results_{ts}')
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load & sample annotations ──────────────────────────────────────────
    print(f"\n[1/3] Loading NExT-QA descriptive questions from {qa_csv_path} ...")
    all_desc = load_nextqa_csv(qa_csv_path, descriptive_only=True)
    print(f"  Descriptive QA pairs found: {len(all_desc)}")

    random.seed(seed)
    sample = random.sample(all_desc, min(max_samples, len(all_desc)))
    print(f"  Sampled {len(sample)} pairs (seed={seed})")

    subtypes = sorted(set(r['qtype'] for r in sample))
    print(f"  Subtypes in sample: {subtypes}")

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
    print("RUNNING NExT-QA BENCHMARK (DESCRIPTIVE)")
    print(f"{'='*70}\n")

    results = []

    for i, qa in enumerate(sample):
        video_id = qa['video_id']
        options  = qa['options']
        mc_q     = build_mc_question(qa['question'], options)
        expected = qa['answer_text']
        qtype    = qa['qtype']

        # NExT-QA videos use the raw video_id as filename
        video_path = None
        for name in [f"{video_id}.mp4", f"{video_id}.avi"]:
            p = os.path.join(video_dir, name)
            if os.path.exists(p):
                video_path = p
                break

        print(f"\n{'#'*70}")
        print(f"[{i+1}/{len(sample)}] video_id={video_id}  subtype={qtype}")
        print(f"  Q:       {qa['question']}")
        print(f"  Options: {' | '.join(f'{OPTION_LETTERS[j]}) {o}' for j, o in enumerate(options))}")
        print(f"  Answer:  {OPTION_LETTERS[qa['answer_idx']]}) {expected}")
        print(f"{'#'*70}")

        if video_path is None:
            print(f"  SKIPPED — video not found in {video_dir}")
            results.append({
                'video_id': video_id, 'question_subtype': qtype, 'qid': qa['qid'],
                'question': qa['question'], 'options': ' | '.join(options),
                'expected_idx': qa['answer_idx'], 'expected': expected,
                'raw_prediction': 'N/A', 'predicted_idx': -1,
                'confidence': 0.0, 'correct': False,
                'status': 'SKIPPED', 'error': 'Video not found',
            })
            continue

        gc.collect()
        torch.cuda.empty_cache()

        tc_out = os.path.join(output_dir, f"{video_id}_{qa['qid']}") if save_results else 'nextqa_analysis'
        os.makedirs(tc_out, exist_ok=True)

        try:
            raw = conditional_query_vqa_interpret(
                video_path=video_path,
                questions=[mc_q],
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
                force_simple=True,
            )

            if mc_q in raw:
                r = raw[mc_q]
                if 'error' in r:
                    raw_pred, confidence, err = 'ERROR', 0.0, r['error']
                else:
                    raw_pred   = r.get('prediction', 'N/A')
                    confidence = r.get('confidence', 0.0)
                    err        = None
            else:
                raw_pred, confidence, err = 'NO_RESULT', 0.0, 'Key not found in results'

            predicted_idx = parse_mc_answer(raw_pred, options) if err is None else -1
            correct = (predicted_idx == qa['answer_idx']) if predicted_idx >= 0 else False

            results.append({
                'video_id': video_id, 'question_subtype': qtype, 'qid': qa['qid'],
                'question': qa['question'], 'options': ' | '.join(options),
                'expected_idx': qa['answer_idx'], 'expected': expected,
                'raw_prediction': raw_pred, 'predicted_idx': predicted_idx,
                'confidence': confidence, 'correct': correct,
                'status': 'OK' if err is None else 'ERROR', 'error': err,
            })
            pred_letter = OPTION_LETTERS[predicted_idx] if 0 <= predicted_idx < 5 else '?'
            ans_letter  = OPTION_LETTERS[qa['answer_idx']]
            print(f"  >> Raw output  : {raw_pred}")
            print(f"  >> Parsed      : {pred_letter}  |  Correct: {ans_letter}  |  Match: {correct}")
            print(f"  >> Confidence  : {confidence:.2%}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'video_id': video_id, 'question_subtype': qtype, 'qid': qa['qid'],
                'question': qa['question'], 'options': ' | '.join(options),
                'expected_idx': qa['answer_idx'], 'expected': expected,
                'raw_prediction': 'FAILED', 'predicted_idx': -1,
                'confidence': 0.0, 'correct': False,
                'status': 'FAILED', 'error': str(e),
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("NExT-QA BENCHMARK — RESULTS BY QUESTION SUBTYPE (DESCRIPTIVE)")
    print(f"{'='*70}")
    print(f"{'Subtype':<12} {'Total':<8} {'OK':<6} {'Correct':<10} {'Accuracy'}")
    print(f"{'-'*12} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")

    all_subtypes = sorted(set(r['question_subtype'] for r in results))
    for qt in all_subtypes:
        subset  = [r for r in results if r['question_subtype'] == qt]
        ok_sub  = [r for r in subset if r['status'] == 'OK']
        correct = sum(1 for r in ok_sub if r['correct'])
        acc     = correct / len(ok_sub) if ok_sub else 0.0
        print(f"{qt:<12} {len(subset):<8} {len(ok_sub):<6} {correct:<10} {acc:.2%}")

    ok_all      = [r for r in results if r['status'] == 'OK']
    correct_all = sum(1 for r in ok_all if r['correct'])
    overall_acc = correct_all / len(ok_all) if ok_all else 0.0
    err_cnt  = sum(1 for r in results if r['status'] == 'ERROR')
    skip_cnt = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail_cnt = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"{'─'*54}")
    print(f"{'OVERALL':<12} {len(results):<8} {len(ok_all):<6} {correct_all:<10} {overall_acc:.2%}")
    print(f"\nTotal: {len(results)} | OK: {len(ok_all)} | ERROR: {err_cnt} | SKIPPED: {skip_cnt} | FAILED: {fail_cnt}")

    if save_results:
        csv_path = os.path.join(output_dir, f'nextqa_results_{datetime.now():%Y%m%d_%H%M%S}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'video_id', 'question_subtype', 'qid', 'question', 'options',
                'expected_idx', 'expected', 'raw_prediction', 'predicted_idx',
                'confidence', 'correct', 'status', 'error',
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {csv_path}")
        print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--qa_csv',     required=True, help='Path to NExT-QA val.csv')
    p.add_argument('--video_dir',  required=True, help='Directory with <video_id>.mp4 files')
    p.add_argument('--output_dir', default=None)
    p.add_argument('--max_samples', type=int, default=200)
    args = p.parse_args()
    run_nextqa_benchmark(qa_csv_path=args.qa_csv, video_dir=args.video_dir,
                         output_dir=args.output_dir, max_samples=args.max_samples)
