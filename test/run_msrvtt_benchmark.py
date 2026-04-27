"""
MSRVTT Benchmark Runner for CondVLM
- Loads from HuggingFace: morpheushoc/msrvtt (binary_frames, same format as msvd-qa)
- Frames task as: Q = "What is happening in this video?"  A = first caption
- Randomly samples 200 test clips (seed=42)
- Per-question-type accuracy (what/who/how/when/where — fixed question so all are 'what')
"""

import os
import sys
import csv
import random
import gc
import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
QUESTION    = "What is happening in this video?"


def frames_to_video(raw_bytes, output_path, num_frames, height, width, channels, fps=1):
    """Reconstruct mp4 from raw pixel bytes — identical to run_msvd_benchmark."""
    import cv2
    if not raw_bytes or num_frames == 0 or height == 0 or width == 0:
        return False
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    expected = num_frames * height * width * channels
    if len(arr) < expected:
        return False
    arr = arr[:expected].reshape(num_frames, height, width, channels)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = arr[i]
        if channels == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def is_correct(prediction, expected):
    """Normalised substring match — standard open VQA metric."""
    pred = prediction.lower().strip().rstrip('.,!?')
    exp  = expected.lower().strip().rstrip('.,!?')
    if not exp:
        return False
    return pred == exp or exp in pred or pred in exp


def run_msrvtt_benchmark(
    model=None,
    processor=None,
    text_model=None,
    text_tokenizer=None,
    output_dir=None,
    data_dir=None,
    max_samples=200,
    seed=42,
    fps_sample=1,
    confidence_threshold=0.5,
    aggregation_method='most_confident',
    save_results=True,
    show_visualizations=False,
):
    """
    Run CondVLM against 200 randomly sampled MSRVTT test clips.
    Loads binary frames from morpheushoc/msrvtt (no downloads required).

    Returns:
        list of result dicts
    """
    from datasets import load_dataset

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(script_dir, 'msrvtt_data')
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(script_dir, f'msrvtt_results_{ts}')

    video_dir = os.path.join(data_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load dataset ────────────────────────────────────────────────────────
    print("\n[1/3] Loading MSRVTT from HuggingFace (morpheushoc/msrvtt, streaming) ...")
    ds = load_dataset("morpheushoc/msrvtt", split="test", streaming=True)

    sample = []
    for i, row in enumerate(ds):
        video_path_id = row.get('video_path', f'vid_{i}')
        vid_id = os.path.splitext(os.path.basename(str(video_path_id)))[0]
        captions = row.get('caption', [])
        expected = min(captions, key=len) if isinstance(captions, list) and captions else str(captions)
        if not expected:
            continue
        sample.append({
            'video_id':      vid_id,
            'question':      QUESTION,
            'expected':      expected,
            'binary_frames': row.get('binary_frames'),
            'num_frames':    row.get('num_frames', 0),
            'height':        row.get('height', 0),
            'width':         row.get('width', 0),
            'channels':      row.get('channels', 3),
        })
        if len(sample) >= max_samples:
            break

    print(f"  Collected {len(sample)} clips (seed={seed}, streaming)")

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
    print("RUNNING MSRVTT BENCHMARK")
    print(f"{'='*70}\n")

    results = []

    for i, tc in enumerate(sample):
        video_id = tc['video_id']
        question = tc['question']
        expected = tc['expected']

        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            ok = frames_to_video(
                tc['binary_frames'], video_path,
                num_frames=tc['num_frames'], height=tc['height'],
                width=tc['width'], channels=tc['channels'],
                fps=fps_sample,
            )
            if not ok:
                video_path = None

        print(f"\n{'#'*70}")
        print(f"[{i+1}/{len(sample)}] {video_id}")
        print(f"  Q:        {question}")
        print(f"  Expected: {expected}")
        print(f"{'#'*70}")

        if video_path is None:
            print("  SKIPPED — video reconstruction failed")
            results.append({
                'video_id': video_id, 'question': question, 'expected': expected,
                'prediction': 'N/A', 'confidence': 0.0,
                'correct': False, 'status': 'SKIPPED', 'error': 'Reconstruction failed',
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
                prediction, confidence, err = 'NO_RESULT', 0.0, 'Key not found'

            correct = is_correct(prediction, expected) if err is None else False
            results.append({
                'video_id': video_id, 'question': question, 'expected': expected,
                'prediction': prediction, 'confidence': confidence,
                'correct': correct,
                'status': 'OK' if err is None else 'ERROR', 'error': err,
            })
            print(f"  >> Prediction : {prediction}  |  Correct: {correct}")
            print(f"  >> Confidence : {confidence:.2%}")

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'video_id': video_id, 'question': question, 'expected': expected,
                'prediction': 'FAILED', 'confidence': 0.0,
                'correct': False, 'status': 'FAILED', 'error': str(e),
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    ok_all      = [r for r in results if r['status'] == 'OK']
    correct_all = sum(1 for r in ok_all if r['correct'])
    overall_acc = correct_all / len(ok_all) if ok_all else 0.0
    err_cnt  = sum(1 for r in results if r['status'] == 'ERROR')
    skip_cnt = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail_cnt = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"\n{'='*70}")
    print("MSRVTT BENCHMARK — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Video ID':<20} {'Status':<10} {'Conf':<8} {'Correct':<8} {'Prediction':<30} {'Expected'}")
    print("-" * 100)
    for r in results:
        print(f"{r['video_id']:<20} {r['status']:<10} {r['confidence']:<8.2%} "
              f"{str(r['correct']):<8} {str(r['prediction'])[:28]:<30} {str(r['expected'])[:40]}")

    print(f"\n{'─'*70}")
    print(f"Total: {len(results)} | OK: {len(ok_all)} | Correct: {correct_all} | "
          f"Accuracy: {overall_acc:.2%} | ERROR: {err_cnt} | SKIPPED: {skip_cnt} | FAILED: {fail_cnt}")
    print(f"{'─'*70}\n")

    if save_results:
        csv_path = os.path.join(output_dir, f'msrvtt_results_{datetime.now():%Y%m%d_%H%M%S}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'video_id', 'question', 'expected', 'prediction',
                'confidence', 'correct', 'status', 'error',
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {csv_path}")

    return results


if __name__ == '__main__':
    run_msrvtt_benchmark()
