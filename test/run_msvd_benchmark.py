"""
MSVD-QA Benchmark Runner for CondVLM
- Loads annotations AND video frames directly from HuggingFace (morpheushoc/msvd-qa)
- No manual downloads required, no yt-dlp needed
"""

import os
import sys
import csv
import json
import tempfile
import struct
import torch
import gc
import numpy as np
from datetime import datetime

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


def frames_to_video(frames_data, output_path, fps=1):
    """
    Write video frames from the dataset to an mp4 file using OpenCV.
    frames_data: list of numpy arrays (H, W, C) or raw bytes
    """
    import cv2

    if not frames_data:
        return False

    # Handle different possible formats
    first = frames_data[0]
    if isinstance(first, (bytes, bytearray)):
        # Try to decode as JPEG/PNG bytes
        frames = []
        for f in frames_data:
            arr = np.frombuffer(f, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
    elif isinstance(first, np.ndarray):
        frames = [f for f in frames_data if f is not None]
    else:
        return False

    if not frames:
        return False

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def run_msvd_benchmark(
    model=None,
    processor=None,
    text_model=None,
    text_tokenizer=None,
    output_dir=None,
    data_dir=None,
    max_samples=50,
    fps_sample=1,
    confidence_threshold=0.5,
    aggregation_method='most_confident',
    save_results=True,
    show_visualizations=False,
):
    """
    Run CondVLM against the MSVD-QA test set loaded from HuggingFace.

    Args:
        model / processor:           Vision-language model (auto-loaded if None)
        text_model / text_tokenizer: Text LLM for parsing (auto-loaded if None)
        output_dir:                  Where to save results (default: test/msvd_results)
        data_dir:                    Where to cache temp videos (default: test/msvd_data)
        max_samples:                 How many videos to evaluate (None = all)
        fps_sample:                  Frames per second to sample
        confidence_threshold:        Frame-matching confidence cutoff
        aggregation_method:          How to aggregate multi-frame results
        save_results:                Write results to CSV
        show_visualizations:         Show IG plots (disable for batch)

    Returns:
        list of result dicts
    """
    from datasets import load_dataset

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(script_dir, 'msvd_data')
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'msvd_results')

    video_dir = os.path.join(data_dir, 'videos')
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Load dataset from HuggingFace ──
    print("\n[1/3] Loading MSVD-QA from HuggingFace (morpheushoc/msvd-qa)...")
    ds = load_dataset("morpheushoc/msvd-qa", split="test")
    print(f"  Loaded {len(ds)} videos")
    print(f"  Columns: {ds.column_names}")

    # Inspect first row to understand structure
    row0 = ds[0]
    print(f"\n  Sample entry keys: {list(row0.keys())}")
    qa0 = row0.get('qa', [])
    if qa0:
        first_qa = qa0[0] if isinstance(qa0, list) else qa0
        print(f"  Sample QA: {first_qa}")

    # ── 2. Build test cases ──
    test_cases = []
    for i, row in enumerate(ds):
        video_path_id = row.get('video_path', f'vid_{i}')
        vid_id = os.path.splitext(os.path.basename(str(video_path_id)))[0]

        qa_pairs = row.get('qa', [])
        if not qa_pairs:
            continue

        # qa is a list: [question, answer]
        first_qa = qa_pairs[0] if isinstance(qa_pairs, list) else qa_pairs
        if isinstance(first_qa, list) and len(first_qa) >= 2:
            question = first_qa[0]
            answer   = first_qa[1]
        elif isinstance(first_qa, dict):
            question = first_qa.get('question', '')
            answer   = first_qa.get('answer', '')
        elif isinstance(first_qa, str):
            try:
                parsed = json.loads(first_qa)
                question = parsed.get('question', '')
                answer   = parsed.get('answer', '')
            except Exception:
                question = first_qa
                answer   = ''
        else:
            continue

        if not question:
            continue

        test_cases.append({
            'video_id':     vid_id,
            'question':     question,
            'expected':     answer,
            'binary_frames': row.get('binary_frames', None),
            'num_frames':   row.get('num_frames', 0),
            'height':       row.get('height', 0),
            'width':        row.get('width', 0),
            'channels':     row.get('channels', 3),
        })

        if max_samples and len(test_cases) >= max_samples:
            break

    print(f"\n  Selected {len(test_cases)} test cases")

    # ── 3. Load models ──
    print("\n[2/3] Loading models...")
    if model is None or processor is None:
        print("  Loading vision-language model...")
        model, processor = load_model()
    if text_model is None or text_tokenizer is None:
        print("  Loading text model...")
        text_model, text_tokenizer = load_text_model()

    # ── 4. Run inference ──
    print(f"\n{'='*70}")
    print("RUNNING MSVD-QA BENCHMARK")
    print(f"{'='*70}\n")

    results = []

    for i, tc in enumerate(test_cases):
        print(f"\n{'#'*70}")
        print(f"[{i+1}/{len(test_cases)}] {tc['video_id']}")
        print(f"  Question : {tc['question']}")
        print(f"  Expected : {tc['expected']}")
        print(f"{'#'*70}")

        # Build temp video from binary_frames
        video_path = os.path.join(video_dir, f"{tc['video_id']}.mp4")
        if not os.path.exists(video_path):
            frames_data = tc.get('binary_frames')
            if frames_data is None:
                print("  SKIPPED — no frame data")
                results.append({
                    'video_id': tc['video_id'], 'question': tc['question'],
                    'expected': tc['expected'], 'prediction': 'N/A',
                    'confidence': 0.0, 'status': 'SKIPPED', 'error': 'No frame data'
                })
                continue

            if not isinstance(frames_data, list):
                frames_data = [frames_data]

            ok = frames_to_video(frames_data, video_path, fps=fps_sample)
            if not ok:
                print("  SKIPPED — could not reconstruct video")
                results.append({
                    'video_id': tc['video_id'], 'question': tc['question'],
                    'expected': tc['expected'], 'prediction': 'N/A',
                    'confidence': 0.0, 'status': 'SKIPPED', 'error': 'Video reconstruction failed'
                })
                continue

        gc.collect()
        torch.cuda.empty_cache()

        tc_output_dir = os.path.join(output_dir, tc['video_id']) if save_results else None
        if tc_output_dir:
            os.makedirs(tc_output_dir, exist_ok=True)

        try:
            cond_results = conditional_query_vqa_interpret(
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
                output_dir=tc_output_dir or 'msvd_analysis',
                include_text_attribution=False
            )

            question_key = tc['question']
            if question_key in cond_results:
                res = cond_results[question_key]
                if 'error' in res:
                    prediction = 'ERROR'
                    confidence = 0.0
                    error_msg  = res['error']
                else:
                    prediction = res.get('prediction', 'N/A')
                    confidence = res.get('confidence', 0.0)
                    error_msg  = None
            else:
                prediction = 'NO_RESULT'
                confidence = 0.0
                error_msg  = 'Question key not found in results'

            print(f"\n  >> Prediction : {prediction}")
            print(f"  >> Expected   : {tc['expected']}")
            print(f"  >> Confidence : {confidence:.2%}")

            results.append({
                'video_id':   tc['video_id'],
                'question':   tc['question'],
                'expected':   tc['expected'],
                'prediction': prediction,
                'confidence': confidence,
                'status':     'OK' if error_msg is None else 'ERROR',
                'error':      error_msg
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'video_id':   tc['video_id'],
                'question':   tc['question'],
                'expected':   tc['expected'],
                'prediction': 'FAILED',
                'confidence': 0.0,
                'status':     'FAILED',
                'error':      str(e)
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print("MSVD-QA BENCHMARK RESULTS")
    print(f"{'='*70}\n")

    ok   = sum(1 for r in results if r['status'] == 'OK')
    err  = sum(1 for r in results if r['status'] == 'ERROR')
    skip = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"{'Video ID':<20} {'Status':<10} {'Conf':<8} {'Prediction':<30} {'Expected':<25}")
    print("-" * 93)
    for r in results:
        print(f"{r['video_id']:<20} {r['status']:<10} {r['confidence']:<8.2%} "
              f"{str(r['prediction'])[:28]:<30} {str(r['expected'])[:23]:<25}")

    print(f"\n{'─'*70}")
    print(f"Total: {len(results)} | OK: {ok} | Errors: {err} | Skipped: {skip} | Failed: {fail}")
    print(f"{'─'*70}\n")

    if save_results:
        csv_path = os.path.join(output_dir, f'msvd_results_{datetime.now():%Y%m%d_%H%M%S}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'video_id', 'question', 'expected', 'prediction',
                'confidence', 'status', 'error'
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to: {csv_path}")

    return results


if __name__ == '__main__':
    run_msvd_benchmark()
