"""
MSVD-QA Benchmark Runner for CondVLM
- Loads annotations from Hugging Face (no manual download)
- Downloads videos automatically via yt-dlp
- No manual downloads required
"""

import os
import sys
import csv
import subprocess
import torch
import gc
from datetime import datetime

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


# ── Video download ─────────────────────────────────────────────────────────────

def download_video_yt_dlp(youtube_id, output_path, max_duration=60):
    """Download a YouTube video via yt-dlp. Returns True on success."""
    if os.path.exists(output_path):
        return True

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--match-filter", f"duration < {max_duration}",
        "-f", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "-o", output_path,
        url
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(output_path):
            return True
        webm_path = output_path.replace(".mp4", ".webm")
        if os.path.exists(webm_path):
            os.rename(webm_path, output_path)
            return True
        return False
    except Exception as e:
        print(f"    yt-dlp error for {youtube_id}: {e}")
        return False


# ── Main benchmark function ────────────────────────────────────────────────────

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
    max_video_duration=60,
    save_results=True,
    show_visualizations=False,
):
    """
    Run CondVLM against the MSVD-QA test set.

    Args:
        model / processor:           Vision-language model (auto-loaded if None)
        text_model / text_tokenizer: Text LLM for parsing (auto-loaded if None)
        output_dir:                  Where to save results (default: test/msvd_results)
        data_dir:                    Where to cache videos (default: test/msvd_data)
        max_samples:                 How many test questions to evaluate (None = all)
        fps_sample:                  Frames per second to sample from each video
        confidence_threshold:        Frame-matching confidence cutoff
        aggregation_method:          How to aggregate multi-frame results
        max_video_duration:          Skip videos longer than this many seconds
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

    # ── 1. Load annotations from Hugging Face ──
    print("\n[1/4] Loading MSVD-QA annotations from Hugging Face...")
    ds = load_dataset("AlexZigma/msvd-qa", split="test")
    print(f"  Loaded {len(ds)} QA pairs from HuggingFace")
    print(f"  Columns: {ds.column_names}")

    # Deduplicate by video, cap at max_samples
    seen_videos = set()
    test_cases = []
    for item in ds:
        vid_id = str(item.get('video_id') or item.get('video_name') or item.get('id'))
        # get youtube id if available
        yt_id   = item.get('youtube_id') or item.get('ytid') or item.get('yt_id') or None
        question = item.get('question', '')
        answer   = item.get('answer', '')

        if vid_id not in seen_videos:
            seen_videos.add(vid_id)
            test_cases.append({
                'video_id':   vid_id,
                'question':   question,
                'expected':   answer,
                'youtube_id': yt_id,
            })
        if max_samples and len(test_cases) >= max_samples:
            break

    print(f"  Selected {len(test_cases)} test cases (max_samples={max_samples})")

    # Print a sample so we can verify the format
    print(f"\n  Sample entry:")
    print(f"    video_id   : {test_cases[0]['video_id']}")
    print(f"    question   : {test_cases[0]['question']}")
    print(f"    expected   : {test_cases[0]['expected']}")
    print(f"    youtube_id : {test_cases[0]['youtube_id']}")

    # ── 2. Download videos via yt-dlp ──
    print(f"\n[2/4] Downloading videos via yt-dlp...")
    for i, tc in enumerate(test_cases):
        yt_id = tc['youtube_id']
        if not yt_id:
            print(f"  [{i+1}/{len(test_cases)}] {tc['video_id']} — no YouTube ID, skipping")
            tc['video_path'] = None
            continue

        video_path = os.path.join(video_dir, f"{tc['video_id']}.mp4")
        tc['video_path'] = video_path

        if os.path.exists(video_path):
            print(f"  [{i+1}/{len(test_cases)}] {tc['video_id']} — cached")
            continue

        print(f"  [{i+1}/{len(test_cases)}] {tc['video_id']} ({yt_id}) — downloading...")
        success = download_video_yt_dlp(yt_id, video_path, max_duration=max_video_duration)
        if not success:
            print(f"    Failed to download {tc['video_id']}")
            tc['video_path'] = None

    # ── 3. Load models ──
    print("\n[3/4] Loading models...")
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

        if tc.get('video_path') is None or not os.path.exists(tc['video_path']):
            print("  SKIPPED — video not available")
            results.append({**tc, 'prediction': 'N/A', 'confidence': 0.0,
                            'status': 'SKIPPED', 'error': 'Video unavailable'})
            continue

        gc.collect()
        torch.cuda.empty_cache()

        tc_output_dir = os.path.join(output_dir, tc['video_id']) if save_results else None
        if tc_output_dir:
            os.makedirs(tc_output_dir, exist_ok=True)

        try:
            cond_results = conditional_query_vqa_interpret(
                video_path=tc['video_path'],
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

    print(f"{'Video ID':<12} {'Status':<10} {'Conf':<8} {'Prediction':<30} {'Expected':<25}")
    print("-" * 85)
    for r in results:
        print(f"{r['video_id']:<12} {r['status']:<10} {r['confidence']:<8.2%} "
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
