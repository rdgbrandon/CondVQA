"""
MSVD-QA Benchmark Runner for CondVLM
- Downloads annotations automatically from GitHub
- Downloads videos automatically via yt-dlp using the YouTube ID mapping
- No manual downloads required
"""

import os
import sys
import json
import csv
import subprocess
import urllib.request
import torch
import gc
from datetime import datetime

# Add project src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TemporalVLM_IG', 'src'))

from src import load_model, conditional_query_vqa_interpret
from src.model_loader import load_text_model


# ── URLs ──────────────────────────────────────────────────────────────────────
MSVD_QA_TEST_URL = (
    "https://raw.githubusercontent.com/xudejing/video-question-answering"
    "/master/data/msvd/test_qa.json"
)
MSVD_YOUTUBE_MAPPING_URL = (
    "https://raw.githubusercontent.com/xudejing/video-question-answering"
    "/master/data/msvd/youtube_mapping.txt"
)


# ── Download helpers ───────────────────────────────────────────────────────────

def download_file(url, dest_path):
    """Download a file from url to dest_path if not already present."""
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return
    print(f"  Downloading {os.path.basename(dest_path)} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved to {dest_path}")


def load_youtube_mapping(mapping_path):
    """
    Parse youtube_mapping.txt → {vid_id: youtube_id}
    Format per line:  vid1 <youtube_id>
    """
    mapping = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                vid_id, yt_id = parts[0], parts[1]
                mapping[vid_id] = yt_id
    return mapping


def download_video_yt_dlp(youtube_id, output_path, max_duration=60):
    """
    Download a YouTube video to output_path using yt-dlp.
    Skips if file already exists.
    Returns True on success, False on failure.
    """
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
        # yt-dlp may save with a different extension — check for .webm fallback
        webm_path = output_path.replace(".mp4", ".webm")
        if os.path.exists(webm_path):
            os.rename(webm_path, output_path)
            return True
        return False
    except Exception as e:
        print(f"    yt-dlp error for {youtube_id}: {e}")
        return False


# ── Main benchmark function ───────────────────────────────────────────────────

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
        model / processor:          Vision-language model (auto-loaded if None)
        text_model / text_tokenizer: Text LLM for parsing (auto-loaded if None)
        output_dir:                 Where to save results (default: test/msvd_results)
        data_dir:                   Where to cache annotations + videos
                                    (default: test/msvd_data)
        max_samples:                How many test questions to evaluate (None = all ~13k)
        fps_sample:                 Frames per second to sample from each video
        confidence_threshold:       Frame-matching confidence cutoff
        aggregation_method:         How to aggregate multi-frame results
        max_video_duration:         Skip videos longer than this many seconds
        save_results:               Write per-question results to CSV
        show_visualizations:        Show IG plots (disable for batch)

    Returns:
        list of result dicts
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if data_dir is None:
        data_dir = os.path.join(script_dir, 'msvd_data')
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'msvd_results')

    video_dir = os.path.join(data_dir, 'videos')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Download annotations ──
    print("\n[1/4] Downloading MSVD-QA annotations...")
    qa_path      = os.path.join(data_dir, 'test_qa.json')
    mapping_path = os.path.join(data_dir, 'youtube_mapping.txt')
    download_file(MSVD_QA_TEST_URL, qa_path)
    download_file(MSVD_YOUTUBE_MAPPING_URL, mapping_path)

    # ── 2. Load annotations ──
    print("\n[2/4] Loading annotations...")
    with open(qa_path, 'r') as f:
        qa_data = json.load(f)

    youtube_mapping = load_youtube_mapping(mapping_path)
    print(f"  Loaded {len(qa_data)} QA pairs")
    print(f"  YouTube mapping: {len(youtube_mapping)} videos")

    # Deduplicate by video so we test diverse videos, then cap at max_samples
    seen_videos = set()
    test_cases = []
    for item in qa_data:
        vid_id = f"vid{item['video_id']}"
        if vid_id not in seen_videos:
            seen_videos.add(vid_id)
            test_cases.append({
                'video_id':  vid_id,
                'question':  item['question'],
                'expected':  item['answer'],
                'youtube_id': youtube_mapping.get(vid_id, None)
            })
        if max_samples and len(test_cases) >= max_samples:
            break

    print(f"  Selected {len(test_cases)} test cases (max_samples={max_samples})")

    # ── 3. Download videos via yt-dlp ──
    print(f"\n[3/4] Downloading videos via yt-dlp (skipping already cached)...")
    for i, tc in enumerate(test_cases):
        yt_id = tc['youtube_id']
        if yt_id is None:
            print(f"  [{i+1}/{len(test_cases)}] {tc['video_id']} — no YouTube ID, will skip")
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

    # ── 4. Load models ──
    print("\n[4/4] Loading models...")
    if model is None or processor is None:
        print("  Loading vision-language model...")
        model, processor = load_model()
    if text_model is None or text_tokenizer is None:
        print("  Loading text model...")
        text_model, text_tokenizer = load_text_model()

    # ── 5. Run inference ──
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
                    prediction  = 'ERROR'
                    confidence  = 0.0
                    error_msg   = res['error']
                else:
                    prediction  = res.get('prediction', 'N/A')
                    confidence  = res.get('confidence', 0.0)
                    error_msg   = None
            else:
                prediction  = 'NO_RESULT'
                confidence  = 0.0
                error_msg   = 'Question key not found in results'

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

    ok = sum(1 for r in results if r['status'] == 'OK')
    err = sum(1 for r in results if r['status'] == 'ERROR')
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

    # Save CSV
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
