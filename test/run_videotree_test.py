"""
VideoTree-style batch test runner for CondVQA.

Pipeline (Wang et al. 2024 - VideoTree, arXiv:2405.19209):
  1. Extract frames from video at fps_sample rate
  2. Encode every frame with CLIP (ViT-B/32)
  3. Encode the question with CLIP text encoder
  4. K-means cluster the frame embeddings
  5. Score each cluster centroid by cosine similarity to the query embedding
  6. Pick one representative keyframe from each top-scoring cluster
  7. Answer the question with Qwen2.5-VL on the selected keyframes only

Uses openai/clip-vit-base-patch32 (lightweight; original paper uses EVA-CLIP-8B).
Uses Qwen2.5-VL-7B-Instruct for the final answer.
No OpenAI API required.

Install extra deps if missing:
  pip install scikit-learn
"""

import os
import csv
import cv2
import torch
import gc
import numpy as np
from datetime import datetime
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    CLIPModel, CLIPProcessor,
)
from sklearn.cluster import KMeans

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Install qwen_vl_utils: pip install qwen-vl-utils")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_clip(model_id="openai/clip-vit-base-patch32"):
    print(f"Loading CLIP ({model_id})...")
    clip_model = CLIPModel.from_pretrained(model_id).eval()
    clip_proc  = CLIPProcessor.from_pretrained(model_id)
    if torch.cuda.is_available():
        clip_model = clip_model.cuda()
    print("[OK] CLIP loaded")
    return clip_model, clip_proc


def load_qwen_vl(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    print(f"Loading processor ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    print("[OK] Qwen2.5-VL loaded")
    return model, processor


def extract_frames(video_path, fps_sample=1):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    interval = max(1, int(video_fps / fps_sample))
    frames, timestamps = [], []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            timestamps.append(round(idx / video_fps, 2))
        idx += 1
    cap.release()
    return frames, timestamps


def encode_frames(frames, clip_model, clip_proc):
    device = next(clip_model.parameters()).device
    feats = []
    for frame in frames:
        inp = clip_proc(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            f = clip_model.get_image_features(**inp)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().float().numpy())
    return np.vstack(feats)  # (N, D)


def encode_query(query, clip_model, clip_proc):
    device = next(clip_model.parameters()).device
    inp = clip_proc(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        f = clip_model.get_text_features(**inp)
    f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().float().numpy()  # (1, D)


def select_keyframes(frame_feats, query_feat, n_clusters=4, top_k=4):
    """
    Cluster frames, rank clusters by cosine similarity to query,
    and return the index of the frame closest to each top cluster centroid.
    """
    n = len(frame_feats)
    if n <= top_k:
        return list(range(n))

    k = min(n_clusters, n)
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(frame_feats)

    # cosine sim between each centroid and the query
    scores = (km.cluster_centers_ @ query_feat.T).squeeze()  # (k,)
    top_clusters = np.argsort(scores)[::-1][:top_k]

    selected = []
    for c in top_clusters:
        members = np.where(labels == c)[0]
        dists = np.linalg.norm(frame_feats[members] - km.cluster_centers_[c], axis=1)
        selected.append(int(members[np.argmin(dists)]))

    return sorted(selected)


def ask_qwen_images(images, question, model, processor, max_new_tokens=64):
    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": f"Answer briefly. {question}"})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def load_test_cases(csv_path):
    cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            cases.append({
                'video_id': row['Video ID'].strip(),
                'question': row['CQ'].strip(),
                'expected': row['Expected Answer'].strip(),
            })
    return cases


def run_videotree_test(
    csv_path=None,
    video_dir=None,
    clip_model=None,
    clip_proc=None,
    vl_model=None,
    vl_processor=None,
    fps_sample=1,
    n_clusters=4,
    top_k=4,
    output_dir=None,
):
    """
    Run CondVQA test cases with the VideoTree pipeline:
      1. Extract frames
      2. CLIP-encode frames + query
      3. K-means cluster; pick top-k frames by query relevance
      4. Answer with Qwen2.5-VL on selected keyframes

    Args:
        csv_path:    Path to CondVQA_3cols.csv (default: test/CondVQA_3cols.csv)
        video_dir:   Folder containing tc*.mp4 files (default: test/testcases/)
        clip_model:  CLIP model (loaded if None)
        clip_proc:   CLIP processor (loaded if None)
        vl_model:    Qwen2.5-VL model (loaded if None)
        vl_processor: VL processor (loaded if None)
        fps_sample:  Frames per second to sample
        n_clusters:  Number of k-means clusters
        top_k:       Number of keyframes to pass to VLM
        output_dir:  Results directory (auto-timestamped if None)

    Returns:
        list of result dicts
    """
    if csv_path is None:
        csv_path = os.path.join(SCRIPT_DIR, 'CondVQA_3cols.csv')
    if video_dir is None:
        video_dir = os.path.join(SCRIPT_DIR, 'testcases')
    if output_dir is None:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(SCRIPT_DIR, f'videotree_results_{ts}')

    os.makedirs(output_dir, exist_ok=True)

    if clip_model is None or clip_proc is None:
        clip_model, clip_proc = load_clip()
    if vl_model is None or vl_processor is None:
        vl_model, vl_processor = load_qwen_vl()

    test_cases = load_test_cases(csv_path)
    print(f"\nLoaded {len(test_cases)} test cases")
    print(f"Video dir:  {video_dir}")
    print(f"n_clusters={n_clusters}, top_k={top_k}")
    print(f"Output dir: {output_dir}\n")

    results = []

    for i, tc in enumerate(test_cases):
        video_path = os.path.join(video_dir, f"{tc['video_id']}.mp4")

        print(f"\n{'#'*70}")
        print(f"TEST {i+1}/{len(test_cases)}: {tc['video_id']}")
        print(f"Question: {tc['question']}")
        print(f"Expected: {tc['expected']}")
        print(f"{'#'*70}")

        if not os.path.exists(video_path):
            print(f"  SKIPPED — video not found: {video_path}")
            results.append({
                'video_id': tc['video_id'], 'question': tc['question'],
                'expected': tc['expected'], 'prediction': 'N/A',
                'num_frames': 0, 'keyframes_used': 0,
                'status': 'SKIPPED', 'error': 'Video not found',
            })
            continue

        gc.collect()
        torch.cuda.empty_cache()

        try:
            frames, timestamps = extract_frames(video_path, fps_sample=fps_sample)
            print(f"  Extracted {len(frames)} frames")

            frame_feats = encode_frames(frames, clip_model, clip_proc)
            query_feat  = encode_query(tc['question'], clip_model, clip_proc)

            selected = select_keyframes(frame_feats, query_feat, n_clusters=n_clusters, top_k=top_k)
            keyframes  = [frames[j] for j in selected]
            key_times  = [f"{timestamps[j]}s" for j in selected]
            print(f"  Selected {len(keyframes)} keyframes at: {', '.join(key_times)}")

            prediction = ask_qwen_images(keyframes, tc['question'], vl_model, vl_processor)

            print(f"\n  >> Prediction: {prediction}")
            print(f"  >> Expected:   {tc['expected']}")

            results.append({
                'video_id': tc['video_id'], 'question': tc['question'],
                'expected': tc['expected'], 'prediction': prediction,
                'num_frames': len(frames), 'keyframes_used': len(keyframes),
                'status': 'OK', 'error': None,
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'video_id': tc['video_id'], 'question': tc['question'],
                'expected': tc['expected'], 'prediction': 'FAILED',
                'num_frames': 0, 'keyframes_used': 0,
                'status': 'FAILED', 'error': str(e),
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    ok   = sum(1 for r in results if r['status'] == 'OK')
    skip = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"\n{'='*70}")
    print("VIDEOTREE RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'#':<6} {'Video ID':<10} {'Status':<10} {'Frames':<8} {'Keys':<6} {'Prediction':<35} {'Expected'}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*35} {'-'*30}")
    for i, r in enumerate(results, 1):
        print(
            f"{i:<6} {r['video_id']:<10} {r['status']:<10} "
            f"{r['num_frames']:<8} {r['keyframes_used']:<6} "
            f"{str(r['prediction'])[:33]:<35} {r['expected']}"
        )

    print(f"\n{'─'*70}")
    print(f"Total: {len(results)} | OK: {ok} | SKIPPED: {skip} | FAILED: {fail}")
    print(f"{'─'*70}")

    out_csv = os.path.join(output_dir, 'videotree_results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['video_id', 'question', 'expected', 'prediction',
                           'num_frames', 'keyframes_used', 'status', 'error']
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {out_csv}\n")

    return results


if __name__ == '__main__':
    run_videotree_test()
