"""
LLoVi-style batch test runner for CondVQA.

Pipeline (Zhang et al. 2024 - LLoVi, arXiv:2312.17235):
  Stage 1 — Dense frame captioning: Qwen2.5-VL generates a one-sentence description per frame.
  Stage 2 — LLM reasoning: the same VL model reads all captions as text and answers the question.

No OpenAI API required — runs fully locally with Qwen2.5-VL-7B.
"""

import os
import csv
import cv2
import torch
import gc
from datetime import datetime
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Install qwen_vl_utils: pip install qwen-vl-utils")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        idx += 1
    cap.release()
    return frames


def caption_frame(image, model, processor, max_new_tokens=64):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe what you see in this frame in one concise sentence."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = out[0][inputs.input_ids.shape[1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def reason_over_captions(captions, question, model, processor, max_new_tokens=128):
    caption_block = "\n".join(f"Frame {i+1}: {c}" for i, c in enumerate(captions))
    prompt = (
        "You are analyzing a video. Below are descriptions of sequential frames.\n\n"
        f"{caption_block}\n\n"
        f"Based only on these descriptions, answer briefly:\n"
        f"Question: {question}\n"
        "Answer:"
    )
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
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


def run_llovi_test(
    csv_path=None,
    video_dir=None,
    model=None,
    processor=None,
    fps_sample=1,
    output_dir=None,
):
    """
    Run CondVQA test cases with the LLoVi pipeline:
      1. Caption every frame with Qwen2.5-VL
      2. Reason over captions with the same model to answer the question

    Args:
        csv_path:    Path to CondVQA_3cols.csv (default: test/CondVQA_3cols.csv)
        video_dir:   Folder containing tc*.mp4 files (default: test/testcases/)
        model:       Qwen2.5-VL model (loaded if None)
        processor:   Model processor (loaded if None)
        fps_sample:  Frames per second to sample from video
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
        output_dir = os.path.join(SCRIPT_DIR, f'llovi_results_{ts}')

    os.makedirs(output_dir, exist_ok=True)

    if model is None or processor is None:
        model, processor = load_qwen_vl()

    test_cases = load_test_cases(csv_path)
    print(f"\nLoaded {len(test_cases)} test cases")
    print(f"Video dir:  {video_dir}")
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
                'num_frames': 0, 'status': 'SKIPPED', 'error': 'Video not found',
            })
            continue

        gc.collect()
        torch.cuda.empty_cache()

        try:
            frames = extract_frames(video_path, fps_sample=fps_sample)
            print(f"  Extracted {len(frames)} frames — captioning...")

            captions = []
            for j, frame in enumerate(frames):
                cap = caption_frame(frame, model, processor)
                captions.append(cap)
                print(f"  Frame {j+1}/{len(frames)}: {cap}")

            prediction = reason_over_captions(captions, tc['question'], model, processor)

            print(f"\n  >> Prediction: {prediction}")
            print(f"  >> Expected:   {tc['expected']}")

            results.append({
                'video_id': tc['video_id'], 'question': tc['question'],
                'expected': tc['expected'], 'prediction': prediction,
                'num_frames': len(frames), 'status': 'OK', 'error': None,
            })

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                'video_id': tc['video_id'], 'question': tc['question'],
                'expected': tc['expected'], 'prediction': 'FAILED',
                'num_frames': 0, 'status': 'FAILED', 'error': str(e),
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────────────
    ok   = sum(1 for r in results if r['status'] == 'OK')
    skip = sum(1 for r in results if r['status'] == 'SKIPPED')
    fail = sum(1 for r in results if r['status'] == 'FAILED')

    print(f"\n{'='*70}")
    print("LLOVI RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'#':<6} {'Video ID':<10} {'Status':<10} {'Frames':<8} {'Prediction':<35} {'Expected'}")
    print(f"{'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*35} {'-'*30}")
    for i, r in enumerate(results, 1):
        print(
            f"{i:<6} {r['video_id']:<10} {r['status']:<10} "
            f"{r['num_frames']:<8} {str(r['prediction'])[:33]:<35} {r['expected']}"
        )

    print(f"\n{'─'*70}")
    print(f"Total: {len(results)} | OK: {ok} | SKIPPED: {skip} | FAILED: {fail}")
    print(f"{'─'*70}")

    out_csv = os.path.join(output_dir, 'llovi_results.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=['video_id', 'question', 'expected', 'prediction', 'num_frames', 'status', 'error']
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {out_csv}\n")

    return results


if __name__ == '__main__':
    run_llovi_test()
