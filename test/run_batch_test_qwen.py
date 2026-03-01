"""Batch test runner for CondVQA using Qwen2.5-VL-7B - passes video directly, no frame extraction."""

import os
import csv
import torch
import gc
import sys

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("Install qwen_vl_utils: pip install qwen-vl-utils")


def load_qwen_model(model_id="Qwen/Qwen2.5-VL-7B-Instruct"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading processor ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto"
    )

    model.eval()
    print("[OK] Qwen2.5-VL model loaded successfully")

    return model, processor


def ask_qwen_video(video_path, question, model, processor, fps=1, max_new_tokens=64):
    """Pass video directly to Qwen and get an answer."""
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                },
                {"type": "text", "text": f"Answer briefly. {question}"},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    answer = processor.decode(generated_tokens, skip_special_tokens=True).strip()

    return answer


def load_test_cases(csv_path):
    """Load test cases from CSV file."""
    test_cases = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append({
                'video_id': row['Video ID'].strip(),
                'question': row['CQ'].strip(),
                'expected': row['Expected Answer'].strip()
            })
    return test_cases


def run_batch_test_qwen(
    csv_path=None,
    video_dir=None,
    model=None,
    processor=None,
    fps=1,
    output_dir=None
):
    """
    Run all test cases from CSV using Qwen2.5-VL-7B passing video directly.

    Args:
        csv_path: Path to CondVQA_3cols.csv
        video_dir: Path to folder containing tc*.mp4 files
        model: Qwen2.5-VL model (loaded if None)
        processor: Model processor (loaded if None)
        fps: Frames per second passed to Qwen for video sampling
        output_dir: Directory for all output

    Returns:
        list of result dicts with predictions and comparisons
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        csv_path = os.path.join(script_dir, 'CondVQA_3cols.csv')
    if video_dir is None:
        video_dir = os.path.join(script_dir, 'testcases')
    if output_dir is None:
        output_dir = os.path.join(script_dir, 'batch_results_qwen')

    os.makedirs(output_dir, exist_ok=True)

    if model is None or processor is None:
        print("Loading Qwen2.5-VL-7B model...")
        model, processor = load_qwen_model()

    test_cases = load_test_cases(csv_path)
    print(f"\nLoaded {len(test_cases)} test cases from {csv_path}")
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}\n")

    results = []

    for i, tc in enumerate(test_cases):
        video_file = f"{tc['video_id']}.mp4"
        video_path = os.path.join(video_dir, video_file)

        print(f"\n{'#'*70}")
        print(f"TEST CASE {i+1}/{len(test_cases)}: {tc['video_id']}")
        print(f"Question: {tc['question']}")
        print(f"Expected: {tc['expected']}")
        print(f"{'#'*70}")

        if not os.path.exists(video_path):
            print(f"  SKIPPED - video not found: {video_path}")
            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': 'N/A',
                'confidence': 'N/A',
                'status': 'SKIPPED',
                'error': 'Video file not found'
            })
            continue

        gc.collect()
        torch.cuda.empty_cache()

        try:
            prediction = ask_qwen_video(
                video_path=video_path,
                question=tc['question'],
                model=model,
                processor=processor,
                fps=fps,
            )

            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': prediction,
                'confidence': 'N/A',
                'status': 'OK',
                'error': None
            })

            print(f"\n  >> Prediction: {prediction}")
            print(f"  >> Expected:   {tc['expected']}")

        except Exception as e:
            print(f"  FAILED: {str(e)}")
            results.append({
                'video_id': tc['video_id'],
                'question': tc['question'],
                'expected': tc['expected'],
                'prediction': 'FAILED',
                'confidence': 'N/A',
                'status': 'FAILED',
                'error': str(e)
            })

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary Report ──
    print(f"\n{'='*70}")
    print("QWEN BATCH TEST RESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Video ID':<10} {'Status':<10} {'Prediction':<40} {'Expected':<30}")
    print("-" * 92)

    ok_count = error_count = skip_count = fail_count = 0

    for r in results:
        status = r['status']
        if status == 'OK':
            ok_count += 1
        elif status == 'ERROR':
            error_count += 1
        elif status == 'SKIPPED':
            skip_count += 1
        else:
            fail_count += 1

        pred_display = str(r['prediction'])[:38]
        exp_display = str(r['expected'])[:28]
        print(f"{r['video_id']:<10} {status:<10} {pred_display:<40} {exp_display:<30}")

    print(f"\n{'─'*70}")
    print(f"Total: {len(results)} | OK: {ok_count} | Errors: {error_count} | Skipped: {skip_count} | Failed: {fail_count}")
    print(f"{'─'*70}\n")

    results_csv_path = os.path.join(output_dir, 'batch_results_qwen.csv')
    with open(results_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'question', 'expected', 'prediction', 'confidence', 'status', 'error'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to: {results_csv_path}")

    return results


if __name__ == '__main__':
    run_batch_test_qwen()
