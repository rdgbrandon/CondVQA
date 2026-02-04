"""Model loading utilities for LLaVA Vision-Language Model"""

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlavaOnevisionForConditionalGeneration,
    BitsAndBytesConfig
)


def load_model(model_id="llava-hf/llava-onevision-qwen2-1.5b-ov-hf"):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_id,
        use_fast=False
    )

    print("Loading model...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        device_map="auto",
        ignore_mismatched_sizes=False
    )

    model.eval()
    print("[OK] Model loaded successfully")

    return model, processor



def load_text_model(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Load a text-only LLM for question parsing

    Args:
        model_id: HuggingFace model identifier for text-only model

    Returns:
        tuple: (model, tokenizer)
    """
    # Configure 4-bit quantization to reduce memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"Loading text model tokenizer ({model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("Loading text model (this may take a minute)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print("[OK] Text model loaded successfully")

    return model, tokenizer
