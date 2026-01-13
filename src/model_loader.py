"""Model loading utilities for LLaVA Vision-Language Model"""

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig


def load_model(model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"):
    """
    Load the LLaVA vision-language model with 4-bit quantization

    Args:
        model_id: HuggingFace model identifier

    Returns:
        tuple: (model, processor)
    """
    # Configure 4-bit quantization to reduce memory usage
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print("Loading model (this may take a minute)...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto"
    )

    print("✓ Model loaded successfully")

    return model, processor
