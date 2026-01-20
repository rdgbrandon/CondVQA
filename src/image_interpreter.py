"""Main interpretation module for Visual Question Answering with Integrated Gradients"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization

from .utils import load_image_from_file, replace_with_padding, setup_colormap
from .text_interpreter import text_vqa_interpret


def compute_single_image_attribution(image_path, question, model, processor, n_steps=5, top_k=5):
    """Compute attribution for a single image and question."""
    device = next(model.parameters()).device
    img = load_image_from_file(image_path)

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Answer briefly."},
            {"type": "text", "text": question},
        ],
    }]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_sizes = inputs['image_sizes']

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )

        prompt_length = input_ids.shape[1]
        generated_tokens = generated_ids[0, prompt_length:]
        predicted_answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_sizes=image_sizes,
            attention_mask=attention_mask
        )

    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)

    first_token_id = generated_tokens[0] if len(generated_tokens) > 0 else torch.argmax(logits, dim=-1).item()
    confidence_score = probs[0, first_token_id].item()

    top_probs, top_indices = torch.topk(probs[0], k=top_k)
    top_predictions = [
        (processor.tokenizer.decode(token_id).strip(), prob.item())
        for prob, token_id in zip(top_probs, top_indices)
    ]

    pixel_values_baseline = torch.zeros_like(pixel_values).to(device)
    q_tokens = processor.tokenizer(question, return_tensors="pt").to(device)
    input_ids_baseline = replace_with_padding(
        input_ids.clone().to(device, torch.int64),
        q_tokens["input_ids"],
        processor.tokenizer.pad_token_id
    ).to(device)

    def custom_forward(pixel_values, input_ids, image_sizes=None, attention_mask=None):
        input_ids = input_ids.to(torch.int64)
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
        )
        return outputs.logits[:, -1, :]

    ig = IntegratedGradients(custom_forward)
    attributions = ig.attribute(
        inputs=pixel_values,
        baselines=pixel_values_baseline,
        additional_forward_args=(input_ids, image_sizes, attention_mask),
        target=first_token_id,
        n_steps=n_steps,
        internal_batch_size=1,
    )

    attributions_squeezed = attributions.squeeze(0)
    combined_patches = attributions_squeezed[0] + attributions_squeezed[1]
    attributions_img = combined_patches.permute(1, 2, 0).cpu().detach().numpy()

    return {
        'prediction': predicted_answer,
        'confidence': confidence_score,
        'top_predictions': top_predictions,
        'attributions': attributions_img,
        'original_image': np.asarray(img)
    }


def vqa_interpret(image_path, questions, model, processor, show_top_k=10, include_text_attribution=False):
    """Visual Question Answering with Integrated Gradients attribution."""
    default_cmap = setup_colormap()
    print(f"Loading image from file: {image_path}")

    for idx, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {idx}/{len(questions)}: {question}")
        print('='*60)

        torch.cuda.empty_cache()

        print("\nGenerating answer and computing attributions...")
        result = compute_single_image_attribution(
            image_path, question, model, processor,
            n_steps=5, top_k=show_top_k
        )

        print(f"\nPredicted Answer: '{result['prediction']}'")
        print(f"Confidence Score: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")

        print(f"\nTop {show_top_k} most probable first tokens:")
        print("-" * 50)
        for i, (token_text, prob) in enumerate(result['top_predictions'], 1):
            print(f"{i:2d}. '{token_text:20s}' - {prob:.4f} ({prob*100:.2f}%)")
        print("-" * 50)

        fig, _ = visualization.visualize_image_attr_multiple(
            result['attributions'],
            result['original_image'],
            ["original_image", "heat_map"],
            ["all", "absolute_value"],
            titles=["Original Image", f"Attribution for '{result['prediction']}'"],
            cmap=default_cmap,
            show_colorbar=True,
            fig_size=(12, 6)
        )
        plt.tight_layout()
        plt.show()

        print("✓ Visualization complete")

        if include_text_attribution:
            text_vqa_interpret(
                image_path=image_path,
                question=question,
                model=model,
                processor=processor,
                mode='both',
                n_steps=10,
                show_visualizations=True
            )

        torch.cuda.empty_cache()
