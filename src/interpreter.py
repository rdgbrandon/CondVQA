"""Main interpretation module for Visual Question Answering with Integrated Gradients"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from captum.attr import visualization

from .utils import load_image_from_file, replace_with_padding, setup_colormap
from .text_interpreter import text_vqa_interpret


def vqa_interpret(image_path, questions, model, processor, show_top_k=10, include_text_attribution=False):
    """
    Visual Question Answering with Integrated Gradients attribution

    Args:
        image_path: Path to the image file
        questions: List of questions to ask about the image
        model: The vision-language model
        processor: The model's processor
        show_top_k: Number of top probable tokens to display (default: 10)
        include_text_attribution: If True, also compute text and joint attributions (default: False)
    """
    # Get device from model
    device = next(model.parameters()).device

    # Load image from file
    print(f"Loading image from file: {image_path}")
    img = load_image_from_file(image_path)

    # Setup colormap for visualizations
    default_cmap = setup_colormap()

    for idx, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {idx}/{len(questions)}: {question}")
        print('='*60)

        # Clear cache before each question
        torch.cuda.empty_cache()

        # Prepare conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Answer with one word only."},
                    {"type": "text", "text": question},
                ],
            },
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=img, text=prompt, return_tensors='pt').to(device)

        pixel_values = inputs["pixel_values"]
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        image_sizes = inputs['image_sizes']

        # Create baselines for Integrated Gradients
        pixel_values_baseline = torch.zeros_like(pixel_values).to(device)  # Black image baseline
        q_tokens = processor.tokenizer(question, return_tensors="pt").to(device)
        input_ids_baseline = input_ids.clone().to(device, torch.int64)
        pad_token_id = processor.tokenizer.pad_token_id
        input_ids_baseline = replace_with_padding(
            input_ids_baseline,
            q_tokens["input_ids"],
            pad_token_id
        ).to(device)

        # Forward function for attribution
        def custom_forward(pixel_values, input_ids, image_sizes=None, attention_mask=None):
            input_ids = input_ids.to(torch.int64)
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
            )
            # Get logits for the next token (first generated token)
            next_token_logits = outputs.logits[:, -1, :]
            return next_token_logits

        # Get model prediction - generate complete answer
        print("\nGenerating answer...")
        with torch.no_grad():
            # Generate complete answer (not just first token)
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=attention_mask,
                max_new_tokens=20,  # Allow up to 20 tokens for answer
                do_sample=False,  # Use greedy decoding for consistency
                pad_token_id=processor.tokenizer.pad_token_id
            )

            # Decode the generated answer (skip the prompt tokens)
            prompt_length = input_ids.shape[1]
            generated_tokens = generated_ids[0, prompt_length:]
            predicted_answer = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Get logits for first generated token to show alternatives
        with torch.no_grad():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
                attention_mask=attention_mask
            )

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Get confidence score for the first token of generated answer
        first_token_id = generated_tokens[0] if len(generated_tokens) > 0 else 0
        confidence_score = probs[0, first_token_id].item()

        print(f"\nPredicted Answer: '{predicted_answer}'")
        print(f"Confidence Score: {confidence_score:.4f} ({confidence_score*100:.2f}%)")

        # Get top-k tokens and their probabilities for alternative answers
        top_probs, top_indices = torch.topk(probs[0], k=show_top_k)

        print(f"\nTop {show_top_k} most probable first tokens:")
        print("-" * 50)
        for i, (prob, token_id) in enumerate(zip(top_probs, top_indices), 1):
            token_text = processor.tokenizer.decode(token_id).strip()
            print(f"{i:2d}. '{token_text:20s}' - {prob.item():.4f} ({prob.item()*100:.2f}%)")
        print("-" * 50)

        # Compute attributions using Integrated Gradients
        print("\nComputing Integrated Gradients (this may take a moment)...")
        ig = IntegratedGradients(custom_forward)

        # Use fewer steps to save memory
        attributions = ig.attribute(
            inputs=pixel_values,
            baselines=pixel_values_baseline,
            additional_forward_args=(input_ids, image_sizes, attention_mask),
            target=first_token_id,
            n_steps=5,  # Reduced steps to save memory
            internal_batch_size=1,  # Process one step at a time
        )

        # Visualize image attributions
        original_im_mat = np.asarray(img)

        # Process attributions: shape is [1, 2, 3, 384, 384]
        attributions_squeezed = attributions.squeeze(0)  # [2, 3, 384, 384]
        combined_patches = attributions_squeezed[0] + attributions_squeezed[1]  # Combine patches
        attributions_img = combined_patches.permute(1, 2, 0).cpu().detach().numpy()

        # Create visualization
        fig, _ = visualization.visualize_image_attr_multiple(
            attributions_img,
            original_im_mat,
            ["original_image", "heat_map"],
            ["all", "absolute_value"],
            titles=["Original Image", f"Attribution for '{predicted_answer}'"],
            cmap=default_cmap,
            show_colorbar=True,
            fig_size=(12, 6)
        )
        plt.tight_layout()
        plt.show()

        print("✓ Visualization complete")

        # Optionally compute text attribution analysis
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

        # Clear cache after each question
        torch.cuda.empty_cache()
