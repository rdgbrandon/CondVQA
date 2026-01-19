"""Text attribution module for analyzing question importance in VQA predictions"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import IntegratedGradients

from .utils import load_image_from_file


def compute_text_attributions(
    image_path,
    question,
    model,
    processor,
    n_steps=10
):
    """
    Compute text-only attributions showing which question words influence the prediction

    Args:
        image_path: Path to the image file
        question: Question text to analyze
        model: The vision-language model
        processor: The model's processor
        n_steps: Number of Integrated Gradients steps

    Returns:
        dict containing:
            - predicted_answer: The model's predicted answer
            - confidence: Confidence score
            - text_attributions: Attribution scores for each token
            - tokens: List of token strings
            - token_ids: List of token IDs
    """
    device = next(model.parameters()).device

    # Load image
    img = load_image_from_file(image_path)

    # Prepare conversation
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

    # Generate prediction
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

        # Get confidence
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

    # Compute text attributions using a gradient-based approach
    # We'll compute input gradients for each token position

    # Get token embeddings
    embed_layer = model.language_model.embed_tokens

    # Enable gradients for input_ids by converting to embeddings
    input_embeds = embed_layer(input_ids).detach()
    input_embeds.requires_grad_(True)

    # Forward pass with embeddings
    # Note: LLaVA processes image and text together, so we pass embeddings directly
    outputs = model(
        pixel_values=pixel_values,
        inputs_embeds=input_embeds,
        image_sizes=image_sizes,
        attention_mask=attention_mask
    )

    # Get logits and compute loss for target token
    logits = outputs.logits[:, -1, :]
    target_logit = logits[0, first_token_id]

    # Backward pass to get gradients
    target_logit.backward()

    # Get gradients with respect to input embeddings
    text_grads = input_embeds.grad

    # Compute attribution as gradient * input (gradient x input method)
    text_attrs = (text_grads * input_embeds).sum(dim=-1)

    # Sum across embedding dimension to get per-token attribution
    text_attrs_sum = text_attrs.squeeze(0).cpu().detach().numpy()

    # Get token strings
    tokens = [processor.tokenizer.decode([token_id]) for token_id in input_ids[0]]

    # Separate image and text tokens for analysis
    image_token_indices = [i for i, t in enumerate(tokens) if t.strip() == '<image>']
    text_token_indices = [i for i, t in enumerate(tokens)
                          if t.strip() not in ['<image>', '<s>', '</s>', '<pad>', '<|im_start|>', '<|im_end|>', 'user', '']]

    # Calculate attribution breakdown
    image_token_attr = sum(abs(text_attrs_sum[i]) for i in image_token_indices) if image_token_indices else 0
    text_token_attr = sum(abs(text_attrs_sum[i]) for i in text_token_indices) if text_token_indices else 0
    total_attr = image_token_attr + text_token_attr

    image_pct = (image_token_attr / total_attr * 100) if total_attr > 0 else 0
    text_pct = (text_token_attr / total_attr * 100) if total_attr > 0 else 0

    return {
        'predicted_answer': predicted_answer,
        'confidence': confidence_score,
        'text_attributions': text_attrs_sum,
        'tokens': tokens,
        'token_ids': input_ids[0].cpu().numpy(),
        'question': question,
        'image_token_attribution': image_token_attr,
        'text_token_attribution': text_token_attr,
        'image_token_percentage': image_pct,
        'text_token_percentage': text_pct
    }


def compute_joint_attributions(
    image_path,
    question,
    model,
    processor,
    n_steps=5
):
    """
    Compute joint image+text attributions showing relative contribution of each modality

    Args:
        image_path: Path to the image file
        question: Question text to analyze
        model: The vision-language model
        processor: The model's processor
        n_steps: Number of Integrated Gradients steps

    Returns:
        dict containing:
            - predicted_answer: The model's predicted answer
            - confidence: Confidence score
            - image_attribution_score: Total attribution from image
            - text_attribution_score: Total attribution from text
            - image_percentage: Percentage contribution from image
            - text_percentage: Percentage contribution from text
    """
    device = next(model.parameters()).device

    # Load image
    img = load_image_from_file(image_path)

    # Prepare conversation
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

    # Generate prediction
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

        # Get confidence
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

    # Compute image attributions
    def custom_forward_img(pixel_values):
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
        )
        next_token_logits = outputs.logits[:, -1, :]
        return next_token_logits

    pixel_values_baseline = torch.zeros_like(pixel_values).to(device)

    ig_img = IntegratedGradients(custom_forward_img)
    image_attrs = ig_img.attribute(
        inputs=pixel_values,
        baselines=pixel_values_baseline,
        target=first_token_id,
        n_steps=n_steps,
        internal_batch_size=1,
    )

    # Compute text attributions using gradient-based approach
    embed_layer = model.language_model.embed_tokens

    # Get text embeddings with gradients
    input_embeds = embed_layer(input_ids).detach()
    input_embeds.requires_grad_(True)

    # Forward pass with embeddings
    outputs_text = model(
        pixel_values=pixel_values,
        inputs_embeds=input_embeds,
        image_sizes=image_sizes,
        attention_mask=attention_mask
    )

    # Get logits and compute loss for target token
    logits_text = outputs_text.logits[:, -1, :]
    target_logit_text = logits_text[0, first_token_id]

    # Backward pass
    target_logit_text.backward()

    # Get text attributions
    text_grads = input_embeds.grad
    text_attrs = (text_grads * input_embeds).sum(dim=-1)

    # Calculate total attributions (sum of absolute values)
    image_attr_total = torch.abs(image_attrs).sum().item()
    text_attr_total = torch.abs(text_attrs).sum().item()

    total_attr = image_attr_total + text_attr_total

    # Calculate percentages
    image_percentage = (image_attr_total / total_attr * 100) if total_attr > 0 else 50.0
    text_percentage = (text_attr_total / total_attr * 100) if total_attr > 0 else 50.0

    return {
        'predicted_answer': predicted_answer,
        'confidence': confidence_score,
        'image_attribution_score': image_attr_total,
        'text_attribution_score': text_attr_total,
        'image_percentage': image_percentage,
        'text_percentage': text_percentage,
        'question': question
    }


def visualize_text_attributions(result, save_path=None):
    """
    Visualize text attributions with color-coded tokens

    Args:
        result: Result dictionary from compute_text_attributions
        save_path: Optional path to save the visualization
    """
    tokens = result['tokens']
    attributions = result['text_attributions']
    predicted_answer = result['predicted_answer']
    question = result['question']

    # Normalize attributions for color mapping
    abs_max = np.max(np.abs(attributions))
    if abs_max > 0:
        norm_attrs = attributions / abs_max
    else:
        norm_attrs = attributions

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Color-coded text
    ax1.axis('off')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    # Create colormap (blue = negative, white = zero, red = positive)
    cmap = LinearSegmentedColormap.from_list('attribution', ['blue', 'white', 'red'])

    # Display tokens with background colors
    x_pos = 0.05
    y_pos = 0.5

    for i, (token, attr) in enumerate(zip(tokens, norm_attrs)):
        # Skip special tokens and image tokens for cleaner visualization
        token_stripped = token.strip()
        if token_stripped in ['<s>', '</s>', '<pad>', '<image>', '<|im_start|>', '<|im_end|>', 'user', '']:
            continue

        # Get color based on attribution
        color = cmap((attr + 1) / 2)  # Map [-1, 1] to [0, 1]

        # Display token with colored background
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='black', linewidth=1)
        text_obj = ax1.text(x_pos, y_pos, token, fontsize=12, bbox=bbox_props, verticalalignment='center')

        # Get text width to position next token
        bbox = text_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox_data = bbox.transformed(ax1.transData.inverted())
        x_pos += bbox_data.width + 0.02

        # Wrap to next line if needed
        if x_pos > 0.95:
            x_pos = 0.05
            y_pos -= 0.15

    ax1.set_title(f"Text Attribution for Question: '{question}'\nPredicted Answer: '{predicted_answer}'",
                  fontsize=14, fontweight='bold', pad=20)

    # Add colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='horizontal', pad=0.1, aspect=30)
    cbar.set_label('Attribution (Blue = Negative, Red = Positive)', fontsize=10)

    # Plot 2: Bar chart of attributions
    # Filter out special tokens and image tokens
    valid_indices = [i for i, t in enumerate(tokens)
                     if t.strip() not in ['<s>', '</s>', '<pad>', '<image>', '<|im_start|>', '<|im_end|>', 'user', '']]
    filtered_tokens = [tokens[i] for i in valid_indices]
    filtered_attrs = [attributions[i] for i in valid_indices]

    # Truncate very long token lists
    if len(filtered_tokens) > 30:
        filtered_tokens = filtered_tokens[:30]
        filtered_attrs = filtered_attrs[:30]

    colors = ['red' if a > 0 else 'blue' for a in filtered_attrs]
    bars = ax2.barh(range(len(filtered_tokens)), filtered_attrs, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(filtered_tokens)))
    ax2.set_yticklabels(filtered_tokens, fontsize=9)
    ax2.set_xlabel('Attribution Score', fontsize=11)
    ax2.set_title('Token Attribution Scores', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Text attribution visualization saved to {save_path}")

    plt.show()


def visualize_joint_attributions(result, save_path=None):
    """
    Visualize joint image+text attribution comparison

    Args:
        result: Result dictionary from compute_joint_attributions
        save_path: Optional path to save the visualization
    """
    predicted_answer = result['predicted_answer']
    confidence = result['confidence']
    image_pct = result['image_percentage']
    text_pct = result['text_percentage']
    question = result['question']

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Pie chart
    sizes = [image_pct, text_pct]
    labels = [f'Image\n{image_pct:.1f}%', f'Text\n{text_pct:.1f}%']
    colors = ['#ff9999', '#66b3ff']
    explode = (0.05, 0.05)

    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='',
            shadow=True, startangle=90, textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax1.set_title(f"Modality Contribution\nPrediction: '{predicted_answer}' ({confidence*100:.1f}% confidence)",
                  fontsize=14, fontweight='bold')

    # Plot 2: Bar chart
    modalities = ['Image', 'Text']
    percentages = [image_pct, text_pct]
    bars = ax2.bar(modalities, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Attribution (%)', fontsize=12)
    ax2.set_title('Attribution Percentage by Modality', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)

    # Add percentage labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add question as subtitle
    fig.suptitle(f"Question: '{question}'", fontsize=12, y=0.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Joint attribution visualization saved to {save_path}")

    plt.show()


def text_vqa_interpret(
    image_path,
    question,
    model,
    processor,
    mode='both',
    n_steps=10,
    show_visualizations=True,
    save_dir=None
):
    """
    Unified function for text attribution analysis

    Args:
        image_path: Path to the image file
        question: Question to analyze
        model: The vision-language model
        processor: The model's processor
        mode: 'text' for text-only, 'joint' for image vs text, 'both' for both analyses
        n_steps: Number of Integrated Gradients steps
        show_visualizations: Whether to display visualizations
        save_dir: Optional directory to save visualizations

    Returns:
        dict with results based on mode
    """
    results = {}

    if mode in ['text', 'both']:
        print("\n" + "="*70)
        print("COMPUTING TEXT ATTRIBUTIONS")
        print("="*70)

        text_result = compute_text_attributions(
            image_path, question, model, processor, n_steps
        )
        results['text'] = text_result

        print(f"\nPredicted Answer: '{text_result['predicted_answer']}'")
        print(f"Confidence: {text_result['confidence']:.4f} ({text_result['confidence']*100:.2f}%)")
        print(f"\nToken Attribution Breakdown:")
        print(f"  Image Tokens: {text_result['image_token_percentage']:.1f}%")
        print(f"  Text Tokens: {text_result['text_token_percentage']:.1f}%")

        if show_visualizations:
            save_path = f"{save_dir}/text_attribution.png" if save_dir else None
            visualize_text_attributions(text_result, save_path)

    if mode in ['joint', 'both']:
        print("\n" + "="*70)
        print("COMPUTING JOINT IMAGE+TEXT ATTRIBUTIONS")
        print("="*70)

        joint_result = compute_joint_attributions(
            image_path, question, model, processor, n_steps
        )
        results['joint'] = joint_result

        print(f"\nPredicted Answer: '{joint_result['predicted_answer']}'")
        print(f"Confidence: {joint_result['confidence']:.4f} ({joint_result['confidence']*100:.2f}%)")
        print(f"\nImage Contribution: {joint_result['image_percentage']:.1f}%")
        print(f"Text Contribution: {joint_result['text_percentage']:.1f}%")

        if show_visualizations:
            save_path = f"{save_dir}/joint_attribution.png" if save_dir else None
            visualize_joint_attributions(joint_result, save_path)

    print("\n" + "="*70)
    print("✓ TEXT ATTRIBUTION ANALYSIS COMPLETE")
    print("="*70 + "\n")

    return results
