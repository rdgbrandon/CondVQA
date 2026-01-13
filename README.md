# VLM Integrated Gradients

Explain Vision-Language Model predictions using Integrated Gradients attribution method.

This project uses the LLaVA vision-language model with Captum's Integrated Gradients to provide visual explanations for model predictions on Visual Question Answering tasks.

## Features

- Vision-Language Model (LLaVA) for Visual Question Answering
- Integrated Gradients for explainability
- Heatmap visualizations showing which parts of the image influence predictions
- Optimized for Google Colab with 4-bit quantization

## Project Structure

```
VLM_IG/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── model_loader.py       # Model loading utilities
│   ├── utils.py              # Image processing utilities
│   └── interpreter.py        # Main VQA interpretation logic
├── run_colab.ipynb           # Simplified Colab notebook
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Usage on Google Colab

### Step 1: Setup Repository

1. Push this directory to your GitHub repository
2. Open Google Colab: https://colab.research.google.com
3. Create a new notebook or upload `run_colab.ipynb`

### Step 2: Enable GPU

1. Go to `Runtime > Change runtime type`
2. Select `GPU` as Hardware accelerator
3. Click `Save`

### Step 3: Run the Notebook

Follow the cells in `run_colab.ipynb`:

1. Clone your repository
2. Install dependencies (restart runtime after numpy installation)
3. Import libraries and load the model
4. Upload an image from your computer
5. Define questions and run analysis

### Example Usage

```python
# After loading the model, upload an image and run:
questions = [
    "What animal is in this image?",
    "What color is the animal?",
    "Where is the animal located?"
]

vqa_interpret(
    image_path="your_uploaded_image.jpg",
    questions=questions,
    model=model,
    processor=processor,
    show_top_k=10
)
```

## Local Development (VS Code)

You can edit the code locally in VS Code and push changes to GitHub. The modular structure makes it easy to:

- Modify model parameters in `src/model_loader.py`
- Add new utility functions in `src/utils.py`
- Enhance the interpretation logic in `src/interpreter.py`

After making changes:
```bash
git add .
git commit -m "Your changes"
git push origin main
```

Then in Colab, pull the latest changes:
```bash
!git pull origin main
```

## How It Works

1. **Model**: Uses LLaVA-OneVision (0.5B parameters) with 4-bit quantization
2. **Attribution**: Integrated Gradients computes pixel-wise attributions by interpolating between a baseline (black image) and the actual image
3. **Visualization**: Heatmaps show which image regions most influenced the model's prediction

## Parameters

### `vqa_interpret()`

- `image_path` (str): Path to the uploaded image file
- `questions` (list): List of questions to ask about the image
- `model`: Loaded LLaVA model
- `processor`: Model processor
- `show_top_k` (int): Number of top probable answers to display (default: 10)

## Requirements

- Google Colab with GPU runtime (recommended: T4, V100, or A100)
- Python 3.8+
- See `requirements.txt` for package dependencies

## Memory Optimization

The code includes several optimizations for Colab's memory constraints:
- 4-bit model quantization
- Reduced Integrated Gradients steps (n_steps=5)
- Automatic CUDA cache clearing between questions
- Image resizing to 384x384 pixels

## Troubleshooting

### Out of Memory Error
- Restart runtime and clear all outputs
- Reduce the number of questions
- Use a simpler image (lower resolution)

### Model Loading Issues
- Ensure GPU is enabled in runtime settings
- Check that all dependencies are installed
- Restart runtime after numpy installation

### Attribution Computation Slow
- This is normal; Integrated Gradients requires multiple forward passes
- Each question takes 1-2 minutes depending on GPU

## License

This project uses the following open-source components:
- LLaVA model from HuggingFace
- Captum for Integrated Gradients
- Transformers library

## Citation

If you use this code, please consider citing:
```
LLaVA-OneVision: https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
Captum: https://captum.ai/
```
