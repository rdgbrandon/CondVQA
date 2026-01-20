"""VLM Integrated Gradients - Vision-Language Model Explainability"""

from .model_loader import load_model
from .image_interpreter import vqa_interpret, compute_single_image_attribution
from .frame_extension import temporal_vqa_interpret, extract_frames
from .text_interpreter import text_vqa_interpret, compute_text_attributions, compute_joint_attributions
from .utils import load_image_from_file, setup_colormap

__all__ = [
    'load_model',
    'vqa_interpret',
    'compute_single_image_attribution',
    'temporal_vqa_interpret',
    'extract_frames',
    'text_vqa_interpret',
    'compute_text_attributions',
    'compute_joint_attributions',
    'load_image_from_file',
    'setup_colormap'
]
