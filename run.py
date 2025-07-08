import torch
import numpy as np
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
from tqdm import tqdm

import matplotlib.pyplot as plt

# imports for diffusion loading
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from diffusers import LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer,CLIPProcessor, CLIPModel
from generic_diffusion_feature.feature import diffusion_feature
from PIL import Image


class DetonateT2IAlignmentDataset:

    """
        Dataset class for Detonate T2I Alignment dataset
        columns: Prompt (str), Chosen (PIL Image objects), Rejected (PIL image objects), Category (str)
    """

    def __init__(self, dataset_name="DetonateT2I/T2I_Alignment_Detonate", sample_size=100):
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.data = self._download_samples()
    def _download_samples(self):
        """Download 100 data samples from HuggingFace"""
        # TODO: Implement HF dataset download
        dataset = load_dataset(self.dataset_name, streaming=True, cache_dir="./dataset_cache")
        # columns --> Prompt (str), Chosen (PIL Image objects), Rejected (PIL image objects), Category (str)
        self.data = dataset['train'].take(self.sample_size)
        return self.data

class StableDiffusionVariant():
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", clip_model_name="openai/clip-vit-large-patch14"):
        #"runwayml/stable-diffusion-v1-5"
        # 1. autoencoder (vae) : used to decode the latents into image space
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", cache_dir="./models_cache")
        # 2. tokenizer, text encoder: encoding the prompts into text embeddings
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name, cache_dir="./models_cache")
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name, cache_dir="./models_cache")
        # 3. unet: the main model that does the denoising
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet", cache_dir="./models_cache")        
        # 4. scheduler: used to schedule the denoising process
        self.scheduler = LMSDiscreteScheduler.from_pretrained(model_name, subfolder="scheduler", cache_dir="./models_cache")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.vae.to(self.device)
        self.text_encoder.to(self.device)
        self.unet.to(self.device)
  


def set_diffusion_feature_extractor():
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    print(f"Using device: {device}")
    df = diffusion_feature.FeatureExtractor(
        layer={
            'up-level1-repeat1-vit-block0-cross-q': True,  # a feature within ViT
            'up-level2-repeat1-vit-block0-cross-map': True,  # an attention score map
            'up-level2-upsampler-out': True,  # a feature between two blocks, aka a conventional feature
        },
        version='1-5',
        img_size=512,
        device='cuda',
    )   
    return df

def load_image(image_path):
    img = Image.open(image_path)

def test_diffusion_feature_extractor(image_path, prompt):
    df = set_diffusion_feature_extractor()
    image = load_image(image_path)
    prompt = df.encode_prompt(prompt)
    extracted_features = df.extract(image, prompt)
    for k,v in features.items():
        print(f"{k}: {v.shape}")


if __name__ == "__main__":
    # experiment = DetonateExperiment()
    # experiment.run_experiment()

    # loading n samples from Detonate T2I Alignment dataset (HF)
    # detonate_t2i_dataset = DetonateT2IAlignmentDataset(dataset_name="DetonateT2I/T2I_Alignment_Detonate", sample_size=10)

    # loading Stable Diffusion model variant
    # sd_1_4 = StableDiffusionVariant(model_name="CompVis/stable-diffusion-v1-4", clip_model_name="openai/clip-vit-large-patch14")
    # sd_1_5 = StableDiffusionVariant(model_name="runwayml/stable-diffusion-v1-5", clip_model_name="openai/clip-vit-large-patch14")
    cat_image_path = "/Users/hasnatmdabdullah/Documents/Developer/SDInferSafetyAnalaysis/cat.png"
    cat_image_prompt = "A cat is sitting there"
    test_diffusion_feature_extractor(cat_image_path, cat_image_prompt)

