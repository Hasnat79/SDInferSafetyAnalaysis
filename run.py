import torch
import numpy as np
from datasets import load_dataset
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
from pathlib import Path
import pickle
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    df = diffusion_feature.FeatureExtractor(
        layer={
            'up-level1-repeat1-vit-block0-cross-q': True,  # a feature within ViT
            'up-level2-repeat1-vit-block0-cross-map': True,  # an attention score map
            'up-level2-upsampler-out': True,  # a feature between two blocks, aka a conventional feature
        },
        version='1-5',
        img_size=512,
        device=device,
    )   
    return df

def load_image(image_path):
    img = Image.open(image_path)
    return img 

def calculate_centroids(extracted_safe_unsafe_diffusion_features_per_sample):
    """
    Calculate centroids for safe and unsafe features
    
    Args:
        extracted_safe_unsafe_diffusion_features_per_sample: Dictionary containing safe and unsafe features per sample
        
    Returns:
        tuple: (safe_centroid, unsafe_centroid)
    """
    assert False, "calculate_centroids needs to implement"

def get_intermediate_images_of_diffusion_inference(sd_model, prompt, n_steps=50, p_step=10):
    """
    Generate intermediate images during diffusion inference process
    
    Args:
        sd_model: StableDiffusionVariant model instance
        prompt: Text prompt for generation
        n_steps: Total number of inference steps
        p_step: Step interval for capturing intermediate images
        
    Returns:
        list: List of intermediate images at each p_step interval
    """
    assert False, "get_intermediate_images_of_diffusion_inference needs to implement"

def get_image_tokens(image, sd_model):
    """
    Extract image tokens/features from an image using the diffusion model
    
    Args:
        image: PIL Image or tensor
        sd_model: StableDiffusionVariant model instance
        
    Returns:
        torch.Tensor: Image tokens/features
    """
    assert False, "get_image_tokens needs to implement"

def calculate_distance_from_centroid(image_tokens, centroid):
    """
    Calculate distances between image tokens and a given centroid
    
    Args:
        image_tokens: Tensor of image tokens/features
        centroid: Centroid tensor to calculate distances from
        
    Returns:
        tuple: (distances_per_token, mean_distance)
    """
    assert False, "calculate_distance_from_centroid needs to implement"

def load_detonate_t2I_alignment_dataset(sample_size=100):
    """
    Load the Detonate T2I Alignment dataset
    
    Args:
        sample_size: Number of samples to load
        
    Returns:
        DetonateT2IAlignmentDataset: Dataset instance
    """
    cache_dir = "./dataset_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"detonate_t2i_alignment_{sample_size}.pkl")
    
    if os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file}")
        with open(cache_file, "rb") as f:
            detonate_t2i_dataset = pickle.load(f)
        print(type(detonate_t2i_dataset))
        print(f"detonate_t2i_dataset.data: {detonate_t2i_dataset.data}")
    else:
        print(f"Creating new dataset and caching to {cache_file}")
        detonate_t2i_dataset = DetonateT2IAlignmentDataset(dataset_name="DetonateT2I/T2I_Alignment_Detonate", sample_size=sample_size)
        with open(cache_file, "wb") as f:
            pickle.dump(detonate_t2i_dataset, f)
    
    return detonate_t2i_dataset

def extract_diffusion_features(diffusion_feature_extractor,sample):
    """
    Extract diffusion features from a dataset sample
    
    Args:
        sample: Dataset sample containing prompt, chosen and rejected images
        images are PIL Image objects
        
    Returns:
        tuple: (extracted_safe_features, extracted_unsafe_features)
        both are dictionaries containing extracted features
       keys: (['up-level1-repeat1-vit-block0-cross-q', 'up-level2-repeat1-vit-block0-cross-map']
    """
    
    prompt = diffusion_feature_extractor.encode_prompt(sample['Prompt'])
    chosen_safe_image = sample['Chosen'] 
    rejected_unsafe_image = sample['Rejected'] 

    # Extract features for safe and unsafe images
    extracted_safe_features = diffusion_feature_extractor.extract(prompt, batch_size=1, image=[chosen_safe_image])
    extracted_unsafe_features = diffusion_feature_extractor.extract(prompt, batch_size=1, image=[rejected_unsafe_image])
    
    return extracted_safe_features, extracted_unsafe_features

def plot_safe_unsafe_clusters(extracted_safe_unsafe_diffusion_features_per_sample):
    """
    Plot clusters of safe and unsafe features for visualization using 3D PCA
    
    Args:
        extracted_safe_unsafe_diffusion_features_per_sample: Dictionary containing features per sample
        key name for feature extraction from safe/unsafe: ['up-level1-repeat1-vit-block0-cross-q', 'up-level2-repeat1-vit-block0-cross-map']
    """
    print(f"extracted_safe_unsafe_diffusion_features_per_sample[0]: {extracted_safe_unsafe_diffusion_features_per_sample[0]['safe'].keys()}")
    
    # Feature keys to visualize
    feature_keys = ['up-level1-repeat1-vit-block0-cross-q']
    
    for feature_key in feature_keys:
        print(f"\nProcessing feature: {feature_key}")
        
        # Collect all safe and unsafe features for this feature key
        all_safe_features = []
        all_unsafe_features = []
        sample_indices = []
        
        for sample_id, value in extracted_safe_unsafe_diffusion_features_per_sample.items():
            if feature_key in value['safe'] and feature_key in value['unsafe']:
                safe_features = value['safe'][feature_key]  # shape: torch.Size([1, 1280, 16, 16]) or similar
                unsafe_features = value['unsafe'][feature_key]
                
                print(f"Sample {sample_id} - Safe features shape: {safe_features.shape}")
                print(f"Sample {sample_id} - Unsafe features shape: {unsafe_features.shape}")
                
                # Flatten features for PCA (keeping batch dimension and flattening spatial dimensions)
                if len(safe_features.shape) == 4:  # [batch, channels, height, width]
                    safe_flat = safe_features.reshape(safe_features.shape[0], -1)  # [batch, channels*height*width]
                    unsafe_flat = unsafe_features.reshape(unsafe_features.shape[0], -1)
                elif len(safe_features.shape) == 3:  # [batch, seq_len, features]
                    safe_flat = safe_features.reshape(safe_features.shape[0], -1)
                    unsafe_flat = unsafe_features.reshape(unsafe_features.shape[0], -1)
                else:
                    safe_flat = safe_features
                    unsafe_flat = unsafe_features
                
                # Convert to numpy if tensor
                if torch.is_tensor(safe_flat):
                    safe_flat = safe_flat.cpu().numpy()
                if torch.is_tensor(unsafe_flat):
                    unsafe_flat = unsafe_flat.cpu().numpy()
                
                all_safe_features.append(safe_flat)
                all_unsafe_features.append(unsafe_flat)
                sample_indices.append(sample_id)
        
        if not all_safe_features:
            print(f"No features found for key: {feature_key}")
            continue
            
        # Concatenate all features
        safe_features_matrix = np.vstack(all_safe_features)  # [n_samples, n_features]
        unsafe_features_matrix = np.vstack(all_unsafe_features)
        
        print(f"Safe features matrix shape: {safe_features_matrix.shape}")
        print(f"Unsafe features matrix shape: {unsafe_features_matrix.shape}")
        
        # Combine safe and unsafe features for PCA
        combined_features = np.vstack([safe_features_matrix, unsafe_features_matrix])
        
        # Apply PCA to reduce to 3 dimensions
        pca = PCA(n_components=3)
        features_3d = pca.fit_transform(combined_features)
        
        # Split back into safe and unsafe
        n_safe = safe_features_matrix.shape[0]
        safe_features_3d = features_3d[:n_safe]
        unsafe_features_3d = features_3d[n_safe:]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot safe features in blue
        ax.scatter(safe_features_3d[:, 0], safe_features_3d[:, 1], safe_features_3d[:, 2], 
                  c='blue', label='Safe', alpha=0.7, s=50)
        
        # Plot unsafe features in red
        ax.scatter(unsafe_features_3d[:, 0], unsafe_features_3d[:, 1], unsafe_features_3d[:, 2], 
                  c='red', label='Unsafe', alpha=0.7, s=50)
        
        # Add labels for each point
        # for i, sample_id in enumerate(sample_indices):
        #     ax.text(safe_features_3d[i, 0], safe_features_3d[i, 1], safe_features_3d[i, 2], 
        #            f'S{sample_id}', fontsize=8)
        #     ax.text(unsafe_features_3d[i, 0], unsafe_features_3d[i, 1], unsafe_features_3d[i, 2], 
        #            f'U{sample_id}', fontsize=8)
        
        # Customize plot
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
        ax.set_title(f'3D PCA Visualization of Safe vs Unsafe Features\nFeature: {feature_key}')
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(f'safe_unsafe_pca_3d_{feature_key.replace("-", "_")}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.savefig(f'safe_unsafe_pca_3d_{feature_key.replace("-", "_")}.pdf', dpi=300, bbox_inches='tight')
        
        # Print PCA info
        print(f"\nPCA Results for {feature_key}:")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        print(f"Original feature dimension: {combined_features.shape[1]}")
        print(f"Reduced to 3D")
        
        # Calculate and print distances between safe and unsafe centroids
        safe_centroid = np.mean(safe_features_3d, axis=0)
        unsafe_centroid = np.mean(unsafe_features_3d, axis=0)
        centroid_distance = np.linalg.norm(safe_centroid - unsafe_centroid) # unit: Euclidean distance
        
        print(f"Safe centroid (3D): {safe_centroid}")
        print(f"Unsafe centroid (3D): {unsafe_centroid}")
        print(f"Distance between centroids: {centroid_distance:.4f}")
        
        print("-" * 50)

        # Create rotating GIF
        print(f"Creating rotating GIF for {feature_key}")
        angles = np.linspace(0, 360, 60)  # 60 frames for smooth rotation
        gif_frames = []

        for angle in angles:
            # Create a new figure for each frame
            fig_gif = plt.figure(figsize=(10, 8))
            ax_gif = fig_gif.add_subplot(111, projection='3d')
            
            # Plot the same data
            ax_gif.scatter(safe_features_3d[:, 0], safe_features_3d[:, 1], safe_features_3d[:, 2], 
                          c='blue', label='Safe', alpha=0.7, s=50)
            ax_gif.scatter(unsafe_features_3d[:, 0], unsafe_features_3d[:, 1], unsafe_features_3d[:, 2], 
                          c='red', label='Unsafe', alpha=0.7, s=50)
            
            # Set the viewing angle
            ax_gif.view_init(elev=45, azim=angle)
            
            # Customize plot
            ax_gif.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax_gif.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax_gif.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
            ax_gif.set_title(f'Safe vs Unsafe Features - {feature_key}')
            ax_gif.legend()
            ax_gif.grid(True, alpha=0.3)
            
            # Save frame to memory
            fig_gif.canvas.draw()
            buf = fig_gif.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frame = frame[:, :, :3]  # Remove alpha channel to get RGB
            gif_frames.append(Image.fromarray(frame))
            
            plt.close(fig_gif)

        # Save as GIF
        gif_path = f'safe_unsafe_pca_3d_{feature_key.replace("-", "_")}_rotating.gif'
        gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], 
                           duration=100, loop=0)
        print(f"Rotating GIF saved as {gif_path}")

def test_diffusion_feature_extractor(image_path, prompt):
    diffusion_feature_extractor = set_diffusion_feature_extractor()
    img = load_image(image_path)
    prompt = diffusion_feature_extractor.encode_prompt(prompt)
    extracted_features = diffusion_feature_extractor.extract(prompt,batch_size=1,image=[img])
    print(f"len(extracted_features): {len(extracted_features)}")
    for k,v in extracted_features.items():
        print(f"{k}: {v.shape}")
        # up-level1-repeat1-vit-block0-cross-q: torch.Size([1, 1280, 16, 16])
        # up-level2-repeat1-vit-block0-cross-map: torch.Size([1, 8, 1024, 77])

def run_diffusion_inference_analysis(extracted_safe_unsafe_diffusion_features_per_sample):
    """
        extracted_safe_unsafe_diffusion_features_per_sample:{
            id: {
                "safe": extracted_safe_features,
                "unsafe": extracted_unsafe_features
        }
    """
    
    safe_centroid, unsafe_centroid = calculate_centroids(extracted_safe_unsafe_diffusion_features_per_sample)
    sd_1_5 = StableDiffusionVariant(model_name="runwayml/stable-diffusion-v1-5", clip_model_name="openai/clip-vit-large-patch14")
    
    for sample_id,values in extracted_safe_unsafe_diffusion_features_per_sample.items():
        prompt = values['prompt']
        # we will be expecting 5 intermediate images for each p_step: n = 50 , p_step = 10 --> 50/10 =5 images
        # 'list' of image (obj/tensor) for each p_step
        intermediate_images_after_each_p_step = get_intermediate_images_of_diffusion_inference(sd_1_5,prompt, n_steps=50,p_step=10)

        # traverse_each_image 
        for image in intermediate_images_after_each_p_step:
            image_tokens = get_image_tokens(image, sd_1_5)

            distance_between_each_token_and_safe_centroid, mean_distance_safe = calculate_distance_from_centroid(image_tokens, safe_centroid)
            distance_between_each_token_and_unsafe_centroid, mean_distance_unsafe = calculate_distance_from_centroid(image_tokens, unsafe_centroid)

            print(f"Sample ID: {sample_id}, Image: {image}, Safe Distance: {mean_distance_safe.item()}, Unsafe Distance: {mean_distance_unsafe.item()}")

        break


def run_inference_safety_analysis():
    detonate_t2i_dataset = load_detonate_t2I_alignment_dataset(sample_size=500)
    
    extracted_safe_unsafe_diffusion_features_per_sample = {}
    id = 0
    # traverse samples

    if not os.path.exists(Path("extracted_safe_unsafe_diffusion_features_per_sample.pkl")):
        diffusion_feature_extractor = set_diffusion_feature_extractor()
        for sample in tqdm(detonate_t2i_dataset.data):
            extracted_safe_features, extracted_unsafe_features = extract_diffusion_features(diffusion_feature_extractor,sample)
            
            # Create entry for current sample
            current_sample_data = {
                "prompt": sample['Prompt'],
                "safe": extracted_safe_features,
                "unsafe": extracted_unsafe_features
            }
            
            # Load existing data if file exists
            if os.path.exists(Path("extracted_safe_unsafe_diffusion_features_per_sample.pkl")):
                with open(Path("extracted_safe_unsafe_diffusion_features_per_sample.pkl"), "rb") as f:
                    extracted_safe_unsafe_diffusion_features_per_sample = pickle.load(f)
            
            # Add current sample
            extracted_safe_unsafe_diffusion_features_per_sample[id] = current_sample_data
            
            # Save updated data
            with open(Path("extracted_safe_unsafe_diffusion_features_per_sample.pkl"), "wb") as f:
                pickle.dump(extracted_safe_unsafe_diffusion_features_per_sample, f)
            
            # Clear memory
            del current_sample_data, extracted_safe_features, extracted_unsafe_features
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            id += 1
        
        print("Saved extracted_safe_unsafe_diffusion_features_per_sample to extracted_safe_unsafe_diffusion_features_per_sample.pkl")
    
    # load the features
    with open(Path("extracted_safe_unsafe_diffusion_features_per_sample.pkl"), "rb") as f:
        extracted_safe_unsafe_diffusion_features_per_sample = pickle.load(f)
    print("Loaded extracted_safe_unsafe_diffusion_features_per_sample from extracted_safe_unsafe_diffusion_features_per_sample_500.pkl")

    

    plot_safe_unsafe_clusters(extracted_safe_unsafe_diffusion_features_per_sample)

    run_diffusion_inference_analysis(extracted_safe_unsafe_diffusion_features_per_sample)



if __name__ == "__main__":
    # experiment = DetonateExperiment()
    # experiment.run_experiment()

    # loading n samples from Detonate T2I Alignment dataset (HF)
    # detonate_t2i_dataset = DetonateT2IAlignmentDataset(dataset_name="DetonateT2I/T2I_Alignment_Detonate", sample_size=10)

    # loading Stable Diffusion model variant
    # sd_1_4 = StableDiffusionVariant(model_name="CompVis/stable-diffusion-v1-4", clip_model_name="openai/clip-vit-large-patch14")
    # sd_1_5 = StableDiffusionVariant(model_name="runwayml/stable-diffusion-v1-5", clip_model_name="openai/clip-vit-large-patch14")
    cat_image_path = "cat.png"
    cat_image_prompt = "A cat is sitting there"
    # test_diffusion_feature_extractor(cat_image_path, cat_image_prompt)

    # run inference safety analysis
    run_inference_safety_analysis()
