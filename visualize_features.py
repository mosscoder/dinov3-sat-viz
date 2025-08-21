from typing import Tuple, Optional, Union, Dict, Any
import torch, torch.nn as nn, numpy as np
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModel
from datasets import load_dataset

class SatelliteDINOv3FeatureExtractor:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(torch.device(device if torch.cuda.is_available() else 'cpu')).eval()
        self.device = self.model.device
        self.patch_size, self.hidden_dim, self.num_layers = 16, 1024, 24
        
    
    def make_transform(self, resize_height: int = 224, resize_width: int = 224):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize_height, resize_width), antialias=True),
            transforms.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143))
        ])
    def prepare_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        target_size: Optional[int] = None,
        alpha_handling: str = 'composite_white'
    ) -> Tuple[Image.Image, torch.Tensor, Tuple[int, int]]:
        if isinstance(image, str): image = Image.open(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2: image = Image.fromarray(image, mode='L').convert('RGB')
            elif image.ndim == 3:
                if image.shape[2] == 3: image = Image.fromarray(image, mode='RGB')
                elif image.shape[2] == 4: image = Image.fromarray(image, mode='RGBA')
                else: image = Image.fromarray(image[:, :, :3].astype(np.uint8), mode='RGB')
        
        if image.mode == 'RGBA':
            bg_color = (255, 255, 255) if alpha_handling == 'composite_white' else (0, 0, 0) if alpha_handling == 'composite_black' else None
            if bg_color:
                background = Image.new('RGB', image.size, bg_color)
                background.paste(image, mask=image.split()[3])
                image = background
            else: image = image.convert('RGB')
        elif image.mode not in ['RGB']: image = image.convert('RGB')
        
        if target_size is None:
            w, h = image.size
            target_w, target_h = (w // self.patch_size) * self.patch_size, (h // self.patch_size) * self.patch_size
        else:
            target_w = target_h = (target_size // self.patch_size) * self.patch_size
        img_tensor = self.make_transform(target_h, target_w)(image).unsqueeze(0).to(self.device)
        _, _, height, width = img_tensor.shape
        return image, img_tensor, (height // self.patch_size, width // self.patch_size)
    
    def extract_features(
        self,
        img_tensor: torch.Tensor,
        layer_idx: int = -1,
        include_cls: bool = False
    ) -> Dict[str, Any]:
        if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
        batch_size, channels, height, width = img_tensor.shape
        
        with torch.no_grad():
            output = self.model(img_tensor)
            if isinstance(output, torch.Tensor): features = output
            elif isinstance(output, dict): features = output.get('last_hidden_state', output.get('hidden_states', output.get('x', next(v for v in output.values() if isinstance(v, torch.Tensor) and v.dim() >= 2))))
            elif isinstance(output, (tuple, list)): features = output[0]
            else: raise ValueError(f"Unexpected output type: {type(output)}")
        
        features = features.cpu().numpy()
        if features.ndim == 2: features = features[np.newaxis, :, :]
        elif features.ndim != 3: raise ValueError(f"Unexpected features shape: {features.shape}")
        batch_size, num_tokens, feature_dim = features.shape
        expected_patches = (height // self.patch_size) * (width // self.patch_size)
        features_dict = {'features': features[0], 'shape': features.shape, 'num_tokens': num_tokens, 'feature_dim': feature_dim, 'num_patches': expected_patches, 'actual_tokens': num_tokens}
        extra_tokens = features_dict['features'].shape[0] - expected_patches
        if extra_tokens > 0: features_dict['features'] = features_dict['features'][extra_tokens:, :]
        return features_dict

def perform_pca_analysis(
    features: np.ndarray,
    n_components: int = 10,
    whiten: bool = False
) -> Tuple[PCA, np.ndarray]:
    features_norm = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-7)
    pca = PCA(n_components=n_components, whiten=whiten)
    return pca, pca.fit_transform(features_norm)


def visualize_pca_results_simple(
    pca_features: np.ndarray,
    patch_shape: Tuple[int, int],
    original_image: Image.Image,
    save_path: str
):
    n_h, n_w = patch_shape
    pca_grid = pca_features.reshape(n_h, n_w, -1, order='C')
    
    def normalize_percentile(pc, p_low=5, p_high=95):
        p_low_val, p_high_val = np.percentile(pc, [p_low, p_high])
        return np.clip((pc - p_low_val) / (p_high_val - p_low_val + 1e-7), 0, 1) if p_high_val - p_low_val > 1e-7 else np.zeros_like(pc)
    
    from io import BytesIO
    fig1 = plt.figure(figsize=(10, 5))
    gs1 = gridspec.GridSpec(1, 2, figure=fig1, wspace=0.02)
    
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.imshow(original_image)
    ax1.set_title('Original RGB Image', fontsize=14, pad=10)
    ax1.axis('off')
    
    if pca_grid.shape[2] >= 3:
        ax2 = fig1.add_subplot(gs1[0, 1])
        rgb = np.stack([normalize_percentile(pca_grid[:, :, i]) for i in range(3)], axis=2)
        ax2.imshow(rgb)
        ax2.set_title('RGB Composite (PC1-PC2-PC3)', fontsize=14, pad=10)
        ax2.axis('off')
    plt.tight_layout()
    
    buf1 = BytesIO()
    fig1.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
    buf1.seek(0)
    plt.close(fig1)
    
    fig2 = plt.figure(figsize=(10, 3.33))
    gs2 = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.02)
    
    for i, title in enumerate(['PC1', 'PC2', 'PC3']):
        if i < pca_grid.shape[2]:
            ax = fig2.add_subplot(gs2[0, i])
            pc_vis = normalize_percentile(pca_grid[:, :, i], 1, 99)
            ax.imshow(pc_vis, cmap='viridis', vmin=0, vmax=1)
            ax.set_title(title, fontsize=14, pad=10)
            ax.axis('off')
    plt.tight_layout()
    
    buf2 = BytesIO()
    fig2.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
    buf2.seek(0)
    plt.close(fig2)
    
    from PIL import Image as PILImage
    img1, img2 = PILImage.open(buf1), PILImage.open(buf2)
    combined = PILImage.new('RGB', (max(img1.width, img2.width), img1.height + img2.height), 'white')
    combined.paste(img1, (0, 0))
    combined.paste(img2, (0, img1.height))
    combined.save(save_path, dpi=(150, 150))
    buf1.close()
    buf2.close()
    print(f"✓ Visualization saved to {save_path}")
    return pca_grid


def run_satellite_dinov3_pca(
    model: nn.Module,
    image: Union[str, Image.Image, np.ndarray],
    target_size: Optional[int] = 512,
    n_components: int = 10,
    layer_idx: int = -1,
    alpha_handling: str = 'composite_white',
    output_path: str = 'results/pca_features.png'
) -> Dict[str, Any]:
    print("Initializing DINOv3 feature extractor...")
    extractor = SatelliteDINOv3FeatureExtractor(model)
    
    print("Preparing image for processing...")
    processed_image, img_tensor, patch_shape = extractor.prepare_image(image, target_size, alpha_handling=alpha_handling)
    
    print("Extracting features from model...")
    features = extractor.extract_features(img_tensor, layer_idx=layer_idx)['features']
    expected_patches = patch_shape[0] * patch_shape[1]
    if features.shape[0] < expected_patches: raise ValueError(f"Too few patches: {features.shape[0]} < {expected_patches}")
    
    print(f"Performing PCA analysis with {n_components} components...")
    pca, pca_features = perform_pca_analysis(features, n_components=n_components)
    
    print("Creating visualizations...")
    pca_grid = visualize_pca_results_simple(pca_features, patch_shape, processed_image, save_path=output_path)
    
    print(f"✓ Analysis complete! Results saved to {output_path}")
    
    return {'processed_image': processed_image, 'features': features, 'pca': pca, 'pca_features': pca_features, 'pca_grid': pca_grid, 'patch_shape': patch_shape}

def main():
    print("Loading dataset...")
    dataset = load_dataset("mpg-ranch/light-stable-semantics", split="train")
    image = dataset[1]['image_t1']

    print("Loading satellite DINOv3 model...")
    model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m").to(torch.device("cpu"))
    
    print("Starting PCA analysis...")
    results = run_satellite_dinov3_pca(
        model=model,
        image=image,
        target_size=1024,
        n_components=3,
        layer_idx=-1,
        alpha_handling='composite_white',
        output_path='results/pca_features.png'
    )
    
    print("\n✓ All done!")
    return results

if __name__ == "__main__":
    main()

