"""
Activation Maximization for Global Explanations
Generates synthetic images that maximize class predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class ActivationMaximization:
    """
    Generate images that maximize activation for a target class
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        input_size: Tuple[int, int, int] = (3, 224, 224),
        device: str = 'cuda'
    ):
        """
        Initialize Activation Maximization
        
        Args:
            model: Trained PyTorch model
            input_size: (C, H, W) input size
            device: Device to use
        """
        self.model = model
        self.input_size = input_size
        self.device = device
        self.model.eval()
    
    def generate(
        self,
        target_class: int,
        num_iterations: int = 500,
        learning_rate: float = 0.1,
        l2_reg: float = 1e-4,
        blur_frequency: int = 10,
        blur_sigma: float = 1.0,
        jitter_pixels: int = 4,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate image that maximizes target class activation
        
        Args:
            target_class: Class to maximize
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for gradient ascent
            l2_reg: L2 regularization strength
            blur_frequency: Apply blur every N iterations
            blur_sigma: Gaussian blur sigma
            jitter_pixels: Random jitter for regularization
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (optimized_tensor, scores_history)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize random image
        image = torch.randn(1, *self.input_size, device=self.device, requires_grad=True)
        
        # ImageNet normalization params
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
        scores_history = []
        
        print(f"\n🔥 Generating image for class {target_class}...")
        
        for iteration in range(num_iterations):
            # Random jitter for regularization
            if jitter_pixels > 0:
                ox, oy = np.random.randint(-jitter_pixels, jitter_pixels, 2)
                image_jittered = torch.roll(image, shifts=(ox, oy), dims=(2, 3))
            else:
                image_jittered = image
            
            # Normalize image
            image_normalized = (image_jittered - mean) / std
            
            # Forward pass
            output = self.model(image_normalized)
            
            # Get target class score
            class_score = output[0, target_class]
            
            # L2 regularization (prefer smaller pixel values)
            l2_penalty = l2_reg * torch.norm(image)
            
            # Total loss to maximize (negative for gradient ascent)
            loss = -(class_score - l2_penalty)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Gradient ascent step
            with torch.no_grad():
                image -= learning_rate * image.grad
                
                # Clip to reasonable range
                image.clamp_(-3, 3)
                
                # Apply blur periodically
                if (iteration + 1) % blur_frequency == 0:
                    image_np = image[0].cpu().numpy().transpose(1, 2, 0)
                    from scipy.ndimage import gaussian_filter
                    image_np = gaussian_filter(image_np, sigma=blur_sigma)
                    image.data = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
            
            # Clear gradients
            image.grad.zero_()
            
            # Track progress
            scores_history.append(class_score.item())
            
            if (iteration + 1) % 100 == 0:
                print(f"   Iteration {iteration+1}/{num_iterations}: Score = {class_score.item():.3f}")
        
        print(f"✅ Final score: {scores_history[-1]:.3f}")
        
        return image.detach(), np.array(scores_history)
    
    def generate_multiple(
        self,
        target_class: int,
        num_samples: int = 4,
        **kwargs
    ) -> Tuple[list, list]:
        """
        Generate multiple diverse images for target class
        
        Args:
            target_class: Class to maximize
            num_samples: Number of images to generate
            **kwargs: Arguments for generate()
        
        Returns:
            Tuple of (images_list, scores_list)
        """
        images = []
        scores_histories = []
        
        for i in range(num_samples):
            print(f"\n📸 Sample {i+1}/{num_samples}")
            image, scores = self.generate(target_class, seed=i*42, **kwargs)
            images.append(image)
            scores_histories.append(scores)
        
        return images, scores_histories


def denormalize_for_visualization(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize and convert tensor to numpy for visualization
    
    Args:
        image_tensor: Tensor (1, C, H, W) or (C, H, W)
    
    Returns:
        numpy array (H, W, C) in range [0, 255]
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    
    # Move to CPU
    img = image_tensor.cpu().detach().numpy().transpose(1, 2, 0)
    
    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Scale to [0, 255]
    img = (img * 255).astype(np.uint8)
    
    return img


def visualize_activation_maximization(
    images: list,
    target_class: int,
    class_names: list,
    save_path: Optional[str] = None
):
    """
    Visualize multiple activation maximization results
    
    Args:
        images: List of image tensors
        target_class: Target class index
        class_names: List of class names
        save_path: Path to save figure
    """
    num_images = len(images)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))
    
    if num_images == 1:
        axes = [axes]
    
    for i, image_tensor in enumerate(images):
        img_vis = denormalize_for_visualization(image_tensor)
        
        axes[i].imshow(img_vis)
        axes[i].set_title(f'Sample {i+1}', fontsize=14, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(
        f'Activation Maximization: {class_names[target_class]} Class',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()


def plot_optimization_curves(
    scores_histories: list,
    target_class: int,
    class_names: list,
    save_path: Optional[str] = None
):
    """
    Plot optimization curves for multiple runs
    
    Args:
        scores_histories: List of score arrays
        target_class: Target class index
        class_names: List of class names
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    for i, scores in enumerate(scores_histories):
        plt.plot(scores, label=f'Sample {i+1}', linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('Class Score', fontsize=12, fontweight='bold')
    plt.title(
        f'Optimization Progress: {class_names[target_class]} Class',
        fontsize=14,
        fontweight='bold'
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {save_path}")
    
    plt.show()
