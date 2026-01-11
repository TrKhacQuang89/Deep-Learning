"""
GPU-Accelerated Preprocessing for Electronic Component Classification

Uses PyTorch with CUDA for fast batch image processing.
"""

import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms.functional as TF

# Config
DATA_ROOT = os.path.join(os.path.dirname(__file__), 'images')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data/preprocessed_classifier')
TARGET_SIZE = 224
PAD_VALUE = 128 / 255.0  # Normalized
BATCH_SIZE = 128  # Process images in batches on GPU

SELECTED_CLASSES = [
    'LED', 
    'Integrated-micro-circuit', 
    'potentiometer', 
    'relay', 
    'heat-sink', 
    'Electrolytic-capacitor', 
    'filament', 
    'junction-transistor', 
    'cartridge-fuse', 
    'Bypass-capacitor'
]

CLASS_MAPPING = {name: i for i, name in enumerate(SELECTED_CLASSES)}

def letterbox_batch_gpu(images, target_size=224, pad_value=0.5):
    """
    Letterbox resize batch of images on GPU using PyTorch.
    
    Args:
        images: List of PIL Images
        target_size: Target square size
        pad_value: Padding value (normalized 0-1)
    
    Returns:
        Tensor of shape (N, 3, target_size, target_size)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processed = []
    
    for img in images:
        # Convert to tensor and normalize
        img_tensor = TF.to_tensor(img)  # (3, H, W), range [0, 1]
        
        # Get dimensions
        c, h, w = img_tensor.shape
        
        # Calculate padding to make square
        max_side = max(h, w)
        pad_h = max_side - h
        pad_w = max_side - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad to square
        img_padded = TF.pad(img_tensor, [pad_left, pad_top, pad_right, pad_bottom], fill=pad_value)
        
        # Resize to target size using bilinear interpolation
        img_resized = TF.resize(img_padded, [target_size, target_size], antialias=True)
        
        processed.append(img_resized)
    
    # Stack into batch tensor and move to GPU
    batch = torch.stack(processed).to(device)
    return batch

def main():
    print("="*60)
    print("GPU-Accelerated Classifier Preprocessor")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Input:  {DATA_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Collect all samples
    all_samples = []
    for class_name, class_id in CLASS_MAPPING.items():
        class_dir = os.path.join(DATA_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_samples.append((os.path.join(class_dir, img_name), class_id))
    
    print(f"Found {len(all_samples)} samples across {len(SELECTED_CLASSES)} classes")
    
    # Shuffle samples
    np.random.seed(42)
    np.random.shuffle(all_samples)
    
    # Split into train/val
    val_split = 0.2
    num_val = int(len(all_samples) * val_split)
    train_samples = all_samples[num_val:]
    val_samples = all_samples[:num_val]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for split_name, samples in [('train', train_samples), ('val', val_samples)]:
        print(f"\nProcessing {split_name} split ({len(samples)} samples)...")
        
        all_images = []
        all_labels = []
        
        # Process in batches
        for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=f"Batches ({split_name})"):
            batch_samples = samples[i:i + BATCH_SIZE]
            
            # Load images
            batch_imgs = []
            batch_lbls = []
            for img_path, label in batch_samples:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_imgs.append(img)
                    batch_lbls.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            if not batch_imgs:
                continue
            
            # Process batch on GPU
            processed_batch = letterbox_batch_gpu(batch_imgs, TARGET_SIZE, PAD_VALUE)
            
            # Convert to half precision and move to CPU
            processed_batch = processed_batch.half().cpu()
            
            # Store results
            for j, tensor in enumerate(processed_batch):
                all_images.append(tensor)
                all_labels.append(torch.tensor(batch_lbls[j], dtype=torch.long))
        
        print(f"Stacking {len(all_images)} samples...")
        images_tensor = torch.stack(all_images)
        labels_tensor = torch.stack(all_labels)
        
        output_file = os.path.join(OUTPUT_DIR, f'{split_name}.pt')
        torch.save({
            'images': images_tensor,
            'labels': labels_tensor,
            'num_samples': len(all_images)
        }, output_file)
        
        print(f"Saved to: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
