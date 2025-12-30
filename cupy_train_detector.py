"""
CuPy Object Detector Training Script - GPU Accelerated

Trains SimpleDetector using CuPy for GPU acceleration.
All forward/backward computations run on GPU.

Usage:
    python cupy_train_detector.py --epochs 50 --batch_size 8
"""

import os
import sys
import time
import json
import argparse
import cupy as cp
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cupy_detector import DetectionLoss
from cupy_detector.model import get_model, MODEL_CONFIGS
from cupy_detector.utils import letterbox_image, transform_bboxes_letterbox, create_target


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/pcb-component-detection-DatasetNinja')
PREPROCESSED_DIR = os.path.join(os.path.dirname(__file__), 'data/preprocessed')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints_cupy')

CLASS_MAPPING = {
    'Cap1': 0,
    'Resistor': 1,
    'Transformer': 2,
}

TARGET_SIZE = 224
GRID_SIZE = 7
NUM_CLASSES = 3


# =============================================================================
# DATA LOADING
# =============================================================================

def load_preprocessed_data(split='train'):
    """Load preprocessed tensors and convert to CuPy."""
    split_name = 'val' if split == 'validation' else split
    data_file = os.path.join(PREPROCESSED_DIR, split_name, 'data.pt')
    
    if os.path.exists(data_file):
        import torch
        data = torch.load(data_file)
        images = data['images'].numpy().astype(np.float32)
        targets = data['targets'].numpy().astype(np.float32)
        print(f"[{split}] Loaded from preprocessed: {images.shape}")
        return images, targets
    else:
        print(f"[WARNING] Preprocessed data not found at {data_file}")
        return None, None


def parse_annotation(ann_path):
    """Parse annotation file."""
    bboxes = []
    
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return bboxes
    
    img_w = data.get('size', {}).get('width', 1)
    img_h = data.get('size', {}).get('height', 1)
    
    for obj in data.get('objects', []):
        class_name = obj.get('classTitle', '')
        
        if class_name not in CLASS_MAPPING:
            continue
        
        class_id = CLASS_MAPPING[class_name]
        exterior = obj.get('points', {}).get('exterior', [])
        if len(exterior) != 2:
            continue
        
        x1, y1 = exterior[0]
        x2, y2 = exterior[1]
        
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = abs(x2 - x1) / img_w
        height = abs(y2 - y1) / img_h
        
        bboxes.append([class_id, x_center, y_center, width, height])
    
    return bboxes


def load_raw_data(split='train'):
    """Load data from raw images (slower)."""
    img_dir = os.path.join(DATA_DIR, split, 'img')
    ann_dir = os.path.join(DATA_DIR, split, 'ann')
    
    if not os.path.exists(ann_dir):
        print(f"[ERROR] Annotation dir not found: {ann_dir}")
        return None, None
    
    ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]
    
    images = []
    targets = []
    
    for ann_file in tqdm(ann_files, desc=f"Loading {split}"):
        img_name = ann_file.replace('.json', '')
        img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
        
        ann_path = os.path.join(ann_dir, ann_file)
        bboxes = parse_annotation(ann_path)
        
        if len(bboxes) == 0:
            continue
        
        # Load and process image
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.float32)
        
        img_letterbox, info = letterbox_image(img, target_size=TARGET_SIZE)
        bboxes_letterbox = transform_bboxes_letterbox(bboxes, info)
        target = create_target(bboxes_letterbox, num_classes=NUM_CLASSES, grid_size=GRID_SIZE)
        
        img_letterbox = img_letterbox / 255.0
        img_letterbox = np.transpose(img_letterbox, (2, 0, 1))
        
        images.append(img_letterbox)
        targets.append(target)
    
    if len(images) == 0:
        return None, None
    
    images = np.stack(images).astype(np.float32)
    targets = np.stack(targets).astype(np.float32)
    
    print(f"[{split}] Loaded {len(images)} samples")
    return images, targets


# =============================================================================
# METRICS
# =============================================================================

def compute_accuracy(pred, target, conf_thresh=0.5):
    """Compute object detection accuracy."""
    pred_conf = pred[:, 4, :, :]
    target_mask = target[:, 4, :, :]
    
    pred_obj = (pred_conf > conf_thresh).astype(cp.float32)
    
    # TP, FP, FN
    tp = cp.sum(pred_obj * target_mask)
    fp = cp.sum(pred_obj * (1 - target_mask))
    fn = cp.sum((1 - pred_obj) * target_mask)
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    
    return float(precision), float(recall), float(f1)


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    """Main training function."""
    print("=" * 60)
    print("CuPy Object Detector Training (GPU)")
    print("=" * 60)
    
    # Check GPU
    print(f"CUDA available: {cp.cuda.runtime.getDevice()}")
    print(f"GPU: {cp.cuda.runtime.deviceGetName(0)}")
    
    # Load data
    train_images, train_targets = load_preprocessed_data('train')
    val_images, val_targets = load_preprocessed_data('validation')
    
    if train_images is None:
        print("Falling back to raw data loading...")
        train_images, train_targets = load_raw_data('train')
        val_images, val_targets = load_raw_data('validation')
    
    if train_images is None:
        print("[ERROR] No training data found!")
        return
    
    print(f"\nTraining samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images) if val_images is not None else 0}")
    
    # Create model
    model_size = config['model']
    print(f"\nModel: {model_size}")
    model = get_model(model_size, num_classes=config['num_classes'])
    loss_fn = DetectionLoss()
    
    # Training config
    lr = config['lr']
    epochs = config['epochs']
    batch_size = config['batch_size']
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Shuffle training data
        indices = np.random.permutation(len(train_images))
        
        train_loss = 0.0
        train_batches = 0
        
        # Training
        pbar = tqdm(range(0, len(train_images), batch_size), 
                    desc=f"Epoch {epoch+1}/{epochs}")
        
        for i in pbar:
            batch_idx = indices[i:i + batch_size]
            
            # Move batch to GPU
            batch_images = cp.asarray(train_images[batch_idx])
            batch_targets = cp.asarray(train_targets[batch_idx])
            
            # Forward
            outputs = model.forward(batch_images)
            loss = loss_fn.forward(outputs, batch_targets)
            
            # Backward
            grad = loss_fn.backward()
            model.backward(grad)
            
            # Update
            model.update_params(lr)
            
            train_loss += float(loss)
            train_batches += 1
            
            pbar.set_postfix({'loss': f'{float(loss):.4f}'})
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        val_loss = 0.0
        val_batches = 0
        val_precision = 0.0
        val_recall = 0.0
        val_f1 = 0.0
        
        if val_images is not None:
            for i in range(0, len(val_images), batch_size):
                batch_images = cp.asarray(val_images[i:i + batch_size])
                batch_targets = cp.asarray(val_targets[i:i + batch_size])
                
                outputs = model.forward(batch_images)
                loss = loss_fn.forward(outputs, batch_targets)
                
                p, r, f1 = compute_accuracy(outputs, batch_targets)
                
                val_loss += float(loss)
                val_precision += p
                val_recall += r
                val_f1 += f1
                val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            avg_val_precision = val_precision / val_batches if val_batches > 0 else 0
            avg_val_recall = val_recall / val_batches if val_batches > 0 else 0
            avg_val_f1 = val_f1 / val_batches if val_batches > 0 else 0
        else:
            avg_val_loss = 0
            avg_val_precision = 0
            avg_val_recall = 0
            avg_val_f1 = 0
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val F1: {avg_val_f1:.4f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Convert weights to NumPy for saving
            save_path = os.path.join(config['save_dir'], 'best_model.npz')
            weights = {}
            for i, layer in enumerate(model.all_layers):
                if hasattr(layer, 'W'):
                    weights[f'layer_{i}_W'] = cp.asnumpy(layer.W)
                    weights[f'layer_{i}_b'] = cp.asnumpy(layer.b)
            np.savez(save_path, **weights)
            print(f"  → Saved best model to {save_path}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Train CuPy Object Detector')
    parser.add_argument('--model', type=str, default='large',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Save directory')
    
    args = parser.parse_args()
    
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'save_dir': args.save_dir,
        'num_classes': NUM_CLASSES,
    }
    
    train(config)


if __name__ == "__main__":
    main()
