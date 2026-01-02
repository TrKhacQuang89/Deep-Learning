"""
CuPy Object Detector Training Script - GPU Accelerated

Trains SimpleDetector using CuPy for GPU acceleration.
All forward/backward computations run on GPU.

Usage:
    python cupy_train_detector.py --epochs 50 --batch_size 8
"""
import time

import os
import sys
import time
import json
import csv
import argparse
import cupy as cp
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cupy_core import Adam
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
# METRICS (Proper IoU-based evaluation matching PyTorch)
# =============================================================================

from detector.utils import decode_predictions, nms, compute_iou


def evaluate_batch(outputs_np, targets_np, conf_thresh=0.3, iou_thresh=0.5):
    """
    Evaluate a batch using proper object detection metrics.
    Matches PyTorch evaluate() function exactly.
    
    Args:
        outputs_np: (N, 8, 7, 7) numpy array - predictions
        targets_np: (N, 8, 7, 7) numpy array - ground truth
        conf_thresh: Confidence threshold for predictions
        iou_thresh: IoU threshold for NMS
        
    Returns:
        tp, fp, fn counts
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    batch_size = outputs_np.shape[0]
    
    for b in range(batch_size):
        # Decode predictions and apply NMS (same as PyTorch)
        pred_boxes = decode_predictions(outputs_np[b], conf_thresh=conf_thresh)
        pred_boxes = nms(pred_boxes, iou_thresh=iou_thresh)
        
        # Extract ground truth boxes (same logic as PyTorch)
        gt_boxes = []
        target_b = targets_np[b]
        for i in range(target_b.shape[1]):  # grid height
            for j in range(target_b.shape[2]):  # grid width
                if target_b[4, i, j] > 0.5:
                    x = (j + target_b[0, i, j]) / 7
                    y = (i + target_b[1, i, j]) / 7
                    w = target_b[2, i, j]
                    h = target_b[3, i, j]
                    cls = np.argmax(target_b[5:, i, j])
                    gt_boxes.append([x - w/2, y - h/2, x + w/2, y + h/2, 1.0, cls])
        
        # Match predictions with ground truth using IoU
        matched_gt = [False] * len(gt_boxes)
        
        for pred in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_boxes):
                if matched_gt[gt_idx]:
                    continue
                if pred[5] != gt[5]:  # Class must match
                    continue
                
                iou = compute_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= 0.5 and best_gt_idx >= 0:
                total_tp += 1
                matched_gt[best_gt_idx] = True
            else:
                total_fp += 1
        
        total_fn += sum(1 for m in matched_gt if not m)
    
    return total_tp, total_fp, total_fn


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1 from counts."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    return precision, recall, f1


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    """Main training function with TensorBoard and CSV logging."""
    print("=" * 60)
    print("CuPy Object Detector Training (GPU)")
    print("=" * 60)
    
    # Check GPU
    print(f"CUDA available: {cp.cuda.runtime.getDeviceCount()}")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")

    
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
    print(f"Parameters: {model.get_params_count():,}")
    loss_fn = DetectionLoss()
    
    # Training config
    lr = config['lr']
    min_lr = config.get('min_lr', 1e-6)
    epochs = config['epochs']
    batch_size = config['batch_size']
    log_interval = config.get('log_interval', 20)
    save_interval = config.get('save_interval', 10)
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Setup TensorBoard
    log_dir = os.path.join(config['save_dir'], 'runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Setup CSV logging
    csv_path = os.path.join(config['save_dir'], 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_precision', 'val_recall', 'val_f1', 'lr', 'time'])
    print(f"CSV log: {csv_path}")
    
    # Create Adam optimizer
    optimizer = Adam(lr=lr)
    print(f"Optimizer: Adam (lr={lr})")
    
    # Cosine annealing LR schedule
    def get_lr(epoch):
        """Cosine annealing learning rate."""
        return min_lr + 0.5 * (lr - min_lr) * (1 + np.cos(np.pi * epoch / epochs))
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = get_lr(epoch)
        
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
            # Pass raw_output for numerically stable cross-entropy
            loss = loss_fn.forward(outputs, batch_targets, raw_pred=model.raw_output)
            
            # Backward
            grad = loss_fn.backward()
            model.backward(grad)
            
            # Update with Adam optimizer (use current LR from scheduler)
            optimizer.set_lr(current_lr)
            optimizer.step(model.all_layers)
            
            batch_loss = float(loss)
            train_loss += batch_loss
            train_batches += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'lr': f'{current_lr:.2e}'})
            
            # Log batch loss to TensorBoard
            if global_step % log_interval == 0:
                writer.add_scalar('Train/BatchLoss', batch_loss, global_step)
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.set_training(False)  # Disable dropout and use running stats for BN
        val_loss = 0.0
        val_batches = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        if val_images is not None:
            for i in range(0, len(val_images), batch_size):
                batch_images = cp.asarray(val_images[i:i + batch_size])
                batch_targets = cp.asarray(val_targets[i:i + batch_size])
                
                outputs = model.forward(batch_images)
                loss = loss_fn.forward(outputs, batch_targets, raw_pred=model.raw_output)
                
                # Convert to numpy for proper IoU-based evaluation
                outputs_np = cp.asnumpy(outputs)
                targets_np = cp.asnumpy(batch_targets)
                
                # Evaluate using proper object detection metrics
                tp, fp, fn = evaluate_batch(outputs_np, targets_np, conf_thresh=0.3)
                
                val_loss += float(loss)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            avg_val_precision, avg_val_recall, avg_val_f1 = compute_metrics(total_tp, total_fp, total_fn)
        else:
            avg_val_loss = 0
            avg_val_precision = 0
            avg_val_recall = 0
            avg_val_f1 = 0
        
        model.set_training(True)  # Re-enable training mode
        
        epoch_time = time.time() - epoch_start
        
        # Log to TensorBoard
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        writer.add_scalar('Val/Precision', avg_val_precision, epoch)
        writer.add_scalar('Val/Recall', avg_val_recall, epoch)
        writer.add_scalar('Val/F1', avg_val_f1, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Log to CSV
        csv_writer.writerow([
            epoch + 1, 
            f'{avg_train_loss:.6f}', 
            f'{avg_val_loss:.6f}',
            f'{avg_val_precision:.4f}',
            f'{avg_val_recall:.4f}',
            f'{avg_val_f1:.4f}',
            f'{current_lr:.2e}',
            f'{epoch_time:.1f}'
        ])
        csv_file.flush()  # Ensure data is written
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val F1: {avg_val_f1:.4f}, "
              f"LR: {current_lr:.2e}, "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, config['save_dir'], 'best_model.npz')
            print(f"  → Saved best model (val_loss: {avg_val_loss:.4f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % save_interval == 0:
            save_model(model, config['save_dir'], f'checkpoint_epoch_{epoch+1}.npz')
    
    # Save final model
    save_model(model, config['save_dir'], 'final_model.npz')
    
    # Close logging
    csv_file.close()
    writer.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {config['save_dir']}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")
    print(f"CSV log: {csv_path}")
    print("=" * 60)


def save_model(model, save_dir, filename):
    """Save model weights to .npz file."""
    save_path = os.path.join(save_dir, filename)
    weights = {}
    
    for i, layer in enumerate(model.all_layers):
        # Conv2d weights
        if hasattr(layer, 'W'):
            weights[f'layer_{i}_W'] = cp.asnumpy(layer.W)
            if layer.b is not None:
                weights[f'layer_{i}_b'] = cp.asnumpy(layer.b)
        
        # BatchNorm weights
        if hasattr(layer, 'gamma'):
            weights[f'layer_{i}_gamma'] = cp.asnumpy(layer.gamma)
            weights[f'layer_{i}_beta'] = cp.asnumpy(layer.beta)
            weights[f'layer_{i}_running_mean'] = cp.asnumpy(layer.running_mean)
            weights[f'layer_{i}_running_var'] = cp.asnumpy(layer.running_var)
        
        # SEBlock weights (no bias, matching PyTorch)
        if hasattr(layer, 'fc1_W'):
            weights[f'layer_{i}_fc1_W'] = cp.asnumpy(layer.fc1_W)
            weights[f'layer_{i}_fc2_W'] = cp.asnumpy(layer.fc2_W)
    
    np.savez(save_path, **weights)


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
        'min_lr': 1e-6,
        'save_dir': args.save_dir,
        'num_classes': NUM_CLASSES,
        'log_interval': 20,      # Log batch loss every N batches
        'save_interval': 10,     # Save checkpoint every N epochs
    }
    
    print("\n" + "=" * 60)
    print("CuPy Object Detector Training")
    print("=" * 60)
    print(f"Model: {config['model']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print("=" * 60 + "\n")
    
    train(config)


if __name__ == "__main__":
    main()
