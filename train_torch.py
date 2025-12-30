"""
PyTorch Training Script for Object Detector.

Uses TensorBoard for logging and tqdm for progress bars.
Comparable to train_detector.py (numpy version) for performance comparison.
"""

import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_model import get_model
from detector.utils import decode_predictions, nms, compute_iou


# =============================================================================
# LOSS FUNCTION (PyTorch version)
# =============================================================================

class DetectionLoss(nn.Module):
    """
    Detection loss for grid-based object detector.
    Combines localization loss, confidence loss, and classification loss.
    """
    
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5, num_classes=3):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.num_classes = num_classes
        self.mse = nn.MSELoss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Compute detection loss.
        
        Args:
            predictions: (N, 8, 7, 7) - [x, y, w, h, conf, c1, c2, c3]
            targets: (N, 8, 7, 7) - same format
            
        Returns:
            total_loss: Scalar tensor
        """
        # Convert numpy targets to tensor if needed
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).float().to(predictions.device)
        
        # Object mask: where there are objects (confidence > 0.5)
        obj_mask = targets[:, 4:5, :, :] > 0.5  # (N, 1, 7, 7)
        noobj_mask = ~obj_mask
        
        # Localization loss (only for cells with objects)
        xy_loss = self.mse(predictions[:, 0:2, :, :], targets[:, 0:2, :, :])
        xy_loss = (xy_loss * obj_mask).sum() / (obj_mask.sum() + 1e-6)
        
        wh_loss = self.mse(predictions[:, 2:4, :, :], targets[:, 2:4, :, :])
        wh_loss = (wh_loss * obj_mask).sum() / (obj_mask.sum() + 1e-6)
        
        coord_loss = self.lambda_coord * (xy_loss + wh_loss)
        
        # Confidence loss
        obj_conf_loss = self.mse(predictions[:, 4:5, :, :], targets[:, 4:5, :, :])
        obj_conf_loss = (obj_conf_loss * obj_mask).sum() / (obj_mask.sum() + 1e-6)
        
        noobj_conf_loss = self.mse(predictions[:, 4:5, :, :], targets[:, 4:5, :, :])
        noobj_conf_loss = (noobj_conf_loss * noobj_mask).sum() / (noobj_mask.sum() + 1e-6)
        
        conf_loss = obj_conf_loss + self.lambda_noobj * noobj_conf_loss
        
        # Classification loss (only for cells with objects)
        # targets[:, 5:, :, :] are one-hot encoded
        target_classes = targets[:, 5:, :, :].argmax(dim=1)  # (N, 7, 7)
        
        # Reshape for cross entropy
        pred_cls = predictions[:, 5:, :, :].permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        target_cls = target_classes.reshape(-1)
        obj_mask_flat = obj_mask.squeeze(1).reshape(-1)
        
        cls_loss_all = self.ce(pred_cls, target_cls)
        cls_loss = (cls_loss_all * obj_mask_flat).sum() / (obj_mask_flat.sum() + 1e-6)
        
        total_loss = coord_loss + conf_loss + cls_loss
        
        return total_loss


# =============================================================================
# PYTORCH DATASET (proper implementation with DataLoader support)
# =============================================================================

import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from detector.utils import letterbox_image, transform_bboxes_letterbox, create_target

# Class mapping
CLASS_MAPPING = {
    'Cap1': 0,
    'Resistor': 1,
    'Transformer': 2,
}


class PCBTorchDataset(Dataset):
    """
    PyTorch Dataset for PCB Component Detection.
    Works with DataLoader for multi-worker prefetching.
    """
    
    def __init__(self, data_dir, split='train', target_size=224, num_classes=3, 
                 grid_size=7, pad_value=128):
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.pad_value = pad_value
        self.class_mapping = CLASS_MAPPING
        
        self.img_dir = os.path.join(data_dir, split, 'img')
        self.ann_dir = os.path.join(data_dir, split, 'ann')
        
        self.samples = self._load_samples()
        print(f"[PCBTorchDataset] Loaded {len(self.samples)} samples from {split}")
    
    def _load_samples(self):
        """Load list of valid samples."""
        samples = []
        
        if not os.path.exists(self.ann_dir):
            return samples
        
        ann_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.json')]
        
        for ann_file in ann_files:
            img_name = ann_file.replace('.json', '')
            img_path = os.path.join(self.img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            ann_path = os.path.join(self.ann_dir, ann_file)
            bboxes = self._parse_annotation(ann_path)
            
            if len(bboxes) > 0:
                samples.append({
                    'img_path': img_path,
                    'ann_path': ann_path,
                    'bboxes': bboxes
                })
        
        return samples
    
    def _parse_annotation(self, ann_path):
        """Parse Supervisely JSON annotation."""
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
            
            if class_name not in self.class_mapping:
                continue
            
            class_id = self.class_mapping[class_name]
            
            exterior = obj.get('points', {}).get('exterior', [])
            if len(exterior) != 2:
                continue
            
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]
            
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h
            
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            width = np.clip(width, 0, 1)
            height = np.clip(height, 0, 1)
            
            bboxes.append([class_id, x_center, y_center, width, height])
        
        return bboxes
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a sample as PyTorch tensors.
        
        Returns:
            image: (3, H, W) float tensor, normalized [0, 1]
            target: (8, grid_size, grid_size) float tensor
        """
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['img_path']).convert('RGB')
        img = np.array(img, dtype=np.float32)
        
        # Letterbox preprocessing
        img_letterbox, info = letterbox_image(
            img, 
            target_size=self.target_size,
            pad_value=self.pad_value
        )
        
        # Transform bboxes
        bboxes_letterbox = transform_bboxes_letterbox(sample['bboxes'], info)
        
        # Create target grid
        target = create_target(
            bboxes_letterbox,
            num_classes=self.num_classes,
            grid_size=self.grid_size
        )
        
        # Normalize and convert to tensor format
        img_letterbox = img_letterbox / 255.0
        img_letterbox = np.transpose(img_letterbox, (2, 0, 1))  # (3, H, W)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(img_letterbox.astype(np.float32))
        target_tensor = torch.from_numpy(target.astype(np.float32))
        
        return image_tensor, target_tensor


def create_dataloader(data_dir, split, batch_size, target_size=224, num_classes=3,
                      grid_size=7, num_workers=4, shuffle=True):
    """
    Create a DataLoader for the dataset.
    
    Args:
        data_dir: Path to dataset
        split: 'train' or 'validation'
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader instance
    """
    dataset = PCBTorchDataset(
        data_dir=data_dir,
        split=split,
        target_size=target_size,
        num_classes=num_classes,
        grid_size=grid_size
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Faster GPU transfer
        drop_last=True if split == 'train' else False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return loader


# =============================================================================
# FAST DATASET (loads preprocessed .pt files)
# =============================================================================

class PreprocessedDataset(Dataset):
    """
    Fast dataset that loads preprocessed .pt files.
    Run preprocess_dataset.py first to create the data.
    """
    
    def __init__(self, data_path):
        """
        Args:
            data_path: Path to data.pt file
        """
        print(f"[PreprocessedDataset] Loading {data_path}...")
        data = torch.load(data_path, weights_only=True)
        
        self.images = data['images'].float()  # Convert from half to float
        self.targets = data['targets'].float()
        self.num_samples = data['num_samples']
        
        print(f"[PreprocessedDataset] Loaded {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def create_fast_dataloader(data_dir, split, batch_size, shuffle=True):
    """
    Create a DataLoader using preprocessed data.
    
    Args:
        data_dir: Path to preprocessed data directory (e.g., 'data/preprocessed')
        split: 'train' or 'val'
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader instance
    """
    data_path = os.path.join(data_dir, split, 'data.pt')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}. "
            f"Run 'python preprocess_dataset.py' first."
        )
    
    dataset = PreprocessedDataset(data_path)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Data is already in memory, no need for workers
        pin_memory=True,
        drop_last=True if split == 'train' else False,
    )
    
    return loader


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, dataloader, loss_fn, device, conf_thresh=0.3, iou_thresh=0.5):
    """Evaluate model on dataset using DataLoader."""
    model.eval()
    
    total_loss = 0
    total_samples = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size_actual = images.shape[0]
            
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual
            
            # Convert to numpy for decode_predictions
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for b in range(batch_size_actual):
                pred_boxes = decode_predictions(outputs_np[b], conf_thresh=conf_thresh)
                pred_boxes = nms(pred_boxes, iou_thresh=iou_thresh)
                
                # Extract ground truth boxes
                gt_boxes = []
                target_b = targets_np[b]
                for i in range(target_b.shape[1]):
                    for j in range(target_b.shape[2]):
                        if target_b[4, i, j] > 0.5:
                            x = (j + target_b[0, i, j]) / 7
                            y = (i + target_b[1, i, j]) / 7
                            w = target_b[2, i, j]
                            h = target_b[3, i, j]
                            cls = np.argmax(target_b[5:, i, j])
                            gt_boxes.append([x - w/2, y - h/2, x + w/2, y + h/2, 1.0, cls])
                
                # Match predictions with ground truth
                matched_gt = [False] * len(gt_boxes)
                
                for pred in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt in enumerate(gt_boxes):
                        if matched_gt[gt_idx]:
                            continue
                        if pred[5] != gt[5]:
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
    
    avg_loss = total_loss / max(total_samples, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    model.train()
    
    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    """Main training function."""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup directories
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Setup TensorBoard
    log_dir = os.path.join(config['save_dir'], 'runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs: {log_dir}")
    
    # Load datasets
    if config.get('use_fast_loader', False):
        # Use preprocessed data for fast loading
        print("[FAST MODE] Using preprocessed data")
        train_loader = create_fast_dataloader(
            data_dir=config['preprocessed_dir'],
            split='train',
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        val_loader = create_fast_dataloader(
            data_dir=config['preprocessed_dir'],
            split='val',
            batch_size=config['batch_size'],
            shuffle=False
        )
    else:
        # Load from raw images (slower)
        train_loader = create_dataloader(
            data_dir=config['data_dir'],
            split='train',
            batch_size=config['batch_size'],
            target_size=config['target_size'],
            num_classes=config['num_classes'],
            grid_size=config['grid_size'],
            num_workers=config['num_workers'],
            shuffle=True
        )
        
        val_loader = create_dataloader(
            data_dir=config['data_dir'],
            split='validation',
            batch_size=config['batch_size'],
            target_size=config['target_size'],
            num_classes=config['num_classes'],
            grid_size=config['grid_size'],
            num_workers=config['num_workers'],
            shuffle=False
        )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model
    model = get_model(config['model_size'], num_classes=config['num_classes']).to(device)
    print(f"Model: {config['model_size']} - Parameters: {model.get_params_count():,}")
    
    # Log model graph
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    writer.add_graph(model, dummy_input)
    
    # Loss and optimizer
    loss_fn = DetectionLoss(
        lambda_coord=config['lambda_coord'],
        lambda_noobj=config['lambda_noobj'],
        num_classes=config['num_classes']
    )
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'],
        eta_min=config['min_lr']
    )
    
    # Training loop
    best_val_loss = float('inf')
    global_step = 0
    
    epoch_pbar = tqdm(range(config['epochs']), desc='Training', unit='epoch')
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        train_batches = 0
        
        batch_pbar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}',
            leave=False,
            unit='batch'
        )
        
        for images, targets in batch_pbar:
            # Move to device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            global_step += 1
            
            # Update batch progress bar
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to TensorBoard every N batches
            if global_step % config['log_interval'] == 0:
                writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation
        val_metrics = evaluate(
            model, val_loader, loss_fn, device,
            conf_thresh=config['conf_thresh']
        )
        
        # Update LR
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Log to TensorBoard
        writer.add_scalar('Train/Loss', avg_train_loss, epoch)
        writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
        writer.add_scalar('Val/Precision', val_metrics['precision'], epoch)
        writer.add_scalar('Val/Recall', val_metrics['recall'], epoch)
        writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{val_metrics["loss"]:.4f}',
            'val_f1': f'{val_metrics["f1"]:.3f}'
        })
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_metrics': val_metrics
            }, os.path.join(config['save_dir'], 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss']
            }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'epoch': config['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'val_loss': val_metrics['loss']
    }, os.path.join(config['save_dir'], 'final_model.pth'))
    
    writer.close()
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {config['save_dir']}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train PyTorch Object Detector')
    parser.add_argument('--model', type=str, default='tiny', 
                        choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'huge', 'giant', 'giant_original'],
                        help='Model size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='checkpoints_torch', help='Save directory')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Use preprocessed data for fast loading (run preprocess_dataset.py first)')
    args = parser.parse_args()
    
    config = {
        # Data
        'data_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                  'data/pcb-component-detection-DatasetNinja'),
        'preprocessed_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                          'data/preprocessed'),
        'target_size': 224,
        'num_classes': 3,
        'grid_size': 7,
        
        # Model
        'model_size': args.model,
        
        # Training
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'min_lr': 1e-6,
        
        # Loss weights
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        
        # DataLoader
        'num_workers': args.num_workers,
        'use_fast_loader': args.fast,
        
        # Detection
        'conf_thresh': 0.3,
        
        # Logging & Saving
        'save_dir': args.save_dir,
        'log_interval': 20,
        'save_interval': 10,
    }
    
    print("\n" + "=" * 60)
    print("PyTorch Object Detector Training")
    print("=" * 60)
    print(f"Model: {config['model_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print("=" * 60 + "\n")
    
    train(config)


if __name__ == "__main__":
    main()
