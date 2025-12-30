"""
Training script cho Simple Object Detector.
Huấn luyện model trên PCB Component Detection dataset.

Features:
- Train + Validation datasets
- Learning rate scheduler (cosine/step)
- CSV logging
- Model save/load
- Progress tracking
"""

import os
import sys
import csv
import time
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import SimpleDetector, DetectionLoss, PCBDataset
from detector.utils import decode_predictions, nms, compute_iou


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

class LRScheduler:
    """Simple learning rate scheduler."""
    
    def __init__(self, initial_lr, scheduler_type='cosine', total_epochs=100, 
                 min_lr=1e-6):
        """
        Args:
            initial_lr: Learning rate ban đầu
            scheduler_type: 'cosine', 'step', hoặc 'constant'
            total_epochs: Tổng số epochs
            min_lr: LR tối thiểu
        """
        self.initial_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.total_epochs = total_epochs
        self.min_lr = min_lr
    
    def get_lr(self, epoch):
        """Lấy learning rate cho epoch hiện tại."""
        
        # Progress calculation
        progress = epoch / max(1, self.total_epochs)
        
        if self.scheduler_type == 'cosine':
            # Cosine annealing
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        elif self.scheduler_type == 'step':
            # Step decay (giảm 10% mỗi 5% training)
            num_steps = int(progress / 0.05)
            lr = self.initial_lr * (0.9 ** num_steps)
        
        else:  # constant
            lr = self.initial_lr
        
        return max(lr, self.min_lr)


# =============================================================================
# CSV LOGGER
# =============================================================================

class CSVLogger:
    """Logger để ghi training metrics vào file CSV."""
    
    def __init__(self, log_path):
        """
        Args:
            log_path: Đường dẫn file CSV
        """
        self.log_path = log_path
        self.headers_written = False
    
    def log(self, metrics_dict):
        """
        Ghi một dòng metrics.
        
        Args:
            metrics_dict: Dict chứa metrics, e.g. {'epoch': 1, 'loss': 0.5, ...}
        """
        file_exists = os.path.exists(self.log_path)
        
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            
            if not file_exists or not self.headers_written:
                writer.writeheader()
                self.headers_written = True
            
            writer.writerow(metrics_dict)


# =============================================================================
# MODEL SAVE/LOAD
# =============================================================================

def save_model(model, path, metadata=None):
    """
    Lưu model weights và metadata.
    
    Args:
        model: SimpleDetector instance
        path: Đường dẫn file .npz
        metadata: Dict metadata (epoch, loss, etc.)
    """
    weights = {}
    
    for i, layer in enumerate(model.all_layers):
        if hasattr(layer, 'W'):
            weights[f'layer_{i}_W'] = layer.W
        if hasattr(layer, 'b'):
            weights[f'layer_{i}_b'] = layer.b
    
    if metadata:
        weights['metadata'] = np.array([str(metadata)])
    
    np.savez(path, **weights)
    print(f"[SAVE] Model saved to: {path}")


def load_model(model, path):
    """
    Load model weights từ file.
    
    Args:
        model: SimpleDetector instance
        path: Đường dẫn file .npz
        
    Returns:
        metadata: Dict metadata nếu có
    """
    if not os.path.exists(path):
        print(f"[WARNING] Model file not found: {path}")
        return None
    
    data = np.load(path, allow_pickle=True)
    
    for i, layer in enumerate(model.all_layers):
        if hasattr(layer, 'W') and f'layer_{i}_W' in data:
            layer.W = data[f'layer_{i}_W']
        if hasattr(layer, 'b') and f'layer_{i}_b' in data:
            layer.b = data[f'layer_{i}_b']
    
    metadata = None
    if 'metadata' in data:
        try:
            metadata = eval(data['metadata'][0])
        except:
            pass
    
    print(f"[LOAD] Model loaded from: {path}")
    return metadata


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model, dataset, loss_fn, batch_size=16, conf_thresh=0.3, iou_thresh=0.5):
    """
    Đánh giá model trên dataset.
    
    Returns:
        avg_loss: Loss trung bình
        metrics: Dict các metrics (precision, recall, mAP, etc.)
    """
    total_loss = 0
    total_samples = 0
    
    # Tracking for precision/recall
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives (missed detections)
    
    all_detections = []  # For mAP calculation
    all_ground_truths = []
    
    for images, targets in dataset.iterate_batches(batch_size, shuffle=False):
        batch_size_actual = images.shape[0]
        
        # Forward
        outputs = model.forward(images)
        
        # Loss
        loss = loss_fn.forward(outputs, targets)
        total_loss += loss * batch_size_actual
        total_samples += batch_size_actual
        
        # Decode predictions và so sánh với ground truth
        for b in range(batch_size_actual):
            pred_boxes = decode_predictions(outputs[b], conf_thresh=conf_thresh)
            pred_boxes = nms(pred_boxes, iou_thresh=iou_thresh)
            
            # Extract ground truth boxes từ target
            gt_boxes = []
            target_b = targets[b]
            for i in range(target_b.shape[1]):
                for j in range(target_b.shape[2]):
                    if target_b[4, i, j] > 0.5:  # Has object
                        x = (j + target_b[0, i, j]) / 7
                        y = (i + target_b[1, i, j]) / 7
                        w = target_b[2, i, j]
                        h = target_b[3, i, j]
                        cls = np.argmax(target_b[5:, i, j])
                        gt_boxes.append([x - w/2, y - h/2, x + w/2, y + h/2, 1.0, cls])
            
            # Match predictions với ground truth
            matched_gt = [False] * len(gt_boxes)
            
            for pred in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_boxes):
                    if matched_gt[gt_idx]:
                        continue
                    if pred[5] != gt[5]:  # Different class
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
            
            # Unmatched ground truths = false negatives
            total_fn += sum(1 for m in matched_gt if not m)
    
    # Compute metrics
    avg_loss = total_loss / max(total_samples, 1)
    
    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    metrics = {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn
    }
    
    return avg_loss, metrics


# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    """
    Main training function.
    
    Args:
        config: Dict chứa hyperparameters
    """
    print("=" * 60)
    print("SIMPLE OBJECT DETECTOR - TRAINING")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup directories
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Initialize datasets
    print("[1/5] Loading datasets...")
    
    train_dataset = PCBDataset(
        data_dir=config['data_dir'],
        split='train',
        target_size=config['target_size'],
        num_classes=config['num_classes'],
        grid_size=config['grid_size']
    )
    
    val_dataset = PCBDataset(
        data_dir=config['data_dir'],
        split='validation',
        target_size=config['target_size'],
        num_classes=config['num_classes'],
        grid_size=config['grid_size']
    )
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples:   {len(val_dataset)}")
    
    # Initialize model
    print("\n[2/5] Initializing model...")
    model = SimpleDetector(num_classes=config['num_classes'])
    print(f"   Parameters: {model.get_params_count():,}")
    
    # Initialize loss
    loss_fn = DetectionLoss(
        lambda_coord=config['lambda_coord'],
        lambda_noobj=config['lambda_noobj']
    )
    
    # Initialize scheduler
    scheduler = LRScheduler(
        initial_lr=config['learning_rate'],
        scheduler_type=config['scheduler_type'],
        total_epochs=config['epochs'],
        min_lr=config['min_lr']
    )
    
    # Initialize loggers
    log_path = os.path.join(config['save_dir'], 'training_log.csv')
    batch_log_path = os.path.join(config['save_dir'], 'batch_log.csv')
    
    logger = CSVLogger(log_path)
    batch_logger = CSVLogger(batch_log_path)
    
    print(f"   Logging to: {log_path} and {batch_log_path}")
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.get('resume_from'):
        metadata = load_model(model, config['resume_from'])
        if metadata and 'epoch' in metadata:
            start_epoch = metadata['epoch'] + 1
            best_val_loss = metadata.get('best_val_loss', float('inf'))
            print(f"   Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n[3/5] Starting training...")
    print("-" * 60)
    
    steps_per_epoch = (len(train_dataset) + config['batch_size'] - 1) // config['batch_size']
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()
        
        # Get learning rate
        lr = scheduler.get_lr(epoch)
        
        # Training
        train_loss = 0
        train_batches = 0
        
        for i, (images, targets) in enumerate(train_dataset.iterate_batches(config['batch_size'], shuffle=True)):
            # Forward
            outputs = model.forward(images)
            loss = loss_fn.forward(outputs, targets)

            if (i + 1) % config['log_interval'] == 0:
                print(f"Epoch [{epoch+1}/{config['epochs']}], Batch [{i+1}/{steps_per_epoch}], Loss: {loss:.4f}")
                
                # Log batch metrics
                batch_logger.log({
                    'epoch': epoch + 1,
                    'batch': i + 1,
                    'loss': loss,
                    'learning_rate': lr
                })
            
            # Backward
            grad = loss_fn.backward()
            model.backward(grad)
            
            # Update weights with SGD
            model.update_params(lr)
            
            train_loss += loss
            train_batches += 1
            
        
        avg_train_loss = train_loss / max(train_batches, 1)
        
        # Validation
        val_loss, val_metrics = evaluate(
            model, val_dataset, loss_fn,
            batch_size=config['batch_size'],
            conf_thresh=config['conf_thresh']
        )
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        log_entry = {
            'epoch': epoch + 1,
            'learning_rate': lr,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'epoch_time': epoch_time
        }
        logger.log(log_entry)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['epochs']} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   Val P/R/F1: {val_metrics['precision']:.3f} / "
              f"{val_metrics['recall']:.3f} / {val_metrics['f1']:.3f}")
        print(f"   Time:       {epoch_time:.1f}s")
        print(f"   LR:         {lr:.6f}")
        print(f"{'='*60}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(config['save_dir'], 'best_model.npz')
            save_model(model, best_path, {
                'epoch': epoch,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'val_metrics': val_metrics
            })
        
        # Save checkpoint
        if (epoch + 1) % config['save_interval'] == 0:
            ckpt_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.npz')
            save_model(model, ckpt_path, {
                'epoch': epoch,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            })
    
    # Save final model
    print("\n[4/5] Saving final model...")
    final_path = os.path.join(config['save_dir'], 'final_model.npz')
    save_model(model, final_path, {
        'epoch': config['epochs'] - 1,
        'val_loss': val_loss,
        'best_val_loss': best_val_loss
    })
    
    # Final summary
    print("\n[5/5] Training complete!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {config['save_dir']}")
    print(f"Training log: {log_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    
    # Configuration
    config = {
        # Data
        'data_dir': 'data/pcb-component-detection-DatasetNinja',
        'target_size': 224,
        'num_classes': 3,
        'grid_size': 7,
        
        # Training
        'epochs': 50,
        'batch_size': 8,
        'learning_rate': 1e-5,
        'min_lr': 1e-6,
        'scheduler_type': 'cosine',  # 'cosine', 'step', 'constant'
        
        # Loss weights
        'lambda_coord': 5.0,
        'lambda_noobj': 0.5,
        
        # Detection
        'conf_thresh': 0.3,
        
        # Logging & Saving
        'save_dir': 'checkpoints',
        'log_interval': 20,
        'save_interval': 10,
        
        # Resume
        'resume_from': None,  # Path to checkpoint
    }
    
    # Make data_dir absolute
    config['data_dir'] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        config['data_dir']
    )
    
    # Print config
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Train
    train(config)


if __name__ == "__main__":
    main()
