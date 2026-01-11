"""
CuPy Image Classifier Training Script - GPU Accelerated

Trains DetectorBase in classification mode on electronic components.
"""

import os
import sys
import time
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
from cupy_detector.model import get_model, MODEL_CONFIGS
from cupy_detector.utils import letterbox_image

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'images')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints_classifier')

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
NUM_CLASSES = len(SELECTED_CLASSES)
TARGET_SIZE = 224

# =============================================================================
# LOSS FUNCTION
# =============================================================================

class ClassificationLoss:
    """
    Categorical Cross Entropy Loss for classification.
    Assumes model output is already Softmax probabilities.
    Implicitly handles the Logits backward since we pass (p - y) to model.backward().
    """
    def __init__(self):
        self.pred = None
        self.target = None
        
    def forward(self, pred, target):
        """
        NLL Loss: -sum(y * log(p))
        """
        self.pred = pred
        self.target = target
        N = pred.shape[0]
        eps = 1e-12
        loss = -cp.sum(target * cp.log(pred + eps)) / N
        return loss
        
    def backward(self):
        """
        Gradient of CrossEntropy with Softmax w.r.t. Logits is (p - y).
        Since our model's last layers are Linear, we pass p - y as the gradient 
        of the output layer back to the linear layer.
        """
        N = self.pred.shape[0]
        # (pred - target) is the gradient w.r.t. logits
        return (self.pred - self.target) / N

# =============================================================================
# DATASET LOADING
# =============================================================================

class ElectronicComponentDataset:
    def __init__(self, data_root, class_mapping, target_size=224, split='train', val_split=0.2):
        self.data_root = data_root
        self.class_mapping = class_mapping
        self.target_size = target_size
        self.samples = []
        
        # Collect all images from selected class folders
        for class_name, class_id in class_mapping.items():
            class_dir = os.path.join(data_root, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Directory not found: {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name), class_id))
        
        # Deterministic shuffle
        np.random.seed(42)
        np.random.shuffle(self.samples)
        
        num_val = int(len(self.samples) * val_split)
        if split == 'train':
            self.samples = self.samples[num_val:]
        else:
            self.samples = self.samples[:num_val]
            
        print(f"[{split}] Loaded {len(self.samples)} samples across {len(class_mapping)} classes")

    def __len__(self):
        return len(self.samples)

    def get_batch(self, indices):
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            img_path, label = self.samples[idx]
            try:
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img, dtype=np.float32)
                
                # Letterbox resize
                img_proc, _ = letterbox_image(img_np, target_size=self.target_size)
                
                # Normalize and CHW
                img_proc = img_proc / 255.0
                img_proc = np.transpose(img_proc, (2, 0, 1))
                
                # One-hot label
                one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
                one_hot[label] = 1.0
                
                batch_images.append(img_proc)
                batch_labels.append(one_hot)
            except Exception as e:
                print(f"[ERROR] Failed to load {img_path}: {e}")
                continue
        
        if not batch_images:
            return None, None
            
        return np.stack(batch_images).astype(np.float32), np.stack(batch_labels).astype(np.float32)

# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    print("=" * 60)
    print("CuPy Electronic Component Classifier Training (GPU)")
    print("=" * 60)
    
    # Load data
    train_dataset = ElectronicComponentDataset(DATA_DIR, CLASS_MAPPING, TARGET_SIZE, split='train')
    val_dataset = ElectronicComponentDataset(DATA_DIR, CLASS_MAPPING, TARGET_SIZE, split='validation')
    
    # Create model
    model = get_model(config['model'], num_classes=NUM_CLASSES, classifier=True)
    print(f"Model: {config['model']}, Parameters: {model.get_params_count():,}")
    
    loss_fn = ClassificationLoss()
    optimizer = Adam(lr=config['lr'])
    
    # Logging
    os.makedirs(config['save_dir'], exist_ok=True)
    log_dir = os.path.join(config['save_dir'], 'runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    csv_path = os.path.join(config['save_dir'], 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr'])
    
    def get_lr(epoch):
        return config['min_lr'] + 0.5 * (config['lr'] - config['min_lr']) * (1 + np.cos(np.pi * epoch / config['epochs']))

    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        current_lr = get_lr(epoch)
        optimizer.set_lr(current_lr)
        
        # Train
        model.set_training(True)
        indices = np.random.permutation(len(train_dataset))
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(range(0, len(train_dataset), config['batch_size']), desc=f"Epoch {epoch+1}")
        for i in pbar:
            batch_indices = indices[i:i + config['batch_size']]
            imgs_np, labels_np = train_dataset.get_batch(batch_indices)
            if imgs_np is None: continue
            
            imgs = cp.asarray(imgs_np)
            labels = cp.asarray(labels_np)
            
            # Forward
            preds = model.forward(imgs)
            loss = loss_fn.forward(preds, labels)
            
            # Backward
            grad = loss_fn.backward()
            model.backward(grad)
            optimizer.step(model.all_layers)
            
            bloss = float(loss)
            train_loss += bloss
            train_batches += 1
            pbar.set_postfix({'loss': f'{bloss:.4f}', 'lr': f'{current_lr:.2e}'})
            
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.set_training(False)
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        for i in range(0, len(val_dataset), config['batch_size']):
            batch_indices = range(i, min(i + config['batch_size'], len(val_dataset)))
            imgs_np, labels_np = val_dataset.get_batch(batch_indices)
            if imgs_np is None: continue
            
            imgs = cp.asarray(imgs_np)
            labels = cp.asarray(labels_np)
            
            preds = model.forward(imgs)
            loss = loss_fn.forward(preds, labels)
            
            val_loss += float(loss)
            val_batches += 1
            
            pred_classes = cp.argmax(preds, axis=1)
            gt_classes = cp.argmax(labels, axis=1)
            val_correct += int(cp.sum(pred_classes == gt_classes))
            val_total += len(gt_classes)
            
        avg_val_loss = val_loss / val_batches
        val_acc = val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} done. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {epoch_time:.1f}s")
        
        # Log
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        csv_writer.writerow([epoch+1, avg_train_loss, avg_val_loss, val_acc, current_lr])
        csv_file.flush()
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, config['save_dir'], 'best_model.npz')
            
    writer.close()
    csv_file.close()

def save_model(model, save_dir, filename):
    save_path = os.path.join(save_dir, filename)
    weights = {}
    for i, layer in enumerate(model.all_layers):
        if hasattr(layer, 'W'):
            weights[f'layer_{i}_W'] = cp.asnumpy(layer.W)
            if layer.b is not None: weights[f'layer_{i}_b'] = cp.asnumpy(layer.b)
        if hasattr(layer, 'gamma'):
            weights[f'layer_{i}_gamma'] = cp.asnumpy(layer.gamma)
            weights[f'layer_{i}_beta'] = cp.asnumpy(layer.beta)
            weights[f'layer_{i}_running_mean'] = cp.asnumpy(layer.running_mean)
            weights[f'layer_{i}_running_var'] = cp.asnumpy(layer.running_var)
        if hasattr(layer, 'fc1_W'):
            weights[f'layer_{i}_fc1_W'] = cp.asnumpy(layer.fc1_W)
            weights[f'layer_{i}_fc2_W'] = cp.asnumpy(layer.fc2_W)
            
    # Save classifier head if exists
    if model.classifier:
        weights['cls_fc_W'] = cp.asnumpy(model.cls_fc_W)
        weights['cls_fc_b'] = cp.asnumpy(model.cls_fc_b)
        
    np.savez(save_path, **weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tiny', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'min_lr': 1e-6,
        'save_dir': SAVE_DIR
    }
    
    train(config)
