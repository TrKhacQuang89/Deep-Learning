"""
PyTorch Image Classifier Training Script

Trains DetectorBase in classification mode on electronic components.
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_model import get_model
from detector.utils import letterbox_image

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), 'images')
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints_classifier_torch')

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
# DATASET
# =============================================================================

class ElectronicComponentDataset(Dataset):
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
        random_indices = np.random.permutation(len(self.samples))
        self.samples = [self.samples[i] for i in random_indices]
        
        num_val = int(len(self.samples) * val_split)
        if split == 'train':
            self.samples = self.samples[num_val:]
        else:
            self.samples = self.samples[:num_val]
            
        print(f"[{split}] Loaded {len(self.samples)} samples across {NUM_CLASSES} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img, dtype=np.uint8) # Keep as uint8 for memory efficiency during loading
            
            # Letterbox resize
            img_proc, _ = letterbox_image(img_np.astype(np.float32), target_size=self.target_size)
            
            # Normalize and CHW
            img_proc = img_proc / 255.0
            img_proc = np.transpose(img_proc, (2, 0, 1))
            
            return torch.from_numpy(img_proc.astype(np.float32)), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"[ERROR] Failed to load {img_path}: {e}")
            return torch.zeros((3, self.target_size, self.target_size)), torch.tensor(0, dtype=torch.long)


class PreprocessedDataset(Dataset):
    """Dataset that loads preprocessed tensors for faster training."""
    def __init__(self, pt_path):
        print(f"Loading preprocessed data from {pt_path}...")
        data = torch.load(pt_path, weights_only=True)
        self.images = data['images'] # (N, 3, 224, 224) half-precision
        self.labels = data['labels'] # (N,) long
        self.num_samples = data['num_samples']
        print(f"Loaded {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Convert half to float on the fly (saves memory in RAM)
        return self.images[idx].float(), self.labels[idx]

# =============================================================================
# TRAINING
# =============================================================================

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    print("PyTorch Electronic Component Classifier Training")
    print("=" * 60)
    
    # Load datasets
    if config['preprocessed']:
        train_pt = os.path.join(config['preprocessed_dir'], 'train.pt')
        val_pt = os.path.join(config['preprocessed_dir'], 'val.pt')
        
        if not os.path.exists(train_pt):
            print(f"[ERROR] Preprocessed data not found at {train_pt}. Run preprocess_classifier.py first.")
            sys.exit(1)
            
        train_dataset = PreprocessedDataset(train_pt)
        val_dataset = PreprocessedDataset(val_pt)
        num_workers = 0 # No need for multi-worker when data is already in RAM
    else:
        train_dataset = ElectronicComponentDataset(DATA_DIR, CLASS_MAPPING, TARGET_SIZE, split='train')
        val_dataset = ElectronicComponentDataset(DATA_DIR, CLASS_MAPPING, TARGET_SIZE, split='validation')
        num_workers = config['num_workers']
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=num_workers, pin_memory=True)
    
    # Create model
    model = get_model(config['model'], num_classes=NUM_CLASSES, classifier=True).to(device)
    print(f"Model: {config['model']}, Parameters: {model.get_params_count():,}")
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    
    # Logging
    os.makedirs(config['save_dir'], exist_ok=True)
    log_dir = os.path.join(config['save_dir'], 'runs', datetime.now().strftime('%Y%m%d_%H%M%S'))
    writer = SummaryWriter(log_dir)
    
    csv_path = os.path.join(config['save_dir'], 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'lr'])
    
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs)  # Raw logits
            
            # CrossEntropyLoss expects logits and applies log_softmax internally
            loss = nn.functional.cross_entropy(preds, labels)
            
            loss.backward()
            optimizer.step()
            
            bloss = loss.item()
            train_loss += bloss
            train_batches += 1
            pbar.set_postfix({'loss': f'{bloss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
            
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                loss = nn.functional.cross_entropy(preds, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                pred_classes = preds.argmax(dim=1)
                val_correct += (pred_classes == labels).sum().item()
                val_total += labels.size(0)
                
        avg_val_loss = val_loss / val_batches
        val_acc = val_correct / val_total
        
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
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
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pth'))
            
    writer.close()
    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='tiny', choices=['tiny', 'small', 'medium', 'large', 'xlarge', 'huge'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--preprocessed', action='store_true', help='Use preprocessed data for fast training')
    parser.add_argument('--preprocessed_dir', type=str, default='data/preprocessed_classifier')
    args = parser.parse_args()
    
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'min_lr': 1e-6,
        'save_dir': SAVE_DIR,
        'num_workers': args.num_workers,
        'preprocessed': args.preprocessed,
        'preprocessed_dir': args.preprocessed_dir
    }
    
    train(config)
