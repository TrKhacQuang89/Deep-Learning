"""
Dataset loader cho PCB Component Detection.
Sử dụng letterbox preprocessing để giữ nguyên aspect ratio.
"""

import os
import json
import numpy as np
from PIL import Image

from detector.utils import (
    letterbox_image, 
    transform_bboxes_letterbox,
    create_target
)


# Class mapping (từ meta.json)
# Lọc ra 3 classes chính có hình dạng khác biệt
CLASS_MAPPING = {
    'Cap1': 0,        # Capacitor loại 1 (tụ điện lớn)
    'Resistor': 1,    # Điện trở
    'Transformer': 2, # Biến áp
}

# Có thể mở rộng thêm các class khác nếu cần
FULL_CLASS_MAPPING = {
    'Cap1': 0,
    'Cap2': 1,
    'Cap3': 2,
    'Cap4': 3,
    'MOSFET': 4,
    'Mov': 5,
    'Resistor': 6,
    'Resestor': 6,  # Typo trong dataset, merge với Resistor
    'Transformer': 7,
}


class PCBDataset:
    """
    Dataset class cho PCB Component Detection.
    
    Đọc ảnh và annotations từ format Supervisely JSON,
    áp dụng letterbox preprocessing, và trả về batch.
    """
    
    def __init__(
        self, 
        data_dir,
        split='train',
        target_size=224,
        num_classes=3,
        grid_size=7,
        class_mapping=None,
        pad_value=128
    ):
        """
        Args:
            data_dir: Đường dẫn đến dataset root
            split: 'train', 'validation', hoặc 'test'
            target_size: Kích thước ảnh sau resize (vuông)
            num_classes: Số classes
            grid_size: Kích thước grid output
            class_mapping: Dict mapping class name -> class id
            pad_value: Giá trị padding (0-255)
        """
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.pad_value = pad_value
        
        # Class mapping
        self.class_mapping = class_mapping or CLASS_MAPPING
        
        # Paths
        self.img_dir = os.path.join(data_dir, split, 'img')
        self.ann_dir = os.path.join(data_dir, split, 'ann')
        
        # Load file list
        self.samples = self._load_samples()
        
        print(f"[PCBDataset] Loaded {len(self.samples)} samples from {split}")
        print(f"             Classes: {list(self.class_mapping.keys())}")
    
    def _load_samples(self):
        """Load danh sách các samples có annotations hợp lệ."""
        samples = []
        
        if not os.path.exists(self.ann_dir):
            print(f"[WARNING] Annotation dir not found: {self.ann_dir}")
            return samples
        
        ann_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.json')]
        
        for ann_file in ann_files:
            # Image file name (remove .json suffix)
            img_name = ann_file.replace('.json', '')
            img_path = os.path.join(self.img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            ann_path = os.path.join(self.ann_dir, ann_file)
            
            # Parse annotation để kiểm tra có objects hợp lệ không
            bboxes = self._parse_annotation(ann_path)
            
            if len(bboxes) > 0:  # Chỉ lấy ảnh có objects
                samples.append({
                    'img_path': img_path,
                    'ann_path': ann_path,
                    'bboxes': bboxes
                })
        
        return samples
    
    def _parse_annotation(self, ann_path):
        """
        Parse Supervisely JSON annotation.
        
        Returns:
            bboxes: List of [class_id, x_center, y_center, width, height] (normalized)
        """
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
            
            # Chỉ lấy các class trong mapping
            if class_name not in self.class_mapping:
                continue
            
            class_id = self.class_mapping[class_name]
            
            # Parse bbox
            exterior = obj.get('points', {}).get('exterior', [])
            if len(exterior) != 2:
                continue
            
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]
            
            # Convert to center format và normalize
            x_center = (x1 + x2) / 2 / img_w
            y_center = (y1 + y2) / 2 / img_h
            width = abs(x2 - x1) / img_w
            height = abs(y2 - y1) / img_h
            
            # Clip to valid range
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
        Lấy một sample.
        
        Returns:
            image: (C, H, W) normalized [0, 1]
            target: (5 + num_classes, grid_size, grid_size)
            info: dict với letterbox info
        """
        sample = self.samples[idx]
        
        # Load image
        img = Image.open(sample['img_path']).convert('RGB')
        img = np.array(img, dtype=np.float32)  # (H, W, 3)
        
        # Letterbox preprocessing
        img_letterbox, info = letterbox_image(
            img, 
            target_size=self.target_size,
            pad_value=self.pad_value
        )
        
        # Transform bboxes
        bboxes = sample['bboxes']
        bboxes_letterbox = transform_bboxes_letterbox(bboxes, info)
        
        # Create target grid
        target = create_target(
            bboxes_letterbox,
            num_classes=self.num_classes,
            grid_size=self.grid_size
        )
        
        # Normalize image to [0, 1] và chuyển sang (C, H, W)
        img_letterbox = img_letterbox / 255.0
        img_letterbox = np.transpose(img_letterbox, (2, 0, 1))  # (3, H, W)
        
        return img_letterbox.astype(np.float32), target, info
    
    def get_batch(self, batch_indices):
        """
        Lấy một batch samples.
        
        Args:
            batch_indices: List of indices
            
        Returns:
            images: (N, C, H, W)
            targets: (N, 5 + num_classes, grid_size, grid_size)
            infos: List of info dicts
        """
        images = []
        targets = []
        infos = []
        
        for idx in batch_indices:
            img, target, info = self[idx]
            images.append(img)
            targets.append(target)
            infos.append(info)
        
        return np.stack(images), np.stack(targets), infos
    
    def iterate_batches(self, batch_size, shuffle=True):
        """
        Generator để iterate qua dataset theo batches.
        
        Args:
            batch_size: Số samples mỗi batch
            shuffle: Có shuffle không
            
        Yields:
            images: (N, C, H, W)
            targets: (N, 5 + num_classes, grid_size, grid_size)
        """
        indices = np.arange(len(self))
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(self), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            images, targets, _ = self.get_batch(batch_indices)
            yield images, targets
    
    def get_class_distribution(self):
        """Thống kê phân bố classes trong dataset."""
        counts = {name: 0 for name in self.class_mapping.keys()}
        
        for sample in self.samples:
            for bbox in sample['bboxes']:
                class_id = bbox[0]
                for name, cid in self.class_mapping.items():
                    if cid == class_id:
                        counts[name] += 1
                        break
        
        return counts


def test_dataset():
    """Test dataset loading."""
    print("=" * 50)
    print("Testing PCB Dataset")
    print("=" * 50)
    
    # Dataset path
    data_dir = os.path.join(
        os.path.dirname(__file__),
        '..', 'data', 'pcb-component-detection-DatasetNinja'
    )
    data_dir = os.path.abspath(data_dir)
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Dataset not found at: {data_dir}")
        return
    
    # Load dataset
    dataset = PCBDataset(
        data_dir=data_dir,
        split='train',
        target_size=224,
        num_classes=3,
        grid_size=7
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Class distribution
    dist = dataset.get_class_distribution()
    print("\nClass distribution:")
    for name, count in dist.items():
        print(f"   {name}: {count}")
    
    # Load a sample
    print("\nLoading sample 0...")
    img, target, info = dataset[0]
    print(f"   Image shape: {img.shape}")
    print(f"   Target shape: {target.shape}")
    print(f"   Letterbox info: orig={info['orig_w']}x{info['orig_h']}, padded={info['padded_size']}")
    print(f"   Objects in target: {int(target[4].sum())}")
    
    # Test batch loading
    print("\nTesting batch loading...")
    for i, (images, targets) in enumerate(dataset.iterate_batches(batch_size=4, shuffle=True)):
        print(f"   Batch {i}: images={images.shape}, targets={targets.shape}")
        if i >= 2:  # Only show first 3 batches
            break
    
    print("\n[OK] Dataset test passed!")


if __name__ == "__main__":
    test_dataset()
