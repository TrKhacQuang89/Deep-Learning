"""
Script phân tích thống kê dataset PCB Component Detection.
Tính toán các đặc tính quan trọng của dataset.
"""

import os
import json
from collections import defaultdict
import numpy as np


def analyze_dataset(dataset_path):
    """
    Phân tích dataset PCB Component Detection.
    
    Args:
        dataset_path: Đường dẫn đến thư mục dataset
    
    Returns:
        dict: Thống kê của dataset
    """
    stats = {
        'splits': {},
        'classes': defaultdict(int),
        'class_per_split': {},
        'images_with_objects': 0,
        'images_without_objects': 0,
        'total_objects': 0,
        'objects_per_image': [],
        'bbox_widths': [],
        'bbox_heights': [],
        'bbox_areas': [],
        'image_widths': [],
        'image_heights': [],
        'aspect_ratios': [],  # bbox width / height
    }
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        ann_path = os.path.join(split_path, 'ann')
        img_path = os.path.join(split_path, 'img')
        
        if not os.path.exists(ann_path):
            continue
            
        # Đếm số ảnh
        ann_files = [f for f in os.listdir(ann_path) if f.endswith('.json')]
        img_files = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        stats['splits'][split] = {
            'images': len(img_files),
            'annotations': len(ann_files)
        }
        
        # Khởi tạo class count cho split
        stats['class_per_split'][split] = defaultdict(int)
        
        # Phân tích từng file annotation
        for ann_file in ann_files:
            ann_file_path = os.path.join(ann_path, ann_file)
            
            with open(ann_file_path, 'r', encoding='utf-8') as f:
                try:
                    ann_data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[WARNING] Cannot parse: {ann_file}")
                    continue
            
            # Kích thước ảnh
            if 'size' in ann_data:
                img_width = ann_data['size'].get('width', 0)
                img_height = ann_data['size'].get('height', 0)
                stats['image_widths'].append(img_width)
                stats['image_heights'].append(img_height)
            
            # Các objects trong ảnh
            objects = ann_data.get('objects', [])
            num_objects = len(objects)
            stats['objects_per_image'].append(num_objects)
            
            if num_objects > 0:
                stats['images_with_objects'] += 1
            else:
                stats['images_without_objects'] += 1
            
            stats['total_objects'] += num_objects
            
            for obj in objects:
                # Class name
                class_title = obj.get('classTitle', 'Unknown')
                stats['classes'][class_title] += 1
                stats['class_per_split'][split][class_title] += 1
                
                # Bounding box
                points = obj.get('points', {})
                exterior = points.get('exterior', [])
                
                if len(exterior) == 2:
                    x1, y1 = exterior[0]
                    x2, y2 = exterior[1]
                    
                    bbox_width = abs(x2 - x1)
                    bbox_height = abs(y2 - y1)
                    bbox_area = bbox_width * bbox_height
                    
                    stats['bbox_widths'].append(bbox_width)
                    stats['bbox_heights'].append(bbox_height)
                    stats['bbox_areas'].append(bbox_area)
                    
                    if bbox_height > 0:
                        stats['aspect_ratios'].append(bbox_width / bbox_height)
    
    return stats


def print_statistics(stats):
    """In thống kê ra terminal."""
    
    print("=" * 60)
    print("PCB COMPONENT DETECTION - DATASET STATISTICS")
    print("=" * 60)
    
    # 1. Tổng quan splits
    print("\n1. DATASET SPLITS")
    print("-" * 40)
    total_images = 0
    for split, data in stats['splits'].items():
        print(f"   {split:15}: {data['images']:5} images, {data['annotations']:5} annotations")
        total_images += data['images']
    print(f"   {'TOTAL':15}: {total_images:5} images")
    
    # 2. Phân bố classes
    print("\n2. CLASS DISTRIBUTION (All splits)")
    print("-" * 40)
    sorted_classes = sorted(stats['classes'].items(), key=lambda x: -x[1])
    for class_name, count in sorted_classes:
        pct = count / stats['total_objects'] * 100 if stats['total_objects'] > 0 else 0
        bar = '#' * int(pct / 2)
        print(f"   {class_name:15}: {count:5} ({pct:5.1f}%) {bar}")
    print(f"   {'TOTAL':15}: {stats['total_objects']:5}")
    
    # 3. Phân bố classes theo split
    print("\n3. CLASS DISTRIBUTION BY SPLIT")
    print("-" * 40)
    for split, class_counts in stats['class_per_split'].items():
        print(f"\n   [{split.upper()}]")
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"      {class_name:15}: {count:5}")
    
    # 4. Objects per image
    print("\n4. OBJECTS PER IMAGE")
    print("-" * 40)
    if stats['objects_per_image']:
        objs = np.array(stats['objects_per_image'])
        print(f"   Min:    {np.min(objs):.0f}")
        print(f"   Max:    {np.max(objs):.0f}")
        print(f"   Mean:   {np.mean(objs):.2f}")
        print(f"   Median: {np.median(objs):.1f}")
        print(f"   Std:    {np.std(objs):.2f}")
    print(f"   Images with objects:    {stats['images_with_objects']}")
    print(f"   Images without objects: {stats['images_without_objects']}")
    
    # 5. Image sizes
    print("\n5. IMAGE SIZES")
    print("-" * 40)
    if stats['image_widths']:
        widths = np.array(stats['image_widths'])
        heights = np.array(stats['image_heights'])
        print(f"   Width  - Min: {np.min(widths):.0f}, Max: {np.max(widths):.0f}, Mean: {np.mean(widths):.1f}")
        print(f"   Height - Min: {np.min(heights):.0f}, Max: {np.max(heights):.0f}, Mean: {np.mean(heights):.1f}")
    
    # 6. Bounding box sizes
    print("\n6. BOUNDING BOX SIZES (in pixels)")
    print("-" * 40)
    if stats['bbox_widths']:
        bw = np.array(stats['bbox_widths'])
        bh = np.array(stats['bbox_heights'])
        ba = np.array(stats['bbox_areas'])
        ar = np.array(stats['aspect_ratios'])
        
        print(f"   Width:")
        print(f"      Min: {np.min(bw):.0f}, Max: {np.max(bw):.0f}, Mean: {np.mean(bw):.1f}, Std: {np.std(bw):.1f}")
        print(f"   Height:")
        print(f"      Min: {np.min(bh):.0f}, Max: {np.max(bh):.0f}, Mean: {np.mean(bh):.1f}, Std: {np.std(bh):.1f}")
        print(f"   Area:")
        print(f"      Min: {np.min(ba):.0f}, Max: {np.max(ba):.0f}, Mean: {np.mean(ba):.1f}")
        print(f"   Aspect Ratio (W/H):")
        print(f"      Min: {np.min(ar):.2f}, Max: {np.max(ar):.2f}, Mean: {np.mean(ar):.2f}")
    
    # 7. Normalized bbox sizes (relative to image)
    print("\n7. NORMALIZED BOUNDING BOX SIZES (relative to image)")
    print("-" * 40)
    if stats['bbox_widths'] and stats['image_widths']:
        # Tính normalized size cho mỗi bbox
        norm_widths = []
        norm_heights = []
        
        # Cần track cùng lúc bbox và image size
        # Do đã lưu riêng nên cần tính lại
        # Ước lượng bằng mean image size
        mean_img_w = np.mean(stats['image_widths'])
        mean_img_h = np.mean(stats['image_heights'])
        
        norm_w = np.array(stats['bbox_widths']) / mean_img_w
        norm_h = np.array(stats['bbox_heights']) / mean_img_h
        norm_a = norm_w * norm_h
        
        print(f"   (Using mean image size: {mean_img_w:.0f}x{mean_img_h:.0f})")
        print(f"   Normalized Width:")
        print(f"      Min: {np.min(norm_w):.3f}, Max: {np.max(norm_w):.3f}, Mean: {np.mean(norm_w):.3f}")
        print(f"   Normalized Height:")
        print(f"      Min: {np.min(norm_h):.3f}, Max: {np.max(norm_h):.3f}, Mean: {np.mean(norm_h):.3f}")
        print(f"   Normalized Area:")
        print(f"      Min: {np.min(norm_a):.4f}, Max: {np.max(norm_a):.4f}, Mean: {np.mean(norm_a):.4f}")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


def main():
    # Đường dẫn dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        'data', 'pcb-component-detection-DatasetNinja'
    )
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        return
    
    print(f"Analyzing dataset at: {dataset_path}\n")
    
    # Phân tích
    stats = analyze_dataset(dataset_path)
    
    # In kết quả
    print_statistics(stats)


if __name__ == "__main__":
    main()
