"""
Script phân tích và visualize image sizes của dataset PCB Component Detection.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def collect_image_data(dataset_path):
    """Thu thập dữ liệu kích thước ảnh và bounding box."""
    
    data = {
        'image_widths': [],
        'image_heights': [],
        'image_aspect_ratios': [],  # width / height
        'bbox_widths': [],
        'bbox_heights': [],
        'bbox_aspect_ratios': [],
        'normalized_bbox_widths': [],
        'normalized_bbox_heights': [],
    }
    
    splits = ['train', 'validation', 'test']
    
    for split in splits:
        ann_path = os.path.join(dataset_path, split, 'ann')
        if not os.path.exists(ann_path):
            continue
            
        ann_files = [f for f in os.listdir(ann_path) if f.endswith('.json')]
        
        for ann_file in ann_files:
            with open(os.path.join(ann_path, ann_file), 'r', encoding='utf-8') as f:
                try:
                    ann_data = json.load(f)
                except:
                    continue
            
            # Image size
            if 'size' in ann_data:
                img_w = ann_data['size'].get('width', 0)
                img_h = ann_data['size'].get('height', 0)
                
                if img_w > 0 and img_h > 0:
                    data['image_widths'].append(img_w)
                    data['image_heights'].append(img_h)
                    data['image_aspect_ratios'].append(img_w / img_h)
                    
                    # Bounding boxes
                    for obj in ann_data.get('objects', []):
                        exterior = obj.get('points', {}).get('exterior', [])
                        if len(exterior) == 2:
                            x1, y1 = exterior[0]
                            x2, y2 = exterior[1]
                            
                            bbox_w = abs(x2 - x1)
                            bbox_h = abs(y2 - y1)
                            
                            if bbox_w > 0 and bbox_h > 0:
                                data['bbox_widths'].append(bbox_w)
                                data['bbox_heights'].append(bbox_h)
                                data['bbox_aspect_ratios'].append(bbox_w / bbox_h)
                                data['normalized_bbox_widths'].append(bbox_w / img_w)
                                data['normalized_bbox_heights'].append(bbox_h / img_h)
    
    return data


def plot_image_size_analysis(data, save_path=None):
    """Vẽ các biểu đồ phân tích kích thước ảnh."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PCB Component Detection - Image & BBox Size Analysis', fontsize=14, fontweight='bold')
    
    # 1. Image Aspect Ratio Distribution (Histogram)
    ax1 = axes[0, 0]
    ax1.hist(data['image_aspect_ratios'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(data['image_aspect_ratios']), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(data['image_aspect_ratios']):.2f}")
    ax1.axvline(np.median(data['image_aspect_ratios']), color='orange', linestyle='--', linewidth=2, label=f"Median: {np.median(data['image_aspect_ratios']):.2f}")
    ax1.set_xlabel('Aspect Ratio (W/H)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Image Aspect Ratio Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Image Width vs Height Scatter
    ax2 = axes[0, 1]
    ax2.scatter(data['image_widths'], data['image_heights'], alpha=0.5, c='steelblue', s=20)
    ax2.set_xlabel('Width (pixels)')
    ax2.set_ylabel('Height (pixels)')
    ax2.set_title('Image Dimensions Scatter')
    ax2.grid(True, alpha=0.3)
    # Add reference lines for common aspect ratios
    max_dim = max(max(data['image_widths']), max(data['image_heights']))
    ax2.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='1:1')
    ax2.plot([0, max_dim], [0, max_dim * 4/3], 'g--', alpha=0.5, label='4:3')
    ax2.plot([0, max_dim], [0, max_dim * 16/9], 'b--', alpha=0.5, label='16:9')
    ax2.legend(loc='upper left')
    
    # 3. Image Size Distribution (2D Histogram)
    ax3 = axes[0, 2]
    unique_sizes = list(zip(data['image_widths'], data['image_heights']))
    size_counts = defaultdict(int)
    for size in unique_sizes:
        size_counts[size] += 1
    
    sorted_sizes = sorted(size_counts.items(), key=lambda x: -x[1])[:15]
    labels = [f"{w}x{h}" for (w, h), _ in sorted_sizes]
    counts = [c for _, c in sorted_sizes]
    
    bars = ax3.barh(range(len(labels)), counts, color='steelblue', edgecolor='black')
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels)
    ax3.set_xlabel('Count')
    ax3.set_title('Top 15 Image Sizes')
    ax3.invert_yaxis()
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. BBox Aspect Ratio Distribution
    ax4 = axes[1, 0]
    ax4.hist(data['bbox_aspect_ratios'], bins=50, color='coral', edgecolor='black', alpha=0.7, range=(0, 3))
    ax4.axvline(1.0, color='green', linestyle='-', linewidth=2, label='Square (1:1)')
    ax4.axvline(np.mean(data['bbox_aspect_ratios']), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(data['bbox_aspect_ratios']):.2f}")
    ax4.set_xlabel('Aspect Ratio (W/H)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('BBox Aspect Ratio Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Normalized BBox Size Distribution
    ax5 = axes[1, 1]
    ax5.hist2d(data['normalized_bbox_widths'], data['normalized_bbox_heights'], 
               bins=30, cmap='YlOrRd', range=[[0, 0.6], [0, 0.6]])
    ax5.set_xlabel('Normalized Width (bbox_w / img_w)')
    ax5.set_ylabel('Normalized Height (bbox_h / img_h)')
    ax5.set_title('Normalized BBox Size Distribution')
    ax5.plot([0, 0.6], [0, 0.6], 'b--', alpha=0.5, label='1:1')
    ax5.legend()
    plt.colorbar(ax5.collections[0], ax=ax5, label='Count')
    
    # 6. BBox Size in Pixels
    ax6 = axes[1, 2]
    ax6.hist2d(data['bbox_widths'], data['bbox_heights'], 
               bins=40, cmap='YlOrRd', range=[[0, 500], [0, 500]])
    ax6.set_xlabel('Width (pixels)')
    ax6.set_ylabel('Height (pixels)')
    ax6.set_title('BBox Size Distribution (pixels)')
    ax6.plot([0, 500], [0, 500], 'b--', alpha=0.5)
    plt.colorbar(ax6.collections[0], ax=ax6, label='Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    plt.show()


def print_detailed_stats(data):
    """In thống kê chi tiết."""
    
    print("\n" + "=" * 60)
    print("DETAILED IMAGE SIZE STATISTICS")
    print("=" * 60)
    
    # Image sizes
    widths = np.array(data['image_widths'])
    heights = np.array(data['image_heights'])
    ar = np.array(data['image_aspect_ratios'])
    
    print("\n1. IMAGE DIMENSIONS")
    print("-" * 40)
    print(f"   Total images analyzed: {len(widths)}")
    print(f"\n   Width (pixels):")
    print(f"      Min:    {np.min(widths)}")
    print(f"      Max:    {np.max(widths)}")
    print(f"      Mean:   {np.mean(widths):.1f}")
    print(f"      Median: {np.median(widths):.1f}")
    print(f"      Std:    {np.std(widths):.1f}")
    
    print(f"\n   Height (pixels):")
    print(f"      Min:    {np.min(heights)}")
    print(f"      Max:    {np.max(heights)}")
    print(f"      Mean:   {np.mean(heights):.1f}")
    print(f"      Median: {np.median(heights):.1f}")
    print(f"      Std:    {np.std(heights):.1f}")
    
    print("\n2. IMAGE ASPECT RATIO (Width / Height)")
    print("-" * 40)
    print(f"   Min:    {np.min(ar):.4f}")
    print(f"   Max:    {np.max(ar):.4f}")
    print(f"   Mean:   {np.mean(ar):.4f}")
    print(f"   Median: {np.median(ar):.4f}")
    print(f"   Std:    {np.std(ar):.4f}")
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    print(f"\n   Percentiles:")
    for p in percentiles:
        print(f"      P{p:2d}: {np.percentile(ar, p):.4f}")
    
    # Unique sizes
    unique_sizes = set(zip(data['image_widths'], data['image_heights']))
    print(f"\n3. UNIQUE IMAGE SIZES: {len(unique_sizes)}")
    print("-" * 40)
    
    size_counts = defaultdict(int)
    for w, h in zip(data['image_widths'], data['image_heights']):
        size_counts[(w, h)] += 1
    
    sorted_sizes = sorted(size_counts.items(), key=lambda x: -x[1])[:10]
    for (w, h), count in sorted_sizes:
        pct = count / len(widths) * 100
        print(f"   {w:4d} x {h:4d}: {count:5d} images ({pct:5.1f}%)")
    
    # Portrait vs Landscape
    portrait = np.sum(heights > widths)
    landscape = np.sum(widths > heights)
    square = np.sum(widths == heights)
    
    print(f"\n4. ORIENTATION")
    print("-" * 40)
    print(f"   Portrait  (H > W): {portrait:5d} ({portrait/len(widths)*100:.1f}%)")
    print(f"   Landscape (W > H): {landscape:5d} ({landscape/len(widths)*100:.1f}%)")
    print(f"   Square    (W = H): {square:5d} ({square/len(widths)*100:.1f}%)")
    
    print("\n" + "=" * 60)


def main():
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        'data', 'pcb-component-detection-DatasetNinja'
    )
    
    print(f"Analyzing dataset at: {dataset_path}")
    
    # Thu thập dữ liệu
    data = collect_image_data(dataset_path)
    
    # In thống kê chi tiết
    print_detailed_stats(data)
    
    # Vẽ biểu đồ
    save_path = os.path.join(os.path.dirname(__file__), 'dataset_analysis.png')
    plot_image_size_analysis(data, save_path)


if __name__ == "__main__":
    main()
