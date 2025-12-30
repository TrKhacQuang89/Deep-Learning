import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import SimpleDetector
from detector.utils import letterbox_image, decode_predictions, nms, inverse_transform_bbox_letterbox
from train_detector import load_model

def run_inference(model_path, image_path, conf_thresh=0.3, iou_thresh=0.5):
    """
    Load model và chạy detection trên một tấm ảnh.
    """
    # 1. Khởi tạo model
    num_classes = 3
    class_names = ['Cap1', 'Resistor', 'Transformer']
    model = SimpleDetector(num_classes=num_classes)
    
    # 2. Load weights
    metadata = load_model(model, model_path)
    if metadata:
        print(f"[INFO] Loaded model from epoch {metadata.get('epoch', 'unknown')}")
    
    # 3. Đọc và preprocess image
    if not os.path.exists(image_path):
        print(f"[ERROR] Không tìm thấy ảnh: {image_path}")
        return
        
    orig_img = Image.open(image_path).convert('RGB')
    orig_np = np.array(orig_img)
    
    # Letterbox resize
    img_sized, info = letterbox_image(orig_np, target_size=224)
    
    # Chuẩn hóa (N, C, H, W)
    x = img_sized.transpose(2, 0, 1) / 255.0
    x = x[np.newaxis, ...].astype(np.float32)
    
    # 4. Inference
    output = model.forward(x)[0]  # Tháo batch dimension -> (8, 7, 7)
    
    # 5. Decode & NMS
    boxes = decode_predictions(output, conf_thresh=conf_thresh)
    boxes = nms(boxes, iou_thresh=iou_thresh)
    
    print(f"[INFO] Found {len(boxes)} objects.")
    
    # 6. Visualization
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(orig_np)
    
    for box in boxes:
        # box: [x1, y1, x2, y2, conf, cls_id] trong không gian 224x224
        # Cần chuyển về không gian ảnh gốc
        norm_box = [
            (box[0] + box[2]) / 2, # center x
            (box[1] + box[3]) / 2, # center y
            box[2] - box[0],       # width
            box[3] - box[1]        # height
        ]
        
        orig_box_norm = inverse_transform_bbox_letterbox(norm_box, info)
        
        # Chuyển từ normalized -> pixel coordinates
        w_orig, h_orig = orig_img.size
        x_c, y_c, w, h = orig_box_norm
        
        x1 = (x_c - w/2) * w_orig
        y1 = (y_c - h/2) * h_orig
        bw = w * w_orig
        bh = h * h_orig
        
        # Vẽ box
        rect = patches.Rectangle(
            (x1, y1), bw, bh, 
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Label
        label = f"{class_names[int(box[5])]}: {box[4]:.2f}"
        plt.text(
            x1, y1 - 5, label, 
            color='white', verticalalignment='top',
            bbox={'color': 'red', 'pad': 0}
        )
    
    plt.axis('off')
    plt.title(f"Detection Results: {os.path.basename(image_path)}")
    
    # Save result
    save_dir = 'results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"det_{os.path.basename(image_path)}")
    plt.savefig(save_path)
    print(f"[SAVE] Result saved to: {save_path}")
    
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    # Đường dẫn mặc định (bạn có thể thay đổi)
    model_file = "checkpoints/best_model.npz"
    
    # Lấy một ảnh bất kỳ từ validation set để test
    test_image = "data/pcb-component-detection-DatasetNinja/validation/img/VID20210601144014-45_jpg.rf.1fc301cb2464b72cbf29ddae321791c3.jpg"
    
    # Chuyển thành đường dẫn tuyệt đối
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(curr_dir, model_file)
    image_path = os.path.join(curr_dir, test_image)
    
    run_inference(model_path, image_path)
