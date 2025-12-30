import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_logs(log_dir='checkpoints'):
    """
    Vẽ biểu đồ loss từ training_log.csv và batch_log.csv.
    """
    train_log_path = os.path.join(log_dir, 'training_log.csv')
    batch_log_path = os.path.join(log_dir, 'batch_log.csv')
    
    if not os.path.exists(train_log_path) or not os.path.exists(batch_log_path):
        print(f"[ERROR] Không tìm thấy file log tại: {log_dir}")
        return

    # Load data
    df_epoch = pd.read_csv(train_log_path)
    df_batch = pd.read_csv(batch_log_path)
    
    # Create figure
    plt.figure(figsize=(15, 6))
    
    # --- Biểu đồ 1: Batch Loss (Fine-grained) ---
    plt.subplot(1, 2, 1)
    plt.plot(df_batch.index, df_batch['loss'], alpha=0.3, color='blue', label='Batch Loss')
    
    # Moving average for smoother curve
    window = min(len(df_batch), 20)
    if window > 1:
        sma = df_batch['loss'].rolling(window=window).mean()
        plt.plot(df_batch.index, sma, color='red', label=f'Moving Average (prev {window})')
    
    plt.title('Training Loss per Batch')
    plt.xlabel('Log Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # --- Biểu đồ 2: Epoch Loss (Train vs Val) ---
    plt.subplot(1, 2, 2)
    plt.plot(df_epoch['epoch'], df_epoch['train_loss'], marker='o', label='Train Loss')
    plt.plot(df_epoch['epoch'], df_epoch['val_loss'], marker='s', label='Val Loss')
    
    plt.title('Training vs Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(log_dir, 'loss_plot.png')
    plt.savefig(save_path)
    print(f"[SAVE] Biểu đồ đã được lưu tại: {save_path}")
    
    # Show plot (nếu có môi trường display)
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    # Đảm bảo script chạy đúng thư mục
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(current_dir, 'checkpoints')
    plot_logs(log_dir)
