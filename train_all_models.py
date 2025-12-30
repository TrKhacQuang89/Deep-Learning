"""
Train All Models Script

Trains all detector models (tiny → huge) sequentially.
- Starts with batch size 32
- Falls back to batch size 16 on OOM
- Continues to next model even if one fails
"""

import subprocess
import sys
import os

# Models to train (in order of size)
MODELS = ['tiny', 'small', 'medium', 'large', 'xlarge', 'huge']

# Training config
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
FALLBACK_BATCH_SIZE = 16


def train_model(model_name: str, batch_size: int, epochs: int = DEFAULT_EPOCHS) -> bool:
    """
    Train a single model.
    
    Returns:
        True if successful, False if failed
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} with batch_size={batch_size}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "train_torch.py",
        "--model", model_name,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--save_dir", f"checkpoints_{model_name}"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=False,  # Show output in real-time
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] {model_name}: {e}")
        return False


def train_with_fallback(model_name: str, epochs: int = DEFAULT_EPOCHS) -> dict:
    """
    Train a model with OOM fallback.
    
    First tries with batch_size=32, if OOM detected, retries with batch_size=16.
    
    Returns:
        dict with training result info
    """
    result = {
        'model': model_name,
        'success': False,
        'batch_size': DEFAULT_BATCH_SIZE,
        'error': None
    }
    
    # Try with default batch size
    print(f"\n[INFO] Attempting {model_name} with batch_size={DEFAULT_BATCH_SIZE}")
    
    try:
        success = train_model(model_name, DEFAULT_BATCH_SIZE, epochs)
        
        if success:
            result['success'] = True
            return result
        else:
            # Check if it was likely an OOM (non-zero exit code)
            print(f"[WARNING] {model_name} failed with batch_size={DEFAULT_BATCH_SIZE}, trying with {FALLBACK_BATCH_SIZE}")
            
    except Exception as e:
        print(f"[WARNING] Exception during training: {e}")
    
    # Retry with smaller batch size
    result['batch_size'] = FALLBACK_BATCH_SIZE
    print(f"\n[INFO] Retrying {model_name} with batch_size={FALLBACK_BATCH_SIZE}")
    
    try:
        success = train_model(model_name, FALLBACK_BATCH_SIZE, epochs)
        result['success'] = success
        if not success:
            result['error'] = "Training failed even with reduced batch size"
    except Exception as e:
        result['error'] = str(e)
    
    return result


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train all detector models')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs per model')
    parser.add_argument('--models', nargs='+', default=MODELS, 
                        choices=MODELS + ['giant'],
                        help='Models to train (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MULTI-MODEL TRAINING")
    print("="*60)
    print(f"Models: {args.models}")
    print(f"Epochs: {args.epochs}")
    print(f"Initial batch size: {DEFAULT_BATCH_SIZE}")
    print(f"Fallback batch size: {FALLBACK_BATCH_SIZE}")
    print("="*60)
    
    results = []
    
    for model_name in args.models:
        try:
            result = train_with_fallback(model_name, args.epochs)
            results.append(result)
        except KeyboardInterrupt:
            print(f"\n[INTERRUPTED] Stopping training at {model_name}")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error training {model_name}: {e}")
            results.append({
                'model': model_name,
                'success': False,
                'batch_size': None,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for r in results:
        status = "✓ SUCCESS" if r['success'] else "✗ FAILED"
        batch_info = f"batch={r['batch_size']}" if r['batch_size'] else ""
        error_info = f" ({r['error']})" if r.get('error') else ""
        print(f"  {r['model']:10s}: {status} {batch_info}{error_info}")
    
    # Count successes
    success_count = sum(1 for r in results if r['success'])
    print(f"\nTotal: {success_count}/{len(results)} models trained successfully")
    
    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
