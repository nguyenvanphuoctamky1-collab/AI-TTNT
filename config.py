"""
Configuration file for AI-TTNT project
Contains settings for dataset, model, training parameters, and GPU configuration
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
DATASET_CONFIG = {
    'name': 'default_dataset',
    'data_path': DATA_DIR,
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
    'batch_size': 32,
    'num_workers': 4,
    'shuffle': True,
    'pin_memory': True,
    'drop_last': True,
    'image_size': 224,
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'augmentation': {
        'random_flip': True,
        'random_rotation': 15,
        'random_crop': True,
        'color_jitter': True,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1,
    }
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_CONFIG = {
    'architecture': 'resnet50',  # Options: resnet18, resnet34, resnet50, vgg16, efficientnet, etc.
    'pretrained': True,
    'num_classes': 10,
    'input_channels': 3,
    'dropout_rate': 0.5,
    'use_batch_norm': True,
    'model_checkpoint': None,  # Set to path if resuming from checkpoint
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'optimizer': 'adam',  # Options: sgd, adam, adamw, rmsprop
    'scheduler': 'cosine',  # Options: step, exponential, cosine, warmup_cosine
    'warmup_epochs': 5,
    'patience': 10,  # For early stopping
    'min_delta': 1e-4,  # Minimum change to qualify as improvement
    'save_frequency': 5,  # Save checkpoint every N epochs
    'log_frequency': 10,  # Log metrics every N iterations
    'val_frequency': 1,  # Validate every N epochs
    'seed': 42,
    'mixed_precision': True,  # Enable mixed precision training
    'gradient_accumulation_steps': 1,
    'gradient_clip_max_norm': 1.0,
}

# ============================================================================
# GPU/DEVICE CONFIGURATION
# ============================================================================
GPU_CONFIG = {
    'use_gpu': True,
    'device': 'cuda',  # Options: cuda, cpu, mps (Metal Performance Shaders for Apple)
    'gpu_ids': [0],  # GPU IDs to use for multi-GPU training
    'num_gpus': 1,
    'distributed': False,  # Enable distributed data parallel training
    'find_unused_parameters': False,
    'precision': 'fp32',  # Options: fp32, fp16, bf16
    'cudnn_benchmark': True,  # Enable cudnn auto-tuner
    'cudnn_deterministic': False,  # Ensure reproducibility (may impact performance)
    'max_memory_allocated': None,  # Set max GPU memory in GB, None for unlimited
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    'log_dir': LOG_DIR,
    'log_file': os.path.join(LOG_DIR, 'training.log'),
    'log_level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'wandb_enabled': False,  # Enable Weights & Biases logging
    'wandb_project': 'ai-ttnt',
    'wandb_entity': None,
    'tensorboard_enabled': True,
    'checkpoint_dir': CHECKPOINT_DIR,
    'save_best_only': True,
    'monitor_metric': 'val_accuracy',  # Metric to monitor for best model
    'monitor_mode': 'max',  # Options: min, max
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
INFERENCE_CONFIG = {
    'batch_size': 128,
    'num_workers': 4,
    'use_tta': False,  # Test Time Augmentation
    'tta_iterations': 5,
    'confidence_threshold': 0.5,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_device():
    """
    Determine the device to use based on GPU configuration
    
    Returns:
        str: Device string ('cuda', 'cpu', etc.)
    """
    import torch
    
    if GPU_CONFIG['use_gpu'] and torch.cuda.is_available():
        return GPU_CONFIG['device']
    return 'cpu'

def get_num_gpus():
    """
    Get the number of available GPUs
    
    Returns:
        int: Number of available GPUs
    """
    import torch
    
    if GPU_CONFIG['use_gpu']:
        return torch.cuda.device_count()
    return 0

def print_config():
    """Print all configuration settings"""
    print("\n" + "="*80)
    print("PROJECT CONFIGURATION")
    print("="*80)
    
    print("\nDataset Configuration:")
    for key, value in DATASET_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nModel Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nGPU Configuration:")
    for key, value in GPU_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nLogging Configuration:")
    for key, value in LOGGING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nInference Configuration:")
    for key, value in INFERENCE_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    print_config()
