"""
Data loader module for GTSRB (German Traffic Sign Recognition Benchmark) dataset.
Provides functions to load, preprocess, and split the dataset.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, Optional


class GTSRBDataLoader:
    """Data loader for GTSRB dataset."""
    
    def __init__(self, dataset_path: str = "data/GTSRB"):
        """
        Initialize the GTSRB data loader.
        
        Args:
            dataset_path: Path to the GTSRB dataset directory
        """
        self.dataset_path = dataset_path
        self.images = []
        self.labels = []
        self.class_names = self._load_class_names()
    
    def _load_class_names(self) -> dict:
        """
        Load traffic sign class names.
        
        Returns:
            Dictionary mapping class indices to class names
        """
        class_names = {
            0: "Speed limit (20km/h)",
            1: "Speed limit (30km/h)",
            2: "Speed limit (50km/h)",
            3: "Speed limit (60km/h)",
            4: "Speed limit (70km/h)",
            5: "Speed limit (80km/h)",
            6: "End of speed limit (80km/h)",
            7: "Speed limit (100km/h)",
            8: "Speed limit (120km/h)",
            9: "No passing",
            10: "No passing for vehicles over 3.5 metric tons",
            11: "Right-of-way at the next intersection",
            12: "Priority road",
            13: "Yield",
            14: "Stop",
            15: "No entry",
            16: "Vehicles over 3.5 metric tons prohibited",
            17: "No entry for vehicles over 3.5 metric tons",
            18: "No entry",
            19: "General caution",
            20: "Dangerous curve to the left",
            21: "Dangerous curve to the right",
            22: "Double curve",
            23: "Bumpy road",
            24: "Slippery road",
            25: "Road narrows on the right",
            26: "Road work",
            27: "Traffic signals",
            28: "Pedestrians",
            29: "Children",
            30: "Bicycles",
            31: "Beware of ice/snow",
            32: "Wild animals crossing",
            33: "End of all speed and passing limits",
            34: "Turn right ahead",
            35: "Turn left ahead",
            36: "Ahead only",
            37: "Go straight or right",
            38: "Go straight or left",
            39: "Keep right",
            40: "Keep left",
            41: "Roundabout mandatory",
            42: "End of no passing",
            43: "End of no passing for vehicles over 3.5 metric tons"
        }
        return class_names
    
    def load_images(self, img_size: Tuple[int, int] = (32, 32)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from the GTSRB dataset.
        
        Args:
            img_size: Target image size (height, width)
            
        Returns:
            Tuple of (images array, labels array)
        """
        images = []
        labels = []
        
        # Check if dataset exists
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        # Iterate through class directories
        for class_idx in range(44):  # GTSRB has 43 classes (0-42)
            class_path = os.path.join(self.dataset_path, f"{class_idx:05d}")
            
            if not os.path.exists(class_path):
                continue
            
            # Load all images in the class directory
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.ppm', '.jpg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize image
                            img = cv2.resize(img, img_size)
                            # Convert BGR to RGB
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)
                            labels.append(class_idx)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
        
        self.images = np.array(images)
        self.labels = np.array(labels)
        
        print(f"Loaded {len(images)} images from {self.dataset_path}")
        return self.images, self.labels
    
    def preprocess(self, 
                   images: np.ndarray,
                   normalize: bool = True,
                   augment: bool = False) -> np.ndarray:
        """
        Preprocess images.
        
        Args:
            images: Input images array
            normalize: Whether to normalize pixel values to [0, 1]
            augment: Whether to apply data augmentation
            
        Returns:
            Preprocessed images array
        """
        preprocessed = images.copy().astype(np.float32)
        
        # Normalize pixel values
        if normalize:
            preprocessed = preprocessed / 255.0
        
        # Optional: Apply data augmentation
        if augment:
            preprocessed = self._augment_data(preprocessed)
        
        return preprocessed
    
    def _augment_data(self, images: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to images.
        
        Args:
            images: Input images array
            
        Returns:
            Augmented images array
        """
        augmented = []
        
        for img in images:
            # Random brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            img_aug = img * brightness_factor
            img_aug = np.clip(img_aug, 0, 1)
            
            augmented.append(img_aug)
        
        return np.array(augmented)
    
    def split_data(self,
                   images: np.ndarray,
                   labels: np.ndarray,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray]:
        """
        Split data into training, validation, and test sets.
        
        Args:
            images: Images array
            labels: Labels array
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_distribution(self, labels: np.ndarray) -> dict:
        """
        Get the distribution of classes.
        
        Args:
            labels: Labels array
            
        Returns:
            Dictionary with class distribution
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}
        return distribution


def load_gtsrb_dataset(dataset_path: str = "data/GTSRB",
                       img_size: Tuple[int, int] = (32, 32),
                       normalize: bool = True,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """
    Load and preprocess GTSRB dataset.
    
    Args:
        dataset_path: Path to the GTSRB dataset directory
        img_size: Target image size (height, width)
        normalize: Whether to normalize pixel values
        test_size: Proportion of data for testing
        val_size: Proportion of data for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    loader = GTSRBDataLoader(dataset_path)
    
    # Load images
    images, labels = loader.load_images(img_size)
    
    # Preprocess
    images = loader.preprocess(images, normalize=normalize, augment=False)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        images, labels,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # Example usage
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_gtsrb_dataset()
        print(f"\nDataset loaded successfully!")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the GTSRB dataset is available at the specified path.")
