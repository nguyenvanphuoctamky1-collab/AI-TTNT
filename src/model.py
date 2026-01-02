"""
CNN Architectures for Traffic Sign Classification
Includes SimpleCNN, Sequential CNN, ResNet18, and EfficientNet models
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0
import numpy as np


class SimpleCNN:
    """
    Simple CNN architecture for traffic sign classification.
    Lightweight model suitable for initial experiments.
    """
    
    @staticmethod
    def build(input_shape=(64, 64, 3), num_classes=43):
        """
        Build SimpleCNN model.
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of traffic sign classes
            
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         input_shape=input_shape, name='conv1_1'),
            layers.BatchNormalization(name='bn1_1'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2'),
            layers.BatchNormalization(name='bn1_2'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.25, name='dropout1'),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2_1'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2'),
            layers.BatchNormalization(name='bn2_2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.25, name='dropout2'),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3_1'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2'),
            layers.BatchNormalization(name='bn3_2'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.25, name='dropout3'),
            
            # Flatten and Dense layers
            layers.Flatten(name='flatten'),
            layers.Dense(256, activation='relu', name='dense1'),
            layers.BatchNormalization(name='bn_dense1'),
            layers.Dropout(0.5, name='dropout_dense1'),
            
            layers.Dense(128, activation='relu', name='dense2'),
            layers.BatchNormalization(name='bn_dense2'),
            layers.Dropout(0.5, name='dropout_dense2'),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model


def sequential_cnn(input_shape=(64, 64, 3), num_classes=43):
    """
    Sequential CNN architecture for traffic sign classification.
    Medium complexity model with residual-like connections in mind.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of traffic sign classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # Preprocessing
        layers.Rescaling(1./255., name='rescaling'),
        
        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1'),
        layers.BatchNormalization(name='bn1_1'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2'),
        layers.BatchNormalization(name='bn1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1'),
        layers.BatchNormalization(name='bn2_1'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2'),
        layers.BatchNormalization(name='bn2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1'),
        layers.BatchNormalization(name='bn3_1'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2'),
        layers.BatchNormalization(name='bn3_2'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25, name='dropout3'),
        
        # Block 4
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1'),
        layers.BatchNormalization(name='bn4_1'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2'),
        layers.BatchNormalization(name='bn4_2'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        layers.Dropout(0.25, name='dropout4'),
        
        # Global Average Pooling
        layers.GlobalAveragePooling2D(name='gap'),
        
        # Dense layers
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn_dense1'),
        layers.Dropout(0.5, name='dropout_dense1'),
        
        layers.Dense(256, activation='relu', name='dense2'),
        layers.BatchNormalization(name='bn_dense2'),
        layers.Dropout(0.5, name='dropout_dense2'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def resnet18(input_shape=(64, 64, 3), num_classes=43):
    """
    ResNet18 architecture adapted for traffic sign classification.
    Implements residual blocks for improved gradient flow.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of traffic sign classes
        
    Returns:
        Compiled Keras model
    """
    
    def residual_block(x, filters, kernel_size=3, stride=1, name_prefix=''):
        """Create a residual block."""
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                         name=f'{name_prefix}_conv1')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1')(x)
        
        x = layers.Conv2D(filters, kernel_size, strides=1, padding='same',
                         name=f'{name_prefix}_conv2')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        
        # Shortcut path
        if stride != 1 or x.shape[-1] != shortcut.shape[-1]:
            shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                    name=f'{name_prefix}_shortcut_conv')(shortcut)
            shortcut = layers.BatchNormalization(name=f'{name_prefix}_shortcut_bn')(shortcut)
        
        # Add
        x = layers.Add(name=f'{name_prefix}_add')([x, shortcut])
        x = layers.Activation('relu', name=f'{name_prefix}_relu2')(x)
        
        return x
    
    # Input
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1./255.)(inputs)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', name='conv1')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Activation('relu', name='relu1')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='maxpool')(x)
    
    # Residual blocks (ResNet18 has 2 blocks per layer, 4 layers total)
    x = residual_block(x, 64, stride=1, name_prefix='block1_1')
    x = residual_block(x, 64, stride=1, name_prefix='block1_2')
    
    x = residual_block(x, 128, stride=2, name_prefix='block2_1')
    x = residual_block(x, 128, stride=1, name_prefix='block2_2')
    
    x = residual_block(x, 256, stride=2, name_prefix='block3_1')
    x = residual_block(x, 256, stride=1, name_prefix='block3_2')
    
    x = residual_block(x, 512, stride=2, name_prefix='block4_1')
    x = residual_block(x, 512, stride=1, name_prefix='block4_2')
    
    # Global Average Pooling and classification
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=x, name='ResNet18')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def efficient_net(input_shape=(64, 64, 3), num_classes=43):
    """
    EfficientNet-B0 pretrained architecture adapted for traffic sign classification.
    Transfer learning approach using ImageNet pretrained weights.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of traffic sign classes
        
    Returns:
        Compiled Keras model
    """
    
    # Load pretrained EfficientNetB0
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model weights initially
    base_model.trainable = False
    
    # Create full model
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Rescaling
        layers.Rescaling(1./255., name='rescaling'),
        
        # Base model
        base_model,
        
        # Custom top layers
        layers.GlobalAveragePooling2D(name='gap'),
        
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn_dense1'),
        layers.Dropout(0.5, name='dropout_dense1'),
        
        layers.Dense(256, activation='relu', name='dense2'),
        layers.BatchNormalization(name='bn_dense2'),
        layers.Dropout(0.5, name='dropout_dense2'),
        
        layers.Dense(128, activation='relu', name='dense3'),
        layers.BatchNormalization(name='bn_dense3'),
        layers.Dropout(0.3, name='dropout_dense3'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='EfficientNet-B0-TrafficSigns')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model(model_name='sequential_cnn', input_shape=(64, 64, 3), num_classes=43):
    """
    Factory function to get desired model.
    
    Args:
        model_name: Name of the model ('simple_cnn', 'sequential_cnn', 'resnet18', 'efficient_net')
        input_shape: Input image shape
        num_classes: Number of classes
        
    Returns:
        Compiled Keras model
        
    Raises:
        ValueError: If model_name is not recognized
    """
    models_dict = {
        'simple_cnn': SimpleCNN.build,
        'sequential_cnn': sequential_cnn,
        'resnet18': resnet18,
        'efficient_net': efficient_net
    }
    
    if model_name not in models_dict:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(models_dict.keys())}")
    
    return models_dict[model_name](input_shape=input_shape, num_classes=num_classes)


def print_model_summary(model_name='sequential_cnn', input_shape=(64, 64, 3), num_classes=43):
    """
    Print model summary for inspection.
    
    Args:
        model_name: Name of the model
        input_shape: Input image shape
        num_classes: Number of classes
    """
    model = get_model(model_name, input_shape, num_classes)
    print(f"\n{'='*80}")
    print(f"Model: {model_name.upper()}")
    print(f"{'='*80}\n")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")


if __name__ == '__main__':
    # Print summaries for all models
    models_list = ['simple_cnn', 'sequential_cnn', 'resnet18', 'efficient_net']
    
    for model_name in models_list:
        try:
            print_model_summary(model_name)
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
