"""
Emotion Detection Model Training Script
This script trains a CNN model to detect emotions from facial images
Emotions: Happy, Sad, Angry, Surprise, Neutral, Fear, Disgust
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class EmotionDetectionModel:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build CNN architecture for emotion detection"""
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train_with_data_directory(self, train_dir, val_dir, epochs=50, batch_size=64):
        """
        Train the model using data from directories
        Directory structure should be:
        train_dir/
            angry/
            disgust/
            fear/
            happy/
            neutral/
            sad/
            surprise/
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=(48, 48),
            batch_size=batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            shuffle=False
        )
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'emotion_guardian_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def train_with_sample_data(self, epochs=30):
        """
        Train with synthetic/sample data for demonstration
        Use this if you don't have the FER2013 dataset
        """
        print("Generating sample training data...")
        
        # Generate random sample data (replace with real data)
        x_train = np.random.rand(1000, 48, 48, 1).astype('float32')
        y_train = keras.utils.to_categorical(np.random.randint(0, 7, 1000), 7)
        
        x_val = np.random.rand(200, 48, 48, 1).astype('float32')
        y_val = keras.utils.to_categorical(np.random.randint(0, 7, 200), 7)
        
        # Callbacks
        checkpoint = ModelCheckpoint(
            'emotion_guardian_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[checkpoint, early_stop],
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath='emotion_guardian_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def get_model_summary(self):
        """Print model architecture summary"""
        return self.model.summary()


def main():
    """Main training function"""
    print("=" * 60)
    print("EMOTION DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Initialize model
    emotion_model = EmotionDetectionModel()
    
    # Build and compile
    print("\nBuilding model architecture...")
    emotion_model.build_model()
    emotion_model.compile_model(learning_rate=0.001)
    
    print("\nModel Summary:")
    emotion_model.get_model_summary()
    
    # Choose training method
    print("\n" + "=" * 60)
    print("TRAINING OPTIONS:")
    print("1. Train with FER2013 dataset (requires data directory)")
    print("2. Train with sample data (for demonstration)")
    print("=" * 60)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        train_dir = input("Enter training data directory path: ").strip()
        val_dir = input("Enter validation data directory path: ").strip()
        
        if os.path.exists(train_dir) and os.path.exists(val_dir):
            print("\nStarting training with real data...")
            history = emotion_model.train_with_data_directory(
                train_dir, 
                val_dir, 
                epochs=50, 
                batch_size=64
            )
        else:
            print("Error: Directory not found. Using sample data instead...")
            history = emotion_model.train_with_sample_data(epochs=30)
    else:
        print("\nStarting training with sample data...")
        history = emotion_model.train_with_sample_data(epochs=30)
    
    # Save model
    print("\nSaving model...")
    emotion_model.save_model('emotion_guardian_model.h5')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("Model saved as: emotion_guardian_model.h5")
    print("=" * 60)


if __name__ == "__main__":
    main()