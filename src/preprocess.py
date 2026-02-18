import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_train_val_generators(base_dir, img_size=(224, 224), batch_size=32):
    # Using validation_split since data is not pre-split
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 
    )

    # For validation, we don't want augmentation, but we need rescale
    # Note: When using validation_split, we must use the SAME instance of ImageDataGenerator 
    # or create a new one with same split validation argument if we want clean validation data?
    # Actually, standard practice with validation_split in flow_from_directory is to use the same generator or another with validation_split set.
    # However, usually we want augmentation on train but not on val.
    # To achieve this with `flow_from_directory` and `subset`:
    # We can use two generators.
    
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = test_datagen.flow_from_directory(
        base_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

if __name__ == "__main__":
    base_dir = r"C:\dev\mlops\Dataset\PetImages"
    print(f"Using local dataset at: {base_dir}")
