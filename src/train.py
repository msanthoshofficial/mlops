import mlflow
import mlflow.tensorflow
import os
from src.preprocess import get_train_val_generators
from src.model import create_model
import tensorflow as tf

def train_model():
    # Use local dataset
    base_dir = r"C:\dev\mlops\Dataset\PetImages"
    
    if not os.path.exists(base_dir):
        print(f"Error: Dataset not found at {base_dir}")
        return

    print(f"Training on data from: {base_dir}")
    train_gen, val_gen = get_train_val_generators(base_dir)

    # MLflow Tracking
    mlflow.set_experiment("cats_vs_dogs")
    
    with mlflow.start_run():
        # Parameters
        img_size = (224, 224)
        batch_size = 32
        epochs = 2
        learning_rate = 1e-4

        mlflow.log_param("img_size", img_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)

        # Create Model
        model = create_model(input_shape=img_size + (3,))
        
        # Train
        history = model.fit(
            train_gen,
            steps_per_epoch=train_gen.samples // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=val_gen.samples // batch_size
        )

        # Log Metrics
        mlflow.log_metric("loss", history.history['loss'][-1])
        mlflow.log_metric("accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_loss", history.history['val_loss'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])

        # Save Model
        model.save("model.h5")
        mlflow.log_artifact("model.h5")
        
        print("Training complete. Model saved to model.h5")

if __name__ == "__main__":
    train_model()
