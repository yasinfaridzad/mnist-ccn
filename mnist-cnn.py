import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import time
import os

# Create directory for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and prepare the MNIST dataset
def load_and_prepare_mnist():
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Reshape to be [samples][width][height][channels]
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    
    # Normalize inputs from 0-255 to 0-1
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # One hot encode outputs
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # Number of classes
    num_classes = y_test.shape[1]
    
    return (X_train, y_train), (X_test, y_test), num_classes

# Plot sample images
def plot_sample_images(X_train, y_train):
    plt.figure(figsize=(10, 5))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f"Class: {np.argmax(y_train[i])}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('plots/sample_mnist_images.png')
    plt.show()

# Define the simple CNN model
def simple_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

# Define the large CNN model
def large_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(30, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(),
        Conv2D(15, (3, 3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(50, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

# Define the larger CNN model
def larger_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Dropout(0.1),
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=192, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2),
        Dropout(0.1),
        Conv2D(filters=192, kernel_size=5, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2, padding='same'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model

# Train model and plot results
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=200, model_name="model"):
    start_time = time.time()
    
    # Train the model
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test), 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=1)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    error = 100 - scores[1] * 100
    
    # Print results
    print(f"{model_name} - Training time: {training_time:.2f} seconds")
    print(f"{model_name} - Test accuracy: {scores[1]:.4f}")
    print(f"{model_name} - Error: {error:.2f}%")
    
    # Save the model
    model.save(f'models/{model_name}.h5')
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_performance.png')
    plt.show()
    
    return history, scores, training_time

# Compare models
def compare_models(models_results):
    # Prepare data for comparison
    model_names = list(models_results.keys())
    accuracies = [models_results[model]['accuracy'] for model in model_names]
    times = [models_results[model]['time'] for model in model_names]
    params = [models_results[model]['params'] for model in model_names]
    
    # Create comparison plots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    # Accuracy comparison
    ax[0].bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax[0].set_title('Test Accuracy Comparison')
    ax[0].set_ylabel('Accuracy')
    for i, v in enumerate(accuracies):
        ax[0].text(i, v-0.05, f"{v:.4f}", ha='center')
    
    # Training time comparison
    ax[1].bar(model_names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax[1].set_title('Training Time Comparison')
    ax[1].set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        ax[1].text(i, v+5, f"{v:.2f}s", ha='center')
    
    # Model size comparison
    ax[2].bar(model_names, params, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax[2].set_title('Model Parameters')
    ax[2].set_ylabel('Number of Parameters')
    for i, v in enumerate(params):
        ax[2].text(i, v+50000, f"{v:,}", ha='center')
    
    plt.tight_layout()
    plt.savefig('plots/models_comparison.png')
    plt.show()

def main():
    # Load and prepare data
    (X_train, y_train), (X_test, y_test), num_classes = load_and_prepare_mnist()
    input_shape = (28, 28, 1)
    
    # Plot sample images
    plot_sample_images(X_train, y_train)
    
    # Create models
    model_simple = simple_model(input_shape, num_classes)
    model_large = large_model(input_shape, num_classes)
    model_larger = larger_model(input_shape, num_classes)
    
    # Print model summaries
    print("Simple CNN Architecture:")
    model_simple.summary()
    
    print("\nLarge CNN Architecture:")
    model_large.summary()
    
    print("\nLarger CNN Architecture:")
    model_larger.summary()
    
    # Train and evaluate models
    print("\nTraining Simple CNN...")
    history_simple, scores_simple, time_simple = train_and_evaluate(
        model_simple, X_train, y_train, X_test, y_test, epochs=10, batch_size=200, model_name="simple_cnn"
    )
    
    print("\nTraining Large CNN...")
    history_large, scores_large, time_large = train_and_evaluate(
        model_large, X_train, y_train, X_test, y_test, epochs=10, batch_size=200, model_name="large_cnn"
    )
    
    print("\nTraining Larger CNN...")
    history_larger, scores_larger, time_larger = train_and_evaluate(
        model_larger, X_train, y_train, X_test, y_test, epochs=10, batch_size=100, model_name="larger_cnn"
    )
    
    # Collect results for comparison
    models_results = {
        "Simple CNN": {
            "accuracy": scores_simple[1],
            "time": time_simple,
            "params": model_simple.count_params()
        },
        "Large CNN": {
            "accuracy": scores_large[1],
            "time": time_large,
            "params": model_large.count_params()
        },
        "Larger CNN": {
            "accuracy": scores_larger[1],
            "time": time_larger,
            "params": model_larger.count_params()
        }
    }
    
    # Compare models
    compare_models(models_results)

if __name__ == "__main__":
    main()
