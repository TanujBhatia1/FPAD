import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import load_model

def find_last_conv_layer(model):
    """Finds the last Conv2D layer in the model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

if __name__ == "__main__":
    # Update the model path to your actual model location
    model_path = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\best_kan_model.keras'

    # Load the model
    model = load_model(model_path)

    # Find and print the last Conv2D layer name
    last_conv_layer_name = find_last_conv_layer(model)
    if last_conv_layer_name:
        print(f"Last convolutional layer: {last_conv_layer_name}")
    else:
        print("No Conv2D layer found in the model.")
