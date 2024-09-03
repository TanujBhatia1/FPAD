import os
import tensorflow as tf
from tensorflow import keras 
from keras._tf_keras.keras.models  import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score
from convkan import ConvKAN  # Importing the predefined library

class EpochLogger(tf.keras.callbacks.Callback):
    """Custom callback to log the epoch number and metrics after each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}, "
              f"Val Loss = {logs['val_loss']:.4f}, Val Accuracy = {logs['val_accuracy']:.4f}")

def build_kan_model(input_shape):
    """Builds a Keras model for fingerprint classification."""
    model = Sequential([
        ConvKAN(in_channels=input_shape[2], out_channels=32, kernel_size=3, stride=1, padding=1, version=version),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_kan_model(train_dir, valid_dir, input_shape, batch_size=32, epochs=30):
    """Trains the Keras model using the specified training and validation directories."""
    train_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary'
    )

    # Calculate the steps per epoch
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = valid_generator.samples // batch_size

    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {valid_generator.samples}")

    model = build_kan_model(input_shape)
    
    # Save only the best model based on validation accuracy
    checkpoint_path = r'C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints/best_kan_model.keras'
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # Add the custom callback for logging epochs
    epoch_logger = EpochLogger()

    model.fit(train_generator, 
              validation_data=valid_generator, 
              epochs=epochs, 
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              callbacks=[checkpoint, epoch_logger])
    
    return model

def evaluate_kan_model(model, test_dir, input_shape, batch_size=32):
    """Evaluates the trained Keras model using the test directory."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    print(f"Test samples: {test_generator.samples}")

    y_true = test_generator.classes
    y_pred = model.predict(test_generator)
    y_pred = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    apcer = cm[1][0] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) != 0 else 0
    bpcer = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) != 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"APCER: {apcer:.4f}")
    print(f"BPCER: {bpcer:.4f}")

    return accuracy, apcer, bpcer

if __name__ == "__main__":
    input_shape = (100, 200, 3)  # Update as needed
    base_dir = r"C:\Users\tanuj\OneDrive\Desktop\ROI_Fingerprints\DatasetPartition"  # Base directory containing the dataset partitions

    # Paths to the training, validation, and test directories
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")
    test_dir = os.path.join(base_dir, "test")

    # Train the model using the training and validation datasets
    model = train_kan_model(train_dir, valid_dir, input_shape)

    # Load the best saved model for evaluation
    model = tf.keras.models.load_model('best_kan_model.keras')

    # Evaluate the model using the test dataset
    evaluate_kan_model(model, test_dir, input_shape)

    
