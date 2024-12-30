import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load and preprocess CIFAR-10 dataset
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Build the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1, validation_data=(x_test, y_test))
    return model

# Convert the model to TFLite format with quantization
def convert_to_tflite_quantized(model, x_train):
    # Define representative dataset for quantization
    def representative_dataset():
        for data in x_train[:100]:
            yield [np.expand_dims(data, axis=0).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quantized_model = converter.convert()

    with open("cifar10_cnn_tflite_model_quantized.tflite", "wb") as f:
        f.write(tflite_quantized_model)
    print("Quantized TFLite model saved.")

# Main function
def main():
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    model = create_model()
    model = train_model(model, x_train, y_train, x_test, y_test)
    convert_to_tflite_quantized(model, x_train)

if __name__ == "__main__":
    main()
