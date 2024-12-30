import tensorflow as tf
import numpy as np

def representative_dataset():
    for image in train_images[:100]:  # Use a subset of your training dataset
        yield [image[np.newaxis, ...].astype('float32')]

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0

# Load the saved model
converter = tf.lite.TFLiteConverter.from_saved_model('./models/cifar10_cnn_tf_model')

# Enable optimization and specify INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Ensure both input and output tensors are quantized to int8
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save the quantized model
with open('./models/cifar10_cnn_tflite_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantized model successfully converted to TensorFlow Lite format.")
