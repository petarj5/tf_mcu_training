import tensorflow as tf
import numpy as np

# Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Load the quantized .tflite model
interpreter = tf.lite.Interpreter(model_path='./models/cifar10_cnn_tflite_model_quantized.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare test data (ensure it matches the input format of the model)
test_images_int8 = (test_images * 255).astype(np.int8)

# Evaluate the model
correct_predictions = 0
for i in range(len(test_images_int8)):
    input_data = np.expand_dims(test_images_int8[i], axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    correct_predictions += (prediction == test_labels[i][0])

accuracy = correct_predictions / len(test_images_int8)
print(f"Evaluation Accuracy of the Quantized Model: {accuracy:.4f}")
