import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('models/cifar10_cnn_tf_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('cifar10_cnn_tflite_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format.")