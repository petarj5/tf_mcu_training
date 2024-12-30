
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

####### CONFIG VARIABLES ########
CONFIG_plotPartialDatasetImages = False
####### END CONFIG VARIABLES ########

# Config argument parser
parser = argparse.ArgumentParser(prog='python training_tf.py', usage='%(prog)s [options]')
parser.add_argument('-e', '--epochs', type=int, default=1, required=False,
                    help='Number of training epochs', dest='number_of_epochs')
args = parser.parse_args()

number_of_epochs = args.number_of_epochs

# Load data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Plot dataset
if(CONFIG_plotPartialDatasetImages):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()

# Build model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=number_of_epochs, 
                    validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)

# Save the trained model
model.save('./models/cifar10_cnn_tf_model.h5')
tf.saved_model.save(model, './models/cifar10_cnn_tf_model')