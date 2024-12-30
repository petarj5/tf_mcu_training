# Training, Converting, and Preparing a Model for Execution on the MCXN947 MCU

This folder contains four Python scripts to help train, convert, and evaluate a machine learning model for deployment on the MCXN947 MCU. Below is a detailed explanation of each script and the process:

## Scripts Overview
1. **`training_tf.py`**  
   - Trains a Convolutional Neural Network (CNN) model using the CIFAR-10 dataset.
   - Saves the trained model in **HDF5 (.h5)** and **TensorFlow SavedModel (.pb)** formats.

2. **`converter_tf.py`**  
   - Converts the trained model into a TensorFlow Lite (.tflite) model.
   - Includes post-training quantization for optimized performance on microcontrollers.

3. **`evaluator_tf.py`**  
   - Evaluates the converted `.tflite` model for accuracy and performance.

4. **`full_suite.py`**  
   - Combines the training and conversion steps into a single script for convenience.


## Running Models on the MCXN947 MCU

### Step 1: Training the Model
To train the model, run the `training_tf.py` script as follows:
```bash
python training_tf.py
```

#### Output:
- The trained model is saved in the `models/` folder in two formats:
  - **HDF5 (.h5):** `cifar10_cnn_tf_model.h5`
  - **SavedModel (.pb):** TensorFlow SavedModel format.

#### Optional Flag:
You can specify the number of training epochs using the `-e` or `--epochs` flag:
```bash
python training_tf.py -e <number_of_epochs>
```
For example, to train the model for 10 epochs:
```bash
python training_tf.py -e 10
```

### Step 2: Converting the Model
To convert the trained model into a `.tflite` format, run the `converter_tf.py` script:
```bash
python converter_tf.py
```

#### Output:
- The `.tflite` model is saved in the `models/` folder as `cifar10_cnn_tflite_model_quantized.tflite`.

_Note_: The conversion process includes **quantization** to optimize the model for microcontroller deployment, reducing size and improving performance.

### Step 3: Generating the Header File
After generating the `.tflite` file, convert it into a C-compatible header file using the `xxd` utility:
```bash
xxd -i cifar10_cnn_tflite_model_quantized.tflite > model.h
```

#### Additional Instructions:
- Open `model.h` and make necessary adjustments (e.g., renaming variables if needed) to fit your project requirements. These changes will depend on your MCU's firmware integration process.

### Step 4: Running the Neutron-Converter
The `neutron-converter` tool prepares models for execution on the MCU's NPU (Neural Processing Unit). This tool is located in the `tools/` folder in the project's root directory.

#### Usage:
1. Navigate to the `tools/` directory inside parent directory:

2. Run the `neutron-converter` tool with the `.tflite` model as input:
   ```bash
   ./neutron-converter --input ../models/cifar10_cnn_tflite_model_quantized.tflite --output <filename>.tflite
   ```

#### Output:
- The tool generates an optimized model file specifically designed for the MCXN947's NPU.

## Combined Workflow (Optional)
If you prefer to train and convert the model in one step, use the `full_suite.py` script:
```bash
python full_suite.py
```

#### Output:
- The script trains the model, converts it to `.tflite`, and saves the resulting files in the `models/` folder.


## Summary of Generated Files
After following the steps, you will have the following files:
1. **HDF5 Model:** `models/cifar10_cnn_tf_model.h5`
2. **SavedModel:** `models/cifar10_cnn_tf_model/`
3. **Quantized TensorFlow Lite Model:** `models/cifar10_cnn_tflite_model_quantized.tflite`
4. **C Header File:** `model.h` (converted from the `.tflite` model).


## Notes
- Ensure the `xxd` utility is installed on your system before generating the header file.
- Follow the MCXN947 MCU documentation for integrating the model header file into your firmware project.
- Use the `evaluator_tf.py` script to test the accuracy of the `.tflite` model on your local machine before deploying to the MCU:
  ```bash
  python evaluator_tf.py
  ```