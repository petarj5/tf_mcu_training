# **Activating Python Virtual Environment**

If you're using a Python virtual environment to manage dependencies, follow these steps to activate, verify, and deactivate the environment based on your shell type.

## **Step 1: Create And Locate Your Virtual Environment**
Navigate to your project directory and run the following command to create the virtual environment:

```bash
python -m venv .tf_virt_env
```

This will create a directory named .tf_virt_env containing all the necessary files for your isolated Python environment.
Ensure that your virtual environment is located in the `.tf_virt_env` directory (or any directory you’ve named). This directory should contain the `bin/` folder, which includes activation scripts tailored to different shell types.

## **Step 2: Activate the Environment**

### For **Bash** or **Zsh**:
To activate the virtual environment, run:
```bash
source .tf_virt_env/bin/activate
```

### For **Fish** shell:
Run the following command:
```fish
source .tf_virt_env/bin/activate.fish
```

### For **Csh** or **Tcsh**:
Activate with:
```csh
source .tf_virt_env/bin/activate.csh
```

### For **PowerShell** (Windows):
If you're on Windows, use PowerShell:
```powershell
.tf_virt_env\Scripts\Activate.ps1
```

## **Step 3: Verify Activation**
Once activated, you should see the environment's name in your shell prompt, indicating that the environment is active. For example:
```
(.tf_virt_env) $
```
This means your virtual environment is successfully activated.

## **Step 4: Install Dependencies**
To install necessary dependencies like TensorFlow, Matplotlib, and other packages listed in the `requirements.txt` file, run:
```bash
pip install -r requirements.txt
```
This will install all the dependencies defined in the `requirements.txt` file.

### **Note on TensorFlow Compatibility**:
TensorFlow is **not stable with Python 3.12** as of the latest versions. To avoid unexpected behavior, it is recommended to use **Python 3.11** for TensorFlow projects.

---

## **Step 5: Deactivate the Environment**
When you're done working in the virtual environment, you can deactivate it by running:
```bash
deactivate
```
This will return you to the global Python environment.

---

# **Additional Notes**
- Always ensure that your virtual environment is active before installing or running any dependencies specific to your project.
- If you encounter issues with TensorFlow and Python 3.12, consider downgrading to Python 3.11 for better stability and compatibility.
- You must install older version of TensorFlow. Neutron-converter script is written for TensorFlow 2.10, but is also working with TensorFlow 2.12 which is in requirements.txt.
- Neutron-converter script provided in "cifar10_nn/" has to be used with MCU SDK 2.13.1. Other versions of neutron-converter are provided in eiQ-Toolkit (this one is taken from eiQ-Toolkit 1.13.1).