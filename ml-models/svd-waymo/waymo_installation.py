import subprocess
import sys
import tensorflow as tf

# Get major and minor TF version
tf_version = tf.__version__
major_minor = '.'.join(tf_version.split('.')[:2])
print(f"Detected TensorFlow version: {tf_version}")

# Construct the Waymo package name
waymo_package = f"waymo-open-dataset-tf-{major_minor.replace('.', '-')}"
print(f"Installing package: {waymo_package}")

# Install the package via pip
subprocess.check_call([sys.executable, "-m", "pip", "install", waymo_package])