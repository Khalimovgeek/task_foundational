import os
# Suppress TensorFlow info and warning messages (3 = only errors are shown)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))

if gpus:
    for gpu in gpus:
        print("GPU Device:", gpu)
else:
    print("No GPU detected. TensorFlow will default to CPU.")
