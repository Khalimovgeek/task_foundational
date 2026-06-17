import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress clean-up warnings
import tensorflow as tf

# ==========================================
# 1. CREATE TENSORS
# ==========================================
print("--- 1. Creating Tensors ---")

# Create a 1D tensor (vector) with 12 elements
tensor_1d = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
print("Original 1D Tensor:")
print(tensor_1d.numpy())
print("Shape:", tensor_1d.shape)


# ==========================================
# 2. RESHAPE TENSORS
# ==========================================
print("\n--- 2. Reshaping Tensors ---")

# Reshape the 1D tensor (12 elements) into a 2D matrix of shape (3 rows, 4 columns)
tensor_2d_a = tf.reshape(tensor_1d, [3, 4])
print("Reshaped to (3, 4) Tensor A:")
print(tensor_2d_a.numpy())

# You can also use -1 to let TensorFlow calculate the dimension automatically
# Here we reshape to 2 columns, and let TensorFlow figure out the rows (which will be 6)
tensor_auto = tf.reshape(tensor_1d, [-1, 2])
print("\nReshaped to (6, 2) using auto-dimension:")
print(tensor_auto.numpy())


# ==========================================
# 3. ELEMENTWISE OPERATIONS
# ==========================================
print("\n--- 3. Elementwise Operations ---")

# Create a second 2D tensor of the exact same shape (3, 4) to perform operations with
tensor_2d_b = tf.constant([
    [2.0, 2.0, 2.0, 2.0],
    [3.0, 3.0, 3.0, 3.0],
    [4.0, 4.0, 4.0, 4.0]
])
print("Tensor B (3, 4):")
print(tensor_2d_b.numpy())

# Elementwise Addition
# You can use standard mathematical operators (+, *, -, /) directly on tensors
add_result = tensor_2d_a + tensor_2d_b
print("\nElementwise Addition (Tensor A + Tensor B):")
print(add_result.numpy())

# Elementwise Multiplication (Hadamard Product)
mul_result = tensor_2d_a * tensor_2d_b
print("\nElementwise Multiplication (Tensor A * Tensor B):")
print(mul_result.numpy())

# Elementwise Division
div_result = tensor_2d_a / tensor_2d_b
print("\nElementwise Division (Tensor A / Tensor B):")
print(div_result.numpy())
