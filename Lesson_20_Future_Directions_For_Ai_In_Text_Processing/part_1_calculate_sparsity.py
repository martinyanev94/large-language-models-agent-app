import numpy as np

def calculate_sparsity(vector):
    non_zero_elements = np.count_nonzero(vector)
    total_elements = vector.size
    sparsity = total_elements - non_zero_elements
    sparsity_rate = sparsity / total_elements
    return sparsity_rate

# Example vector
vector = np.array([1, 0, 0, 2, 0, 3, 0, 0])
sparsity = calculate_sparsity(vector)
print(f"Sparsity rate: {sparsity:.2f}")
