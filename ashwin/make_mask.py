
import numpy as np

def create_array_with_zeros(n, k):
    # Create an array of ones
    arr = np.ones((n, n))

    # Calculate the size of the square containing zeros
    zero_size = n // k

    # Generate random coordinates for the top-left corner of the zero square
    start_row = np.random.randint(0, n - zero_size + 1)
    start_col = np.random.randint(0, n - zero_size + 1)

    # Set the values in the zero square region to zeros
    arr[start_row:start_row + zero_size, start_col:start_col + zero_size] = 0

    return arr

size = 256
hole_ratio = 6
num_masks = 1200

output = []

for i in range(num_masks):
    my_mask = create_array_with_zeros(size, hole_ratio)
    output.append(my_mask)

output = np.array(output)

np.save(f"mask_file_small_{hole_ratio}",output)





