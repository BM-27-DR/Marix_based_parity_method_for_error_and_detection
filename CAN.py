import time
import ast
import numpy as np
import matplotlib.pyplot as plt

# The four methods: CRC, Hamming (Original), Hamming (Improved), and Matrix with Parities

def xor(a, b):
    result = []
    for i in range(1, len(b)):
        result.append('0' if a[i] == b[i] else '1')
    return ''.join(result)

def mod2div(dividend, divisor):
    pick = len(divisor)
    tmp = dividend[:pick]
    while pick < len(dividend):
        if tmp[0] == '1':
            tmp = xor(divisor, tmp) + dividend[pick]
        else:
            tmp = xor('0' * pick, tmp) + dividend[pick]
        pick += 1
    if tmp[0] == '1':
        tmp = xor(divisor, tmp)
    else:
        tmp = xor('0' * pick, tmp)
    return tmp

def encode_data(data, generator):
    l_gen = len(generator)
    appended_data = data + '0' * (l_gen - 1)
    remainder = mod2div(appended_data, generator)
    codeword = data + remainder
    return codeword

def encode_entire_file_crc(data, generator, chunk_size):
    encoded_result = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = chunk.ljust(chunk_size, '0')  # Pad with zeros
        encoded_chunk = encode_data(chunk, generator)
        encoded_result.append(encoded_chunk)
    return ''.join(encoded_result)

def calculate_parity_bits(data_bits):
    p1 = data_bits[0] ^ data_bits[1] ^ data_bits[3]
    p2 = data_bits[0] ^ data_bits[2] ^ data_bits[3]
    p3 = data_bits[1] ^ data_bits[2] ^ data_bits[3]
    return [p1, p2, p3]

def encode_hamming(data_bits):
    # Calculate parity bits using the calculate_parity_bits function
    p1, p2, p3 = calculate_parity_bits(data_bits)
    
    # Initialize the Hamming code with placeholders (0 for parity bit positions)
    hamming_code = [0] * 7  # Total length = 4 data bits + 3 parity bits
    
    # Insert the data bits into the Hamming code (excluding parity bit positions)
    data_index = 0
    for i in range(7):
        # Skip parity positions: 1 (2^0), 2 (2^1), 4 (2^2)
        if i+1 not in [pow(2, n) for n in range(3)]:
            hamming_code[i] = data_bits[data_index]
            data_index += 1
    
    # Place the calculated parity bits in their respective positions
    hamming_code[int(pow(2, 0)) - 1] = p1  # p1 at position 1 (2^0 = 1)
    hamming_code[int(pow(2, 1)) - 1] = p2  # p2 at position 2 (2^1 = 2)
    hamming_code[int(pow(2, 2)) - 1] = p3  # p3 at position 4 (2^2 = 4)
    
    return hamming_code


def encode_entire_file_hamming(data_bits):
    encoded_hamming_codes = []
    for i in range(0, len(data_bits), 4):
        chunk = data_bits[i:i + 4]
        if len(chunk) < 4:
            chunk = chunk + [0] * (4 - len(chunk))  # Pad with zeros
        encoded = encode_hamming(chunk)
        encoded_hamming_codes.append(encoded)
    return encoded_hamming_codes

def encode_hamming_improved(data_bits):
    p1, p2, p3 = calculate_parity_bits(data_bits)
    hamming_code = [data_bits[0], data_bits[1], data_bits[2], data_bits[3], p1, p2, p3]
    return hamming_code

def encode_entire_file_hamming_improved(data_bits):
    encoded_hamming_codes = []
    for i in range(0, len(data_bits), 4):
        chunk = data_bits[i:i + 4]
        if len(chunk) < 4:
            chunk = chunk + [0] * (4 - len(chunk))  # Pad with zeros
        encoded = encode_hamming_improved(chunk)
        encoded_hamming_codes.append(encoded)
    return encoded_hamming_codes

def create_matrix(data_bits, size=64):
    if len(data_bits) < size * size:
        # If there is not enough data, pad with zeros
        data_bits += [0] * (size * size - len(data_bits))
    matrix = np.array(data_bits[:size * size]).reshape(size, size)
    return matrix

def calculate_parities(matrix):
    size = matrix.shape[0]
    row_parities = np.sum(matrix, axis=1) % 2
    col_parities = np.sum(matrix, axis=0) % 2
    matrix_with_row_parity = np.vstack([matrix, row_parities])
    matrix_with_parity = np.hstack([matrix_with_row_parity, np.append(col_parities, 0).reshape(-1, 1)])
    return matrix_with_parity

def process_matrices(data_bits, size=64):
    matrices_with_parity = []
    for i in range(0, len(data_bits), size * size):
        chunk = data_bits[i:i + size * size]
        matrix = create_matrix(chunk, size)
        matrix_with_parity = calculate_parities(matrix)
        matrices_with_parity.append(matrix_with_parity)
    return matrices_with_parity

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data_list = ast.literal_eval(file.read().strip())
        return data_list

# Define test files and sizes
test_files = ['test_file_1.txt', 'test_file_2.txt', 'test_file_3.txt', 'test_file_4.txt']
generator = "100000100110000010001110110110111"
chunk_size = 64

# Method labels
methods = ['CRC', 'Hamming', 'Hamming Improved', 'Matrix Parity']
times = {method: [] for method in methods}

# Measure and record time taken by each method for each file
for file in test_files:
    print(f"\nProcessing {file}...")
    data_bits = read_data_from_file(file)
    binary_data = ''.join(str(bit) for bit in data_bits)  # Convert list to binary string for CRC method

    # Measure CRC
    start = time.time()
    encode_entire_file_crc(binary_data, generator, chunk_size)
    crc_time = time.time() - start
    times['CRC'].append(crc_time)

    # Measure Hamming
    start = time.time()
    encode_entire_file_hamming(data_bits)
    hamming_time = time.time() - start
    times['Hamming'].append(hamming_time)

    # Measure Improved Hamming
    start = time.time()
    encode_entire_file_hamming_improved(data_bits)
    hamming_improved_time = time.time() - start
    times['Hamming Improved'].append(hamming_improved_time)

    # Measure Matrix with Parities
    start = time.time()
    process_matrices(data_bits)
    matrix_parity_time = time.time() - start
    times['Matrix Parity'].append(matrix_parity_time)

# Plotting
x_labels = [f'File {i+1}' for i in range(len(test_files))]
x = np.arange(len(x_labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

# Ensure all methods have the same number of entries
max_len = max(len(times[method]) for method in methods)
for method in methods:
    while len(times[method]) < max_len:
        times[method].append(0)  # Append 0 for missing data



# Debugging: Print times to ensure data is correct
for method in methods:
    print(f"{method} times: {times[method]}")

# Create bar plots
rects1 = ax.bar(x - 1.5 * width, times['CRC'], width, label='CRC')
rects2 = ax.bar(x - 0.5 * width, times['Hamming'], width, label='Hamming')
rects3 = ax.bar(x + 0.5 * width, times['Hamming Improved'], width, label='Hamming Improved')
rects4 = ax.bar(x + 1.5 * width, times['Matrix Parity'], width, label='Matrix Parity')

# Customize the plot
ax.set_xlabel('Test Files')
ax.set_ylabel('Time (seconds)')
ax.set_title('Time taken by each method for different input sizes')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

plt.tight_layout()
plt.show()