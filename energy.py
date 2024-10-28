import psutil
import ast
import numpy as np
import matplotlib.pyplot as plt
import time





# Helper functions for CRC
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

# Function to measure energy consumption
def energy(func,cpu, *args):
    start_time = time.time()
    cpu_percent_start = psutil.cpu_percent(interval=None)
    
    func(*args)  # Run the encoding function
    
    end_time = time.time()
    cpu_percent_end = psutil.cpu_percent(interval=None)
    
    duration = end_time - start_time  # Time in seconds
    cpu_usage = (cpu_percent_start + cpu_percent_end) / 2  # Approximate average CPU usage during execution
    
    # Assume average CPU power consumption of 65W (you can adjust this based on your CPU)
    cpu_power = 65  # in watts
    cpu_usage=max(cpu_usage,cpu)
    
    energy_consumed = cpu_power * duration * (cpu_usage / 100)  # Energy in joules
    
    return energy_consumed

# Reading data from the file
def read_data_from_file(filename):
    print(f"Reading data from {filename}...")
    with open(filename, 'r') as file:
        data_list = ast.literal_eval(file.read().strip())
    print("Data read successfully.")
    return data_list

# Define test files and sizes
test_files = ['test_file_1.txt', 'test_file_2.txt', 'test_file_3.txt', 'test_file_4.txt']
generator = "100000100110000010001110110110111"
chunk_size = 64

# Method labels
methods = ['CRC', 'Hamming', 'Hamming Improved', 'Matrix Parity']
energy_consumptions = {method: [] for method in methods}  # To store energy data

#base CPU usage
cpu_percent_start = psutil.cpu_percent(interval=None)
cpu_percent_end = psutil.cpu_percent(interval=None)
cpu = (cpu_percent_start + cpu_percent_end) / 2



# Measure and record energy for each method for each file
for file in test_files:
    print(f"\nProcessing {file}...")
    data_bits = read_data_from_file(file)
    binary_data = ''.join(str(bit) for bit in data_bits)

    # Measure Energy for CRC
    crc_energy = energy(encode_entire_file_crc,cpu, binary_data, generator, chunk_size)
    energy_consumptions['CRC'].append(crc_energy)

    # Measure Energy for Hamming
    hamming_energy = energy(encode_entire_file_hamming,cpu, data_bits)
    energy_consumptions['Hamming'].append(hamming_energy)

    # Measure Energy for Improved Hamming
    hamming_improved_energy = energy(encode_entire_file_hamming_improved,cpu, data_bits)
    energy_consumptions['Hamming Improved'].append(hamming_improved_energy)

    # Measure Energy for Matrix with Parities
    matrix_parity_energy = energy(process_matrices,cpu, data_bits)
    energy_consumptions['Matrix Parity'].append(matrix_parity_energy)

# Plotting energy consumption comparison
x_labels = [f'File {i+1}' for i in range(len(test_files))]
x = np.arange(len(x_labels))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))

# Ensure all methods have the same number of entries
max_len = max(len(energy_consumptions[method]) for method in methods)
for method in methods:
    while len(energy_consumptions[method]) < max_len:
        energy_consumptions[method].append(0)  # Append 0 for missing data



# Debugging: Print energy consumptions to ensure data is correct
for method in methods:
    print(f"{method} energy consumptions: {energy_consumptions[method]}")

# Create bar plots
rects1 = ax.bar(x - 1.5 * width, energy_consumptions['CRC'], width, label='CRC')
rects2 = ax.bar(x - 0.5 * width, energy_consumptions['Hamming'], width, label='Hamming')
rects3 = ax.bar(x + 0.5 * width, energy_consumptions['Hamming Improved'], width, label='Hamming Improved')
rects4 = ax.bar(x + 1.5 * width, energy_consumptions['Matrix Parity'], width, label='Matrix Parity')

# Customize the plot
ax.set_xlabel('Test Files')
ax.set_ylabel('Energy Consumption (Joules)')
ax.set_title('Energy Consumption Comparison by Encoding Method')
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()

plt.tight_layout()
plt.show()