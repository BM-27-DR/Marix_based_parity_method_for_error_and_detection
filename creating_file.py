import random

def generate_random_binary_list(size):
    return [0 if random.randint(1, 100) % 2 == 0 else 1 for _ in range(size)]

def write_to_file(filename, data_list):
    with open(filename, 'w') as file:
        file.write(str(data_list))

# Define sizes for each file
sizes = {
    'test_file_1.txt': 8196 * 16,
    'test_file_2.txt': 8196 * 32,
    'test_file_3.txt': 8196 * 64,
    'test_file_4.txt': 8196 * 128  # Adjusting the fourth file to be consistent with previous sizes
}

# Generate files
for filename, size in sizes.items():
    binary_list = generate_random_binary_list(size)
    write_to_file(filename, binary_list)

print("Files created successfully.")
