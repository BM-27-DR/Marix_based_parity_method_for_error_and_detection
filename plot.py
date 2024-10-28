import matplotlib.pyplot as plt

# Define overhead calculation functions from previous methods
def sec_50_overhead(data_bits):
    total_bits = len(data_bits) * 1.5
    redundant_bits = total_bits - len(data_bits)
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def hamming_overhead(data_bits):
    n = len(data_bits) // 4
    redundant_bits = n * 3
    total_bits = len(data_bits) + redundant_bits
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def turbo_code_overhead(data_bits, rate=1/2):
    total_bits = len(data_bits) / rate
    redundant_bits = total_bits - len(data_bits)
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def fu_8_4_sec_overhead(data_bits):
    n = len(data_bits) // 4
    redundant_bits = n * 4
    total_bits = len(data_bits) + redundant_bits
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def proposed_algo_overhead(p):
    overhead = (2 * p * 100) / (p * p + 2 * p)
    return overhead

# Data bits for 256-bit data
data_bits = [1] * 256  # 256 data bits

# Calculate overhead for each method
sec_ovh = sec_50_overhead(data_bits)
hamming_ovh = hamming_overhead(data_bits)
turbo_ovh = turbo_code_overhead(data_bits, rate=1/2)
fu_8_4_ovh = fu_8_4_sec_overhead(data_bits)

# Choose a value of p for the proposed algorithm (assume p = 16 for example)
proposed_ovh = proposed_algo_overhead(p=16)

# Overhead values
overheads = [sec_ovh, hamming_ovh, turbo_ovh, fu_8_4_ovh, proposed_ovh]

# Corresponding labels
labels = ['SEC (50%)', 'Hamming (7,4)', 'Turbo (1/2)', 'FU (8,4) SEC', 'Proposed Algorithm']

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(labels, overheads, color=['blue', 'orange', 'green', 'red', 'purple'])

# Add titles and labels
plt.title('Overhead Comparison for 256-bit Data')
plt.xlabel('Error-Correction Methods')
plt.ylabel('Overhead (%)')

# Display the overhead values on top of the bars
for i, v in enumerate(overheads):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

# Show the plot
plt.show()