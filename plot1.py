import matplotlib.pyplot as plt

# Define corrected overhead calculation functions for double-bit error correction methods

def bch_overhead(data_bits, target_overhead=53.3):
    """Calculate overhead for BCH Code to match a specific target overhead."""
    total_bits = len(data_bits) / (1 - target_overhead / 100)  # Back-calculate total bits
    redundant_bits = total_bits - len(data_bits)
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def turbo_code_overhead(data_bits, rate=1/2):
    """Calculate overhead for Turbo Code with a given rate."""
    total_bits = len(data_bits) / rate  # Number of output bits after encoding
    redundant_bits = total_bits - len(data_bits)  # Redundant bits added
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def rs_code_overhead(data_bits, parity_symbols=16):
    """Calculate overhead for Reed-Solomon Code."""
    # RS code adds parity symbols for error correction
    total_bits = len(data_bits) + parity_symbols * 8  # Assuming 8 bits per symbol
    redundant_bits = total_bits - len(data_bits)
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def deccs_16_8_overhead(data_bits):
    """Calculate overhead for DECCS(16,8) DEC."""
    # Adds 8 parity bits for every 8 data bits
    redundant_bits = len(data_bits)  # Since it's 50% overhead (8 data + 8 parity)
    total_bits = len(data_bits) + redundant_bits
    overhead = (redundant_bits / total_bits) * 100
    return overhead

def dec_bch_overhead(data_bits, target_overhead=53.3):
    """Calculate overhead for DEC BCH Code to match a specific target overhead."""
    return bch_overhead(data_bits, target_overhead=target_overhead)

def proposed_algo_overhead(p):
    """Calculate overhead for the proposed algorithm with diagonal parity bits."""
    overhead = (4 * p * 100) / (p * p + 4 * p)
    return overhead

# Data bits for 256-bit data
data_bits = [1] * 256  # 256 data bits

# Calculate overhead for each method
bch_ovh = bch_overhead(data_bits, target_overhead=53.3)
turbo_ovh = turbo_code_overhead(data_bits, rate=1/2)
rs_ovh = rs_code_overhead(data_bits, parity_symbols=16)
deccs_ovh = deccs_16_8_overhead(data_bits)
dec_bch_ovh = dec_bch_overhead(data_bits, target_overhead=53.3)

# Proposed Algorithm: Assuming p=16 (similar to previous case)
proposed_ovh = proposed_algo_overhead(p=16)

# Overhead values
overheads = [bch_ovh, turbo_ovh, rs_ovh, deccs_ovh, dec_bch_ovh, proposed_ovh]

# Corresponding labels
labels = ['BCH', 'Turbo (1/2)', 'RS Code', 'DECCS(16,8) DEC', 'DEC BCH', 'Proposed Algorithm']

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(labels, overheads, color=['blue', 'orange', 'green', 'red', 'cyan', 'purple'])

# Add titles and labels
plt.title('Overhead Comparison for Double Error Correction Methods (256-bit Data)')
plt.xlabel('Error-Correction Methods')
plt.ylabel('Overhead (%)')

# Display the overhead values on top of the bars
for i, v in enumerate(overheads):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center')

# Show the plot
plt.show()