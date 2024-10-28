import numpy as np

# Parity Calculation Functions
def calculate_row_parity(matrix):
    return np.bitwise_xor.reduce(matrix, axis=1)

def calculate_column_parity(matrix):
    return np.bitwise_xor.reduce(matrix, axis=0)

def calculate_upper_left_diagonal_parities(matrix):
    p = matrix.shape[0]
    diagonals = []
    for offset in range(p):
        diagonals.append(np.bitwise_xor.reduce(np.diag(np.fliplr(matrix), k=offset)))
    diagonals.reverse()  # Reverse the list of diagonals
    return diagonals

def calculate_upper_right_diagonal_parities(matrix):
    p = matrix.shape[0]
    diagonals = []
    for offset in range(p):
        diagonals.append(np.bitwise_xor.reduce(np.diag(matrix, k=offset)))
    diagonals.reverse()  # Reverse the list of diagonals
    return diagonals

# Encoding Function for Double Error
def encode_double_error(message_bits, p):
    message_matrix = np.array(message_bits[:p*p]).reshape(p, p)
    row_parity = calculate_row_parity(message_matrix)
    col_parity = calculate_column_parity(message_matrix)
    upper_left_diagonal_parities = calculate_upper_left_diagonal_parities(message_matrix)
    upper_right_diagonal_parities = calculate_upper_right_diagonal_parities(message_matrix)
    encoded_message = np.concatenate((message_matrix.flatten(), row_parity, col_parity, upper_left_diagonal_parities, upper_right_diagonal_parities))
    return encoded_message, row_parity, col_parity, upper_left_diagonal_parities, upper_right_diagonal_parities

# Decoding Function for Double Error
def decode_double_error(encoded_message, p):
    message_size = p * p
    message_bits = encoded_message[:message_size]
    row_parity_received = encoded_message[message_size:message_size + p]
    col_parity_received = encoded_message[message_size + p:message_size + 2 * p]
    upper_left_diagonal_parities_received = encoded_message[message_size + 2 * p:message_size + 3 * p]
    upper_right_diagonal_parities_received = encoded_message[message_size + 3 * p:message_size + 4 * p]
    message_matrix = message_bits.reshape(p, p)
    row_parity_calculated = calculate_row_parity(message_matrix)
    col_parity_calculated = calculate_column_parity(message_matrix)
    upper_left_diagonal_parities_calculated = calculate_upper_left_diagonal_parities(message_matrix)
    upper_right_diagonal_parities_calculated = calculate_upper_right_diagonal_parities(message_matrix)
    
    r, c = [], []
    for i in range(p):
        if row_parity_calculated[i] != row_parity_received[i]:
            r.append(i)
        if col_parity_calculated[i] != col_parity_received[i]:
            c.append(i)
    
    upper_left_diagonal_mismatches = [i for i in range(len(upper_left_diagonal_parities_received)) if upper_left_diagonal_parities_received[i] != upper_left_diagonal_parities_calculated[i]]
    upper_right_diagonal_mismatches = [i for i in range(len(upper_right_diagonal_parities_received)) if upper_right_diagonal_parities_received[i] != upper_right_diagonal_parities_calculated[i]]
    
    error_positions = []
    
    # Case 1: Triple errors in the same row
    if len(r) == 1 and len(c) == 3:
        for col in c:
            error_positions.append((r[0], col))

    # Case 2: Triple errors in the same column
    elif len(c) == 1 and len(r) == 3:
        for row in r:
            error_positions.append((row, c[0]))
    
    # Case 3: Triple errors scattered
    elif len(r) == 3 and len(c) == 3:
        for row, col in zip(r, c):
            error_positions.append((row, col))
    
    elif len(c) == 2 and len(upper_left_diagonal_mismatches) == 2:
        # Double error in column and left upper diagonal
        error_positions = [(r[0] if r else 1, c[0]), (r[1] if r else 1, c[1])]
    elif len(c) == 1 and len(upper_left_diagonal_mismatches) == 1:
        # Single error in column and left upper diagonal
        error_positions = [(r[0] if r else 1, c[0])]
    elif len(c) == 2 and len(upper_right_diagonal_mismatches) == 2:
        # Double error in column and right upper diagonal
        error_positions = [(r[0] if r else 1, c[0]), (r[1] if r else 1, c[1])]
    elif len(c) == 1 and len(upper_right_diagonal_mismatches) == 1:
        # Single error in column and right upper diagonal
        error_positions = [(r[0] if r else 1, c[0])]
    elif len(c) == 2 and len(upper_left_diagonal_mismatches) == 1:
        # One error in message and one error in parity block
        error_positions = [(r[0] if r else 1, c[0]), (r[1] if r else 1, c[1])]
    elif len(c) == 1 and len(upper_left_diagonal_mismatches) == 2:
        # One error in message and one error in parity block
        error_positions = [(r[0] if r else 1, c[0])]
    elif len(c) == 2 and len(upper_right_diagonal_mismatches) == 1:
        # One error in message and one error in parity block
        error_positions = [(r[0] if r else 1, c[0]), (r[1] if r else 1, c[1])]
    elif len(c) == 1 and len(upper_right_diagonal_mismatches) == 2:
        # One error in message and one error in parity block
        error_positions = [(r[0] if r else 1, c[0])]
    
    # Correct the errors by flipping the bits at the identified error positions
    for row, col in error_positions:
        message_matrix[row, col] ^= 1  # Correct the error by flipping the bit
    
    return message_matrix.flatten(), error_positions

# Introduce Errors (Single, Double, and Triple)
def introduce_error(encoded_message, p, error_type, positions):
    if error_type == 'single':
        row, col = positions
        encoded_message[row * p + col] ^= 1
    elif error_type == 'double':
        (row1, col1), (row2, col2) = positions
        encoded_message[row1 * p + col1] ^= 1
        encoded_message[row2 * p + col2] ^= 1
    elif error_type == 'triple':
        for row, col in positions:
            encoded_message[row * p + col] ^= 1
    return encoded_message

# Highlight Errors in the Encoded Message
def highlight_errors(encoded_message, p, positions):
    highlighted_message = encoded_message.tolist()  # Convert to list for easier manipulation
    for row, col in positions:
        index = row * p + col
        highlighted_message[index] = f"*{highlighted_message[index]}*"
    return highlighted_message

# Generate Random 2D Array
def generate_random_2d_array(bits):
    size = int(np.sqrt(bits))
    return np.random.randint(2, size=(size, size))

# Write content to file
def write_to_file(filename, content):
    with open(filename, 'w') as file:
        file.write(content)

# Main Execution
if __name__ == "__main__":
    bits = int(input("Enter the number of bits: "))
    random_2d_array = generate_random_2d_array(bits)
    p = random_2d_array.shape[0]
    message_bits = random_2d_array.flatten().tolist()

    # Encode message
    encoded_message, row_parity, col_parity, upper_left_diagonal_parities, upper_right_diagonal_parities = encode_double_error(message_bits, p)

    # Introduce triple errors and decode
    encoded_message_with_triple_error = introduce_error(encoded_message.copy(), p, 'triple', [(2, 1), (3, 2), (4, 3)])
    highlighted_triple_error_message = highlight_errors(encoded_message_with_triple_error, p, [(2, 1), (3, 2), (4, 3)])
    decoded_message_triple_error, error_positions_triple = decode_double_error(encoded_message_with_triple_error, p)

    # Introduce double errors and decode
    encoded_message_with_double_error = introduce_error(encoded_message.copy(), p, 'double', [(2, 1), (5, 3)])
    highlighted_double_error_message = highlight_errors(encoded_message_with_double_error, p, [(2, 1), (5, 3)])
    decoded_message_double_error, error_positions_double = decode_double_error(encoded_message_with_double_error, p)

    # Introduce single error and decode
    encoded_message_with_single_error = introduce_error(encoded_message.copy(), p, 'single', (2, 3))
    highlighted_single_error_message = highlight_errors(encoded_message_with_single_error, p, [(2, 3)])
    decoded_message_single_error, error_positions_single = decode_double_error(encoded_message_with_single_error, p)

    # Prepare content for the file
    content = (
        f"Random 2D array:\n{random_2d_array}\n\n"
        f"Encoded message: {encoded_message.tolist()}\n"
        f"Row parity: {row_parity.tolist()}\n"
        f"Column parity: {col_parity.tolist()}\n"
        f"Upper left diagonal parities: {upper_left_diagonal_parities}\n"
        f"Upper right diagonal parities: {upper_right_diagonal_parities}\n\n"
        f"Encoded message with triple errors: {highlighted_triple_error_message}\n"
        f"Decoded message (triple error): {decoded_message_triple_error.tolist()}\n"
        f"Error positions corrected (triple error): {error_positions_triple}\n\n"
        f"Encoded message with double errors: {highlighted_double_error_message}\n"
        f"Decoded message (double error): {decoded_message_double_error.tolist()}\n"
        f"Error positions corrected (double error): {error_positions_double}\n\n"
        f"Encoded message with single error: {highlighted_single_error_message}\n"
        f"Decoded message (single error): {decoded_message_single_error.tolist()}\n"
        f"Error positions corrected (single error): {error_positions_single}\n"
    )

    # Write content to file
    write_to_file("output.txt", content)