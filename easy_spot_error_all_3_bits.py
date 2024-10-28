import numpy as np
from PIL import Image

def calculate_row_parity(matrix):
    return np.bitwise_xor.reduce(matrix, axis=1)

def calculate_column_parity(matrix):
    return np.bitwise_xor.reduce(matrix, axis=0)

def calculate_upper_left_diagonal_parities(matrix):
    p = matrix.shape[0]
    diagonals = []
    for offset in range(-p + 1, p):  # Iterate through all diagonals
        diagonals.append(np.bitwise_xor.reduce(np.diag(np.fliplr(matrix), k=offset), axis=None))
    return diagonals

def calculate_upper_right_diagonal_parities(matrix):
    p = matrix.shape[0]
    diagonals = []
    for offset in range(-p + 1, p):  # Iterate through all diagonals
        diagonals.append(np.bitwise_xor.reduce(np.diag(matrix, k=offset), axis=None))
    return diagonals

def encode_double_error(message_bits, p):
    message_matrix = np.array(message_bits[:p * p]).reshape(p, p)
    row_parity = calculate_row_parity(message_matrix)
    col_parity = calculate_column_parity(message_matrix)
    upper_left_diagonal_parities = calculate_upper_left_diagonal_parities(message_matrix)
    upper_right_diagonal_parities = calculate_upper_right_diagonal_parities(message_matrix)
    
    encoded_message = np.concatenate((message_matrix.flatten(), row_parity, col_parity, 
                                       upper_left_diagonal_parities, upper_right_diagonal_parities))
    return encoded_message

def decode_double_error(encoded_message, p):
    message_size = p * p
    message_bits = encoded_message[:message_size].reshape(p, p)
    
    # Calculate received parities
    row_parity_received = encoded_message[message_size:message_size + p]
    col_parity_received = encoded_message[message_size + p:message_size + 2 * p]
    upper_left_diagonal_parities_received = encoded_message[message_size + 2 * p:message_size + 3 * p]
    upper_right_diagonal_parities_received = encoded_message[message_size + 3 * p:message_size + 4 * p]

    # Calculate calculated parities
    row_parity_calculated = calculate_row_parity(message_bits)
    col_parity_calculated = calculate_column_parity(message_bits)
    upper_left_diagonal_parities_calculated = calculate_upper_left_diagonal_parities(message_bits)
    upper_right_diagonal_parities_calculated = calculate_upper_right_diagonal_parities(message_bits)

    # Identify rows and columns with mismatched parity
    error_rows = [i for i in range(p) if row_parity_calculated[i] != row_parity_received[i]]
    error_cols = [i for i in range(p) if col_parity_calculated[i] != col_parity_received[i]]

    # Initialize list for error positions
    error_positions = []

    # Detect errors based on row and column mismatches
    if len(error_rows) == 1 and len(error_cols) == 1:
        # Single error detected
        error_positions.append((error_rows[0], error_cols[0]))
    elif len(error_rows) == 2 and len(error_cols) == 2:
        # Double errors detected in different rows and columns
        error_positions.append((error_rows[0], error_cols[0]))
        error_positions.append((error_rows[1], error_cols[1]))
    elif len(error_rows) == 1 and len(error_cols) == 2:
        # Two errors in the same row
        row = error_rows[0]
        for col in error_cols:
            if (row + col) < len(upper_left_diagonal_parities_received) and \
               upper_left_diagonal_parities_calculated[row + col] != upper_left_diagonal_parities_received[row + col]:
                error_positions.append((row, col))
    elif len(error_cols) == 1 and len(error_rows) == 2:
        # Two errors in the same column
        col = error_cols[0]
        for row in error_rows:
            if (row - col + (p - 1)) >= 0 and (row - col + (p - 1)) < len(upper_right_diagonal_parities_received) and \
               upper_right_diagonal_parities_calculated[row - col + (p - 1)] != upper_right_diagonal_parities_received[row - col + (p - 1)]:
                error_positions.append((row, col))
    elif len(error_rows) == 3 and len(error_cols) == 3:
        # Three errors detected in different rows and columns
        for i in range(3):
            error_positions.append((error_rows[i], error_cols[i]))

    # Correct errors
    for row, col in error_positions:
        message_bits[row, col] ^= 1  # Flip the erroneous bits

    return message_bits.flatten(), error_positions

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

def binary_matrix_to_image(binary_matrix, image_path):
    try:
        img = Image.fromarray(np.uint8(binary_matrix * 255), mode='L')  # Convert binary matrix to image
        img.save(image_path)
        print(f"Image saved at: {image_path}")
    except Exception as e:
        print(f"Error saving image at {image_path}: {e}")

def image_to_binary_matrix(image_path, p):
    try:
        image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = image.resize((p, p))  # Resize to p x p matrix
        binary_image = np.array(image) > 128  # Convert to binary (thresholding at 128)
        return binary_image.astype(int)
    except Exception as e:
        print(f"Error reading image from {image_path}: {e}")
        return None

def display_image(image_path):
    try:
        image = Image.open(image_path)
        image.show()  # Display the image
    except Exception as e:
        print(f"Error displaying image: {e}")

if __name__ == "__main__":
    image_path = input("Enter the path of the image file: ")
    p = int(input("Enter the size of the square matrix (e.g., 8 for 8x8): "))
    
    # Save and display the original image
    try:
        original_image = Image.open(image_path).convert('L')  # Convert to black and white
        original_image.save('original_image.png')
        
    except Exception as e:
        print(f"Error saving or displaying the original image: {e}")
    
    binary_matrix = image_to_binary_matrix(image_path, p)
    
    if binary_matrix is not None:  # Proceed only if the binary matrix is successfully created
        message_bits = binary_matrix.flatten().tolist()

        # Double error encoding
        encoded_message = encode_double_error(message_bits, p)

        # Display the encoded matrix
        encoded_matrix = encoded_message[:p * p].reshape(p, p)
        binary_matrix_to_image(encoded_matrix, 'encoded_image.png')
        display_image('encoded_image.png')  # Display encoded image

        # Introduce single error
        encoded_message_with_single_error = introduce_error(encoded_message.copy(), p, 'single', (2, 3))
        erroneous_single_matrix = encoded_message_with_single_error[:p*p].reshape(p, p)
        binary_matrix_to_image(erroneous_single_matrix, 'erroneous_single_image.png')
        
        # Decode and correct the single error
        decoded_message_single_error, error_positions_single = decode_double_error(encoded_message_with_single_error, p)
        corrected_single_matrix = decoded_message_single_error.reshape(p, p)
        binary_matrix_to_image(corrected_single_matrix, 'corrected_single_image.png')

        # Introduce double errors
        encoded_message_with_double_error = introduce_error(encoded_message.copy(), p, 'double', [(1, 2), (2, 4)])
        erroneous_double_matrix = encoded_message_with_double_error[:p*p].reshape(p, p)
        binary_matrix_to_image(erroneous_double_matrix, 'erroneous_double_image.png')
        
        # Decode and correct the double error
        decoded_message_double_error, error_positions_double = decode_double_error(encoded_message_with_double_error, p)
        corrected_double_matrix = decoded_message_double_error.reshape(p, p)
        binary_matrix_to_image(corrected_double_matrix, 'corrected_double_image.png')

        # Introduce triple errors
        encoded_message_with_triple_error = introduce_error(encoded_message.copy(), p, 'triple', [(0, 0), (1, 3), (2, 5)])
        erroneous_triple_matrix = encoded_message_with_triple_error[:p*p].reshape(p, p)
        binary_matrix_to_image(erroneous_triple_matrix, 'erroneous_triple_image.png')
        
        # Decode and correct the triple error
        decoded_message_triple_error, error_positions_triple = decode_double_error(encoded_message_with_triple_error, p)
        corrected_triple_matrix = decoded_message_triple_error.reshape(p, p)
        binary_matrix_to_image(corrected_triple_matrix, 'corrected_triple_image.png')

        print("Processing complete. Check the output images.")