from PIL import Image
import numpy as np

def image_to_binary_file(image_path, output_file):
    try:
        # Open the image and convert to grayscale
        image = Image.open(image_path).convert('L')
        
        # Convert the image to a NumPy array
        image_array = np.array(image)
        
        # Normalize to binary (0 or 1)
        binary_array = (image_array > 128).astype(int)  # Thresholding at 128 for binary
        
        # Save the binary array to a text file
        with open(output_file, 'w') as file:
            for row in binary_array:
                file.write(' '.join(map(str, row)) + '\n')
        
        print(f"Binary data saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    image_path = input("Enter the path of the image file: ")
    output_file = 'c.txt'
    
    image_to_binary_file(image_path, output_file)
