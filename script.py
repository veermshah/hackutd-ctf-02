import os
import base64
import random
import string

def generate_random_data(size):
    """Generate random bytes of a specified size."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))

def encode_to_base64(data):
    """Encode the data to a base64 string."""
    return base64.b64encode(data.encode()).decode('utf-8')

def create_base64_file(file_name, size):
    """Create a file with a base64-encoded string."""
    random_data = generate_random_data(size)
    base64_string = encode_to_base64(random_data)

    with open(file_name, 'w') as f:
        f.write(base64_string)

    print(f"Created file: {file_name} with {size} random characters encoded in base64.")

def create_multiple_base64_files(directory, num_files, file_size):
    """Create multiple base64-encoded files in the specified directory."""
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(num_files):
        file_name = os.path.join(directory, f"file_{i+1}.txt")
        create_base64_file(file_name, file_size)

if __name__ == "__main__":
    # Configuration
    output_directory = '.'  # Directory to store the files
    number_of_files = 10  # Number of files to create
    file_size = 1024  # Size of each file in random characters

    # Generate files
    create_multiple_base64_files(output_directory, number_of_files, file_size)
