import os

def extract_unique_image_names(parent_directory, output_file):
    # Set to hold the unique names of image files
    image_files = set()

    # Define image file extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

    # Walk through all  directories in the specified parent directory
    for root, dirs, files in os.walk(parent_directory):
        # Check each file to see if it's an image and add to set for uniqueness
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                # Add the file name to the set
                image_files.add(file)

    # Writing the unique image file names to a text file
    with open(output_file, 'w') as file:
        for image in sorted(image_files):
            file.write(image + '\n')

    print(f"Unique image names have been written to {output_file}")

parent_directory = r"/home/ml2/Desktop/Vscode/Background_removal/DIS/backdrop"

output_file = 'unique_image_names.txt'

# Call the function with the path to the parent directory and the output file
extract_unique_image_names(parent_directory, output_file)
