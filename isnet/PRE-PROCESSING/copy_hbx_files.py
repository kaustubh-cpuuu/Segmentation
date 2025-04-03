import os
import shutil


def copy_files_with_structure(base_path, dest_path):
    # Walk through the entire directory structure
    for root, dirs, files in os.walk(base_path):
        # Check if the current directory is either "original" or "Original"
        if os.path.basename(root).lower() == "original":
            print(f"Found 'Original' directory: {root}")  # Debug print

            # Retain the folder structure in the destination path
            relative_path = os.path.relpath(root, base_path)
            dest_dir = os.path.join(dest_path, relative_path)
            os.makedirs(dest_dir, exist_ok=True)

            # Copy files excluding "Thumbs.db", ".DS_Store", and "desktop.ini"
            for file in files:
                if file not in ["Thumbs.db", ".DS_Store", "desktop.ini"]:
                    source_file = os.path.join(root, file)

                    # Validate the source path
                    if os.path.exists(source_file):
                        dest_file = os.path.join(dest_dir, file)
                        shutil.copy2(source_file, dest_file)
                        print(f"Copied: {source_file} -> {dest_file}")
                    else:
                        print(f"Source file does not exist: {source_file}")


if __name__ == "__main__":
    # Example usage
    base_path = "/path/to/base/folder"
    dest_path = "/path/to/destination/folder"

    copy_files_with_structure(base_path, dest_path)