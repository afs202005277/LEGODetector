import os
import shutil


# Function to recursively traverse directories and copy images
def copy_images(source_dir):
    counter = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Construct new file name
                parent_dir_name = os.path.basename(root)
                new_filename = f"{counter}_{parent_dir_name}.{file.split('.')[-1]}"
                # Destination directory
                destination_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive_dataset")
                # Copy file to new directory with new filename
                shutil.copy(os.path.join(root, file), os.path.join(destination_dir, new_filename))
                print(counter)
                counter += 1


# Main function
def main():
    source_dir = "original_dataset/photos"  # Set your source directory here
    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive_dataset")):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drive_dataset"))

    copy_images(source_dir)


if __name__ == "__main__":
    main()
