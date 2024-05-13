import os
from PIL import Image
from torchvision import transforms

def resize_images_in_folder(folder_path, target_size):
    # Define transformation
    transform = transforms.Resize(target_size)
    
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Open the image
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            
            # Apply transformation
            resized_image = transform(image)
            
            # Save the resized image back to the same file
            resized_image.save(image_path)

# Example usage
folder_path = "drive_dataset"
target_size = (520, 390)  # Specify your target size here
resize_images_in_folder(folder_path, target_size)

