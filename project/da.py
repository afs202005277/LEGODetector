import os
import cv2
import numpy as np
from multiprocessing import Pool


# Function to perform data augmentation
def augment_image(args):
    image_path, image_id, num_pieces, output_folder = args

    # Load the image
    image = cv2.imread(image_path)

    # Get the original file extension
    original_extension = image_path.split(".")[-1]

    # Save the original image
    cv2.imwrite(os.path.join(output_folder, f"{image_id}_original_{num_pieces}.{original_extension}"), image)

    # Rotate the image by 90, 180, and 270 degrees
    for angle in [90, 180, 270]:
        rotated_image = np.rot90(image, k=angle // 90)
        cv2.imwrite(os.path.join(output_folder, f"{image_id}_rotated_{angle}_{num_pieces}.{original_extension}"),
                    rotated_image)

    # Add noise to the image
    noisy_image = np.uint8(image + np.random.normal(loc=0, scale=25, size=image.shape))
    cv2.imwrite(os.path.join(output_folder, f"{image_id}_noisy_{num_pieces}.{original_extension}"), noisy_image)

    # Mirror the image horizontally and vertically
    mirrored_image_horizontal = cv2.flip(image, 1)
    cv2.imwrite(os.path.join(output_folder, f"{image_id}_mirrored_horizontal_{num_pieces}.{original_extension}"),
                mirrored_image_horizontal)

    mirrored_image_vertical = cv2.flip(image, 0)
    cv2.imwrite(os.path.join(output_folder, f"{image_id}_mirrored_vertical_{num_pieces}.{original_extension}"),
                mirrored_image_vertical)

    # Resize the image to different scales
    scales = [0.25, 0.5, 0.75]  # Example scales
    for scale in scales:
        scaled_width = int(image.shape[1] * scale)
        scaled_height = int(image.shape[0] * scale)
        scaled_image = cv2.resize(image, (scaled_width, scaled_height))
        cv2.imwrite(os.path.join(output_folder, f"{image_id}_scaled_{scale}_{num_pieces}.{original_extension}"),
                    scaled_image)


# Main function
def main():
    input_folder = "original_dataset/photos/"
    output_folder = "extended_dataset"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_id = 0
    args_list = []

    # Iterate through each subfolder
    for folder_name in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, folder_name)

        # Iterate through each image in the subfolder
        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            num_pieces = int(folder_name)
            args_list.append((image_path, image_id, num_pieces, output_folder))
            image_id += 1

    # Multiprocessing
    with Pool(processes=16) as pool:
        pool.map(augment_image, args_list)


if __name__ == "__main__":
    main()
