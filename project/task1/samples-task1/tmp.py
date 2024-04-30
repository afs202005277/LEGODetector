import os
import cv2
import matplotlib.pyplot as plt


def resize_image(image, dimensions):
    height, width = image.shape[:2]
    if dimensions and (width > dimensions[0] or height > dimensions[1]):
        aspect_ratio = width / height
        if width > height:
            new_width = dimensions[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = dimensions[1]
            new_width = int(new_height * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))

    return image


def process_image(image_path):
    print(image_path)
    # Check if output file already exists
    output_path = os.path.join("answers", os.path.splitext(os.path.basename(image_path))[0] + ".txt")
    if os.path.exists(output_path):
        return

    image = cv2.imread(image_path)
    image = resize_image(image, [1000, 1000])
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    num_blocks = int(input("Number of blocks? "))

    # Ask user for the number of colors
    num_colors = int(input("Number of colors? "))

    return num_blocks, num_colors


def main():
    input_folder = "samples"
    output_folder = "answers"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

            p_img = process_image(image_path)
            if p_img:
                num_blocks, num_colors = p_img

                with open(output_path, "w") as file:
                    file.write(f"{num_blocks}\n")
                    file.write(f"{num_colors}\n")


if __name__ == "__main__":
    main()
