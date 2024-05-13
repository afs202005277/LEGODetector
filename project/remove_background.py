import copy
import random

import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

random.seed(42)
DISPLAY = False
WHITE = 200


def display_images(images, window_names):
    named_images = zip(window_names, images)
    for name, image in named_images:
        cv2.imshow(name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def adjust_image_contrast(image, d, s_color, s_space):
    if image.shape[2] == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = 255 - cv2.equalizeHist(v)
        h = cv2.bilateralFilter(h, d, s_color, s_space)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(image)


def image_segmentation(image, bb, its=8):
    image = cv2.imread(image)
    combination = (15, 160, 100)
    masks = []
    original_image = adjust_image_contrast(image, *combination)
    mask = np.zeros(original_image.shape[:2], np.uint8)
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(0)
    (mask, bg_model, fgModel) = cv2.grabCut(original_image, mask, bb, bg_model, fg_model, its,
                                            cv2.GC_INIT_WITH_RECT)

    output_mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)

    output_mask = (output_mask * 255).astype("uint8")
    masks.append(output_mask)

    if len(masks) == 0:
        return np.zeros_like(image)
    merged_mask = np.zeros_like(masks[0])
    for mask in masks:
        merged_mask = cv2.bitwise_or(merged_mask, mask)
    filtered_image = cv2.bitwise_and(image, image, mask=merged_mask)

    # display_images([filtered_image], ['img'])
    return filtered_image


def extract_bounding_box_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    object_elem = root.find('object')
    bndbox = object_elem.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    width = xmax - xmin
    height = ymax - ymin
    return (xmin, ymin, width, height)


def is_image_completely_black(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return not cv2.countNonZero(grayscale_image)


def remove_background(image, hue_margin=30, sat_margin=100, val_margin=255):
    original = image.copy()
    image = cv2.GaussianBlur(image, (55, 55), sigmaX=0)

    if image is None:
        print("Error: Unable to read image.")
        return

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Flatten the image to a 2D array (height * width, 3)
    flattened_image = hsv_image.reshape((-1, 3))

    # Compute the most common color using k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        np.float32(flattened_image), 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    most_common_color = np.uint8(centers[0])
    # Define the range for background color with a margin of error
    lower_bound = np.array(
        [
            most_common_color[0] - hue_margin,
            most_common_color[1] - sat_margin,
            most_common_color[2] - val_margin,
        ]
    )
    upper_bound = np.array(
        [
            most_common_color[0] + hue_margin,
            most_common_color[1] + sat_margin,
            most_common_color[2] + val_margin,
        ]
    )

    # Create a mask for pixels within the defined color range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Invert the mask to represent the background
    background_mask = cv2.bitwise_not(mask)

    # Replace background pixels with original pixels
    result = cv2.bitwise_and(original, original, mask=background_mask)

    # Display the result
    if DISPLAY:
        display_images([result], ["Background Removed"])

    return result


def process_images_with_xml(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            xml_path = os.path.join(folder_path, os.path.splitext(filename)[0] + '.xml')
            if os.path.isfile(xml_path):
                bbox_opencv = extract_bounding_box_from_xml(xml_path)
                segmented_img = image_segmentation(image_path, bbox_opencv)
                final_img = remove_background(segmented_img)
                x, y, w, h = bbox_opencv
                if is_image_completely_black(final_img):
                    cropped_image = segmented_img[y:y + h, x:x + w]
                else:
                    cropped_image = final_img[y:y + h, x:x + w]
                cv2.imwrite('individual_pieces/' + filename, cropped_image)
            else:
                print("XML not found: " + xml_path)


def get_random_images(folder_path, num_images):
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    # Randomly choose num_images from the list
    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    return selected_images


def convert_near_black_to_black(color_image, threshold=15):
    b, g, r = cv2.split(color_image)

    mask_b = b <= threshold
    mask_g = g <= threshold
    mask_r = r <= threshold

    b[mask_b] = 0
    g[mask_g] = 0
    r[mask_r] = 0

    result_image = cv2.merge((b, g, r))

    return result_image


def transform_black_to_white(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to get the binary mask of the black background
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Invert the mask
    mask_inv = cv2.bitwise_not(mask)

    # Extract the foreground
    foreground = cv2.bitwise_and(image, image, mask=mask)

    # Create a white background
    background = np.full(image.shape, WHITE, dtype=np.uint8)

    # Merge the foreground with the white background
    result = cv2.bitwise_or(background, foreground, mask=mask_inv)
    return result


def place_images_in_canvas(image_folder, canvas_size, num_pieces):
    retries = 0

    while retries < 200:
        deadlock = False
        retries += 1
        canvas = WHITE * np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        placed_positions = []
        selected_images = get_random_images(image_folder, num_pieces)
        for image_file in selected_images:
            image = cv2.imread(os.path.join(image_folder, image_file))

            if image.shape[1] > canvas_size[0] or image.shape[0] > canvas_size[1]:
                continue

            its = 0
            while True:
                its += 1
                x = random.randint(0, canvas_size[0] - image.shape[1])
                y = random.randint(0, canvas_size[1] - image.shape[0])
                if all((x + image.shape[1] <= pos[0] or y + image.shape[0] <= pos[1]
                        or pos[0] + placed_image.shape[1] <= x or pos[1] + placed_image.shape[0] <= y)
                       for pos, placed_image in placed_positions):
                    break
                if its > 200:
                    deadlock = True
                    break
            if deadlock:
                break
            canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
            placed_positions.append(((x, y), image))
        if not deadlock:
            return canvas
    print("Deadlock detected: " + str(num_pieces))
    return None


def generate_individual_lego_pieces():
    folder_path = "original_dataset/renders/1/"
    process_images_with_xml(folder_path)


def add_background(image):
    # Access the backgrounds folder and randomly pick an image
    backgrounds_folder = "backgrounds"
    background_files = os.listdir(backgrounds_folder)
    background_file = random.choice(background_files)

    # Load and resize the background image
    background_image = cv2.imread(os.path.join(backgrounds_folder, background_file))
    background_image = cv2.resize(background_image, (image.shape[1], image.shape[0]))

    mask = np.all((image == (WHITE, WHITE, WHITE)) | (image == (0, 0, 0)), axis=-1)
    image[mask] = background_image[mask]
    return image


def enhance_image(image):
    # Apply Gaussian blur to reduce sharpness
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Adjust brightness and contrast
    alpha = 1.0  # Contrast control (1.0-3.0)
    beta = 10  # Brightness control (0-100)
    adjusted_image = cv2.convertScaleAbs(blurred_image, alpha=alpha, beta=beta)

    return adjusted_image


def generate_group_images(num_pieces, augmentation_factor=1):
    image_folder = "individual_pieces"
    final_image_size = (390, 520)
    result_image = place_images_in_canvas(image_folder, (
        round(final_image_size[0] * augmentation_factor), round(final_image_size[1] * augmentation_factor)), num_pieces)

    if result_image is not None:
        result_image = convert_near_black_to_black(result_image)

        result_image = add_background(result_image)
        result_image = cv2.resize(result_image, final_image_size)
        result_image = enhance_image(result_image)
        # display_images([result_image, ['pre'])

        return result_image


def extract_max_counter(folder_path):
    if not os.path.isdir(folder_path):
        return 0

    files = os.listdir(folder_path)
    if not files:
        return 0

    max_counter = 0

    for file_name in files:
        id_value = file_name[:file_name.rfind('_')]
        if id_value.isdigit():
            counter = int(id_value)
            max_counter = max(max_counter, counter)
    return max_counter


def generate_multiple_images(folder_path, num_images, num_pieces):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    starting_id = extract_max_counter(folder_path) + 1
    num_images_generated = 0
    aug_factor = 1
    while num_images_generated < num_images:
        image = generate_group_images(num_pieces, aug_factor)
        if image is not None:
            cv2.imwrite(f"{folder_path}/{starting_id}_{num_pieces}.jpg", image)
            starting_id += 1
            num_images_generated += 1
        else:
            aug_factor += 0.2
            pass


def generate_dataset(folder_path):
    num_images = 1000
    for num_pieces in range(6, 32 + 1):
        generate_multiple_images(folder_path, num_images, num_pieces)
        print(f"Generated {num_images} images with {num_pieces} pieces each.")


def post_process_pieces(folder_path):
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([180, 40, 255], dtype=np.uint8)
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

        white_mask = cv2.bitwise_not(white_mask)

        result = cv2.bitwise_and(image, image, mask=white_mask)

        # display_images([result], ['result'])
        cv2.imwrite(image_path, result)


def blacklist_pieces(folder_path):
    deleted_images = []
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.endswith(('.jpg', '.jpeg', '.png'))]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        cv2.imshow('Image', image)
        key = cv2.waitKey(0)

        # Check if the key pressed is 'x'
        if key == ord('x'):
            os.remove(image_path)
            deleted_images.append(image_path)
        elif key == 27:  # Check if the key pressed is the ESC key (27)
            break

    cv2.destroyAllWindows()
    with open('deleted_images.txt', 'a') as file:
        for image_path in deleted_images:
            file.write(image_path + '\n')


def delete_small_images(folder_path, min_size):
    with open("deleted_images.txt", "a") as delete_file:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                image = cv2.imread(file_path)

                height, width, _ = image.shape
                image_size = height * width

                if image_size < min_size:
                    os.remove(file_path)
                    delete_file.write(file_path + "\n")


def get_smallest_image_size(folder_path):
    smallest_size = None
    dims = None
    small_img = None
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and any(file_path.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            image = cv2.imread(file_path)

            height, width, _ = image.shape
            image_size = height * width

            if smallest_size is None or image_size < smallest_size:
                smallest_size = image_size
                dims = (width, height)
                small_img = image

    return smallest_size, dims, small_img


def small_images():
    delete_small_images('individual_pieces', 550)
    smallest_size, dims, small_img = get_smallest_image_size('individual_pieces')
    print(smallest_size, dims)
    display_images([small_img], ['small'])


def count_files_with_suffix(folder_path, suffix_range):
    suffix_counts = dict()

    for suffix in suffix_range:
        suffix_counts[suffix] = 0
        for filename in os.listdir(folder_path):
            if filename.endswith(f"_{suffix}.jpg") or filename.endswith(f"_{suffix}.png"):
                suffix_counts[suffix] += 1

    return suffix_counts


def main():
    # generate_individual_lego_pieces()
    # generate_dataset('generated_data')
    # small_images()
    num_piece_range = range(1, 33)
    image_counts = count_files_with_suffix('drive_dataset', num_piece_range)
    max_img = max(image_counts.values())

    for num_pieces in num_piece_range:
        current_img_count = image_counts[num_pieces]
        images_needed = max_img - current_img_count
        if images_needed > 0:
            generate_multiple_images('missing', images_needed, num_pieces)
            print(f"Generated {images_needed} images with {num_pieces} pieces each.")


if __name__ == '__main__':
    # post_process_pieces('individual_pieces')
    # blacklist_pieces('individual_pieces')
    main()
