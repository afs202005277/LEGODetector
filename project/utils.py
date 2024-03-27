import cv2
import os
import json


def display_images(images, window_names, dimensions=None):
    named_images = zip(window_names, images)
    for name, image in named_images:
        resized_img = resize_image(image, dimensions)
        cv2.imshow(name, resized_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def create_answers_json(path_answers, path_images, path_out_json):
    photos = os.listdir(path_images)

    answers = {}

    for photo in photos:
        photo_name = photo.split(".")[0]
        with open(f"{path_answers}/{photo_name}.txt", "r") as f:
            num_blocks, num_colors = map(int, f.readlines())
            answers[photo_name] = {
                "path": f"{path_images}/{photo}",
                "pieces": num_blocks,
                "colors": num_colors,
            }

    with open(path_out_json, "w") as f:
        json.dump(answers, f)

    print("JSON file created!")
