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


def draw_bbs(img, bbs):
    img_clone = img.copy()
    for bb in bbs:
        x, y, w, h = bb
        cv2.rectangle(img_clone, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return img_clone


def equalize_hist_wrapper(image, d, s_color, s_space):
    if image.shape[2] == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = 255 - cv2.equalizeHist(v)
        h = cv2.bilateralFilter(h, d, s_color, s_space)
        return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
    else:
        return cv2.equalizeHist(image)


def get_bb(contours, bb_min_width, bb_min_height):
    bb = [cv2.boundingRect(contour) for contour in contours]
    clean_bb = []
    for i in range(len(bb)):
        # Check if the bounding box is not too small
        (x, y, w, h) = bb[i]

        if w < bb_min_width or h < bb_min_height:
            continue

        # Check if the bounding box is not contained in another bounding box
        is_contained = False
        for j in range(len(bb)):
            if i != j:
                (x2, y2, w2, h2) = bb[j]
                if x2 <= x and y2 <= y and x2 + w2 >= x + w and y2 + h2 >= y + h:
                    is_contained = True
                    break

        if not is_contained:
            clean_bb.append(bb[i])

    return clean_bb
