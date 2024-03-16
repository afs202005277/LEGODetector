import cv2


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
