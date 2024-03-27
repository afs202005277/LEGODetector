import cv2
import functools
from daniel import daniel
import json


def calculate_error(result, expected):
    return abs(result - expected) / expected


def grid(answers_json, test_func, job_name, grid_name, error_name):
    pass


def test(answer_json_path, test_func, args):
    result_pieces = 0
    expected_pieces = 0

    result_colors = 0
    expected_colors = 0

    wrong_images_pieces = []
    wrong_images_colors = []

    with open(answer_json_path, "r") as f:
        answer_json = json.load(f)

    for photo_name, photo_data in answer_json.items():
        img = cv2.imread(photo_data["path"])
        num_pieces, num_colors = test_func(img, *args)

        if num_pieces != photo_data["pieces"]:
            wrong_images_pieces.append(photo_name)

        result_pieces += num_pieces
        expected_pieces += photo_data["pieces"]

        if num_colors != photo_data["colors"]:
            wrong_images_colors.append(photo_name)

        result_colors += num_colors
        expected_colors += photo_data["colors"]

    error_pieces = calculate_error(result_pieces, expected_pieces)
    error_colors = calculate_error(result_colors, expected_colors)

    print(f"Error in pieces: {error_pieces}")
    print(f"Error in colors: {error_colors}")
    print(
        f"Wrong images in pieces: {functools.reduce(lambda x, y: x + ', ' + y, wrong_images_pieces)}"
    )
    print(
        f"Wrong images in colors: {functools.reduce(lambda x, y: x + ', ' + y, wrong_images_colors)}"
    )

    return error_pieces, error_colors

if __name__ == "__main__":
    test("samples-task1/answers.json", daniel, {})
