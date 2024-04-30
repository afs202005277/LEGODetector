import cv2
import functools
from daniel import daniel
import json
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


def calculate_error(result, expected):
    return abs(result - expected) / expected


def create_combinations(param_grid):
    combinations = {}
    id = 0

    for combo in list(itertools.product(*param_grid.values())):
        param_dict = {key: value for key, value in zip(param_grid.keys(), combo)}
        if param_dict["canny_threshold1"] < param_dict["canny_threshold2"]:
            combinations[id] = param_dict
            id += 1

    return combinations


def store_results(results, args, answers_json, results_path):
    expected_colors = sum(answer["colors"] for answer in answers_json.values())
    expected_pieces = sum(answer["pieces"] for answer in answers_json.values())

    for id, result in results.items():
        args[id]["error_pieces"] = calculate_error(result["pieces"], expected_pieces)
        args[id]["error_colors"] = calculate_error(result["colors"], expected_colors)

    df = pd.DataFrame.from_dict(args, orient="index")
    df.sort_values(by=["error_pieces", "error_colors"], inplace=True)
    df.to_csv(results_path, index=False)


def grid(answers_json_path, test_func, param_grid, results_path):
    with open(answers_json_path, "r") as f:
        answers_json = json.load(f)

    args = create_combinations(param_grid)
    print(f"Number of combinations: {len(args)}")
    results = {id: {"pieces": 0, "colors": 0} for id in args.keys()}

    with ProcessPoolExecutor() as executor:
        futures = []
        done = 0

        for photo_data in answers_json.values():
            img = cv2.imread(photo_data["path"])

            for id, params in args.items():
                future = executor.submit(test_func, img, id, **params)
                futures.append(future)

        for future in as_completed(futures):
            done += 1
            print(f"{done}/{len(futures)}")
            num_pieces, num_colors, id = future.result()
            results[id]["pieces"] += num_pieces
            results[id]["colors"] += num_colors

    executor.shutdown()

    store_results(results, args, answers_json, results_path)


def test(answer_json_path, test_func, args):
    wrong_images_pieces = []
    wrong_images_colors = []

    errors_pieces = []
    error_colors = []

    with open(answer_json_path, "r") as f:
        answer_json = json.load(f)

    for photo_name, photo_data in answer_json.items():
        img = cv2.imread(photo_data["path"])
        num_pieces, num_colors = test_func(img, **args)

        if num_pieces != photo_data["pieces"]:
            wrong_images_pieces.append(photo_name)

        if num_colors != photo_data["colors"]:
            wrong_images_colors.append(photo_name)

        errors_pieces.append(calculate_error(num_pieces, photo_data["pieces"]))
        error_colors.append(calculate_error(num_colors, photo_data["colors"]))

    error_pieces = sum(errors_pieces) / len(errors_pieces)
    error_colors = sum(error_colors) / len(error_colors)

    print(f"Error in pieces: {error_pieces}")
    print(f"Error in colors: {error_colors}")
    print(
        f"Wrong images in pieces: {functools.reduce(lambda x, y: x + ', ' + y, wrong_images_pieces)}"
    )
    #print(
    #    f"Wrong images in colors: {functools.reduce(lambda x, y: x + ', ' + y, wrong_images_colors)}"
    #)

    return error_pieces, error_colors


if __name__ == "__main__":
    # param_grid = {
    #     "clip_limit": [2.0, 2.5, 3.0],
    #     "tile_grid_size": [(75, 75), (100, 100), (150, 150)],
    #     "bilateral_filter_d": [7, 11, 15],
    #     "bilateral_filter_sigma_color": [50, 70, 90, 110],
    #     "bilateral_filter_sigma_space": [50, 70, 90, 110],
    #     "canny_threshold1": [40, 50],
    #     "canny_threshold2": [60, 70, 80, 110, 140],
    # }

    # answers_json_path = "samples-task1/answers.json"
    # results_path = "results.csv"
    # grid(answers_json_path, daniel, param_grid, results_path)

    test("samples-task1/answers.json", daniel, {"display": False})
