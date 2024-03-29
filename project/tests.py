import datetime
import itertools
import os
import time

import joblib
import pandas as pd
import main
import GridExperiment
import daniel
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed

TEST_COUNT_BLOCKS = True
TEST_COUNT_COLORS = True
NUM_TESTS = 50  # max is 50
TEST_TARGET = main


def check_values(test_values, count_blocks_functions, count_colors_functions, module):
    blocks_dict = dict()
    colors_dict = dict()
    i = 1
    for image_path, (num_blocks, num_colors) in test_values:
        print(i)
        i += 1
        if TEST_COUNT_BLOCKS:
            for count_blocks_name in count_blocks_functions:
                blocks_dict[(image_path, count_blocks_name)] = get_error(
                    module, count_blocks_name, image_path, num_blocks
                )

        if TEST_COUNT_COLORS:
            for count_colors_name in count_colors_functions:
                colors_dict[(image_path, count_colors_name)] = get_error(
                    module, count_colors_name, image_path, num_colors
                )
    return blocks_dict, colors_dict


def get_expected_values(images_folder, values_folder):
    test_values = []
    for filename in os.listdir(images_folder)[:NUM_TESTS]:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            values_path = os.path.join(
                values_folder, filename[: filename.rfind(".")] + ".txt"
            )
            with open(values_path, "r") as file:
                num_blocks, num_colors = map(int, file.readlines())
                test_values.append((image_path, (num_blocks, num_colors)))
    return test_values


def get_testing_functions(module, name_filter):
    return [
        attr
        for attr in dir(module)
        if callable(getattr(module, attr)) and attr.startswith(name_filter)
    ]


def calculate_error(result, expected):
    return abs(result - expected) / expected


def get_error(module, func_name, test_case, expected_value):
    test_func = getattr(module, func_name)
    result = test_func(test_case)
    return calculate_error(result, expected_value)


def dict_to_df(data):
    return pd.DataFrame(
        [
            {"image_path": key[0], "function_name": key[1], "error": value}
            for key, value in data.items()
        ]
    )


def run_tests(images_folder, values_folder, module):
    test_values = get_expected_values(images_folder, values_folder)
    count_blocks_functions = get_testing_functions(module, "detect_pieces")
    count_colors_functions = get_testing_functions(module, "count_colors")

    blocks_dict, colors_dict = check_values(
        test_values, count_blocks_functions, count_colors_functions, module
    )

    blocks_df = dict_to_df(blocks_dict)
    colors_df = dict_to_df(colors_dict)

    blocks_df.to_csv("blocks.csv", index=False)
    colors_df.to_csv("colors.csv", index=False)

    if TEST_COUNT_BLOCKS:
        results_blocks = (
            blocks_df.groupby("function_name")["error"].mean()
        ).sort_values()
        best_function = results_blocks.index[0]
        lowest_error = results_blocks.iloc[0]
        print("BLOCKS:")
        print(
            f"The best function was {best_function} with an error of {round(lowest_error, 2)}\n"
        )

    if TEST_COUNT_COLORS:
        results_colors = (
            colors_df.groupby("function_name")["error"].mean()
        ).sort_values()
        best_function = results_colors.index[0]
        lowest_error = results_colors.iloc[0]
        print("COLORS:")
        print(
            f"The best function was {best_function} with an error of {round(lowest_error, 2)}"
        )

    print()


def prepare_image(image_path):
    image = cv2.imread(image_path)

    ratio = image.shape[1] / image.shape[0]
    height = 800
    width = int(height * ratio)

    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def process_combination_gpe(image_path, num_blocks):
    results = []
    image = prepare_image(image_path)
    for params in get_parameters_gpe_blocks():
        try:
            result = GridExperiment.evaluate_function(image.copy(), params)
            results.append(
                {
                    **params,
                    "image_path": image_path,
                    "error": calculate_error(result, num_blocks),
                }
            )
        except Exception as e:
            print("image: " + image_path)
            print("params: " + str(params))
            print("Error: " + str(e))
    return results


def process_combination_dani(image_path, num_blocks):
    results = []
    image = prepare_image(image_path)
    for params in get_parameters_dani_blocks():
        try:
            result = daniel.daniel(
                image.copy(),
                False,
                params["clip_limit"],
                params["tile_grid_size"],
                params["bilateral_filter_d"],
                params["bilateral_filter_sigma_color"],
                params["bilateral_filter_sigma_space"],
                params["canny_threshold1"],
                params["canny_threshold2"],
            )
            results.append(
                {
                    **params,
                    "image_path": image_path,
                    "error": calculate_error(result, num_blocks),
                }
            )
        except Exception as e:
            print("image: " + image_path)
            print("params: " + str(params))
            print("Error: " + str(e))
    return results


def store_results(data, grid_name, error_name):
    df = pd.DataFrame(data)
    df.to_csv(grid_name, index=False)
    cols = list(data[0].keys())
    cols.remove("image_path")
    cols.remove("error")
    errors = df.groupby(cols)["error"].mean().reset_index()
    errors.to_csv(error_name, index=False)


def get_parameters_gpe_blocks():
    parameters = dict()
    parameters["median_blur"] = [3, 7, 15, 31, 51]
    parameters["gaussian_blur"] = [3, 5, 7, 9, 11, 15, 21]
    parameters["sigma"] = [0, 2, 2.5, 3]
    parameters["canny_min"] = [50, 75, 100, 125, 150, 175, 200]
    parameters["canny_max"] = [100, 125, 150, 175, 200, 225, 250]
    parameters["dilation_it"] = [5, 6, 10, 13]

    param_combinations = []
    for combo in list(itertools.product(*parameters.values())):
        param_dict = {key: value for key, value in zip(parameters.keys(), combo)}
        if param_dict["canny_min"] < param_dict["canny_max"]:
            param_combinations.append(param_dict)
    return param_combinations


def get_parameters_dani_blocks():
    parameters = {
        "clip_limit": [2.0, 2.5, 3.0],
        "tile_grid_size": [(75, 75), (100, 100), (150, 150)],
        "bilateral_filter_d": [7, 11, 15],
        "bilateral_filter_sigma_color": [50, 70, 90, 110],
        "bilateral_filter_sigma_space": [50, 70, 90, 110],
        "canny_threshold1": [40, 50],
        "canny_threshold2": [60, 70, 80, 110, 140],
    }

    param_combinations = []
    for combo in list(itertools.product(*parameters.values())):
        param_dict = {key: value for key, value in zip(parameters.keys(), combo)}
        if param_dict["canny_threshold1"] < param_dict["canny_threshold2"]:
            param_combinations.append(param_dict)
    return param_combinations


def grid(images_folder, values_folder, func, joblib_name, grid_name, error_name):
    test_values = get_expected_values(images_folder, values_folder)

    results = []
    with ProcessPoolExecutor() as executor:
        futures = []
        done = 0
        for image_path, (num_blocks, num_colors) in test_values:
            future = executor.submit(func, image_path, num_blocks)
            futures.append(future)
        for future in as_completed(futures):
            done += 1
            print(f"{done}/{len(futures)} : {datetime.datetime.now()}")
            results.extend(future.result())
            store_results(results, grid_name, error_name)
    print("out")
    executor.shutdown()

    joblib.dump(results, joblib_name)

    store_results(results, grid_name, error_name)


if __name__ == "__main__":
    grid(
        "samples-task1/samples",
        "samples-task1/answers",
        process_combination_dani,
        "list_dani.joblib",
        "grid_dani.csv",
        "error_dani.csv",
    )
    # grid("samples-task1/samples", "samples-task1/answers", process_combination_gpe, 'list_gpe.joblib', 'grid.csv', 'error.csv')
    # main.DISPLAY = False
    # run_tests("samples-task1/samples", "samples-task1/answers", TEST_TARGET)
