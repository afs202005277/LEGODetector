import os
import pandas as pd
import main

if __name__ == '__main__':
    images_folder = "samples-task1/samples"
    values_folder = "samples-task1/answers"

    count_blocks_functions = [attr for attr in dir(main) if
                              callable(getattr(main, attr)) and attr.startswith('detect_pieces')]
    count_colors_functions = [attr for attr in dir(main) if
                              callable(getattr(main, attr)) and attr.startswith('count_colors')]
    main.DISPLAY = False

    test_values = []
    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            values_path = os.path.join(values_folder, filename[:filename.rfind('.')] + ".txt")
            with open(values_path, "r") as file:
                num_blocks, num_colors = map(int, file.readlines())
                test_values.append((image_path, (num_blocks, num_colors)))

    blocks_dict = dict()
    colors_dict = dict()
    for image_path, (num_blocks, num_colors) in test_values:
        for count_blocks_name in count_blocks_functions:
            count_blocks = getattr(main, count_blocks_name)
            result = count_blocks(image_path)
            expected = num_blocks
            blocks_count_error = abs(result - expected) / expected
            blocks_dict[(image_path, count_blocks_name)] = blocks_count_error

        for count_colors_name in count_colors_functions:
            count_colors = getattr(main, count_colors_name)
            result = count_colors(image_path)
            expected = num_colors
            colors_count_error = abs(result - expected) / expected
            colors_dict[(image_path, count_colors_name)] = colors_count_error

    blocks_data_list = [{'image_path': key[0], 'function_name': key[1], 'error': value} for key, value in
                        blocks_dict.items()]
    colors_data_list = [{'image_path': key[0], 'function_name': key[1], 'error': value} for key, value in
                        colors_dict.items()]
    # Create a DataFrame
    blocks_df = pd.DataFrame(blocks_data_list)
    colors_df = pd.DataFrame(colors_data_list)

    results_blocks = (blocks_df.groupby('function_name')['error'].mean()).sort_values()
    results_colors = (colors_df.groupby('function_name')['error'].mean()).sort_values()

    best_function = results_blocks.index[0]
    lowest_error = results_blocks.iloc[0]
    print('BLOCKS:')
    print(f'The best function was {best_function} with an error of {round(lowest_error, 2)}\n')

    best_function = results_colors.index[0]
    lowest_error = results_colors.iloc[0]
    print('COLORS:')
    print(f'The best function was {best_function} with an error of {round(lowest_error, 2)}')

    print()
