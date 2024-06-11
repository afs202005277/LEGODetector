# README

## Model Inference 

The application can be run from the command line using the following syntax:

```bash
python run_cnn.py --model_name <model_name> --path_to_model <path_to_model> --batch_size <batch_size> --input_json <input_json> --output_json <output_json>
```

Here is a description of the command line arguments:

- `--model_name`: The name of the model to use. Options are 'custom', 'resnet', 'vgg', 'densenet', 'efficientnet'. Default is 'custom'. But the best model is 'efficientnet'.
- `--path_to_model`: The path to the pre-trained model file weights. Default is 'models/custom.pth'.
- `--batch_size`: The number of images to process in a single batch. Default is 32.
- `--input_json`: The path to the JSON file containing the list of image file paths to process. Default is 'test.json'.
- `--output_json`: The path to the JSON file where the output predictions will be saved. Default is 'output.json'.

## Input & Output JSON

The input JSON file should have the following format:

```json
[
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    "path/to/image3.jpg"
]
```

The output JSON file will have the following format:

```json
{
  "path/to/image1.jpg": 1,
  "path/to/image2.jpg": 2,
  "path/to/image3.jpg": 3
}
```

## Example

Here is an example of how to run the application:

```bash
python run_cnn.py --model_name "resnet" --path_to_model "models/resnet.pth" --batch_size 64 --input_json "test.json" --output_json "output.json"
```

This will run the application using the ResNet model, with a batch size of 64, processing the images listed in 'test.json', and saving the output predictions to 'output.json'.

## Models Weights

The weights for the models (.pth files) can be downloaded from this Google Drive link: [Models Weights](https://drive.google.com/drive/folders/1u5RZb548dpJL0DCff4QrVjuk2707T_79?usp=sharing). In addition to the weights, there are also all the notebooks and files created during the development of task 2.