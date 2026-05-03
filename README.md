# Artifact Prediction and Explanation

This project predicts and explains artifacts in AI generated images using various machine learning models, including CNNs, CLIP, and the Ovis1.6-Gemma2-9B model. The system integrates these models to provide artifact classification and explanations for predictions.

## Installation

1. Create a new Conda environment:
   ```bash
   conda create --name artifact-env python=3.10
   ```

2. Activate the Conda environment:
   ```bash
   conda activate artifact-env
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

Download model weights from [here](https://drive.google.com/drive/folders/1sccm2sgXpiGM-eSK6c9EXqfTq2n_mXUA?usp=sharing) and put them in the cnn_weights folder.
Download svm weights from [here](https://drive.google.com/drive/folders/1ffVpkRhtRLpGxsHjxRq42hbRcAdvo9sq?usp=sharing).

To run the code, use the `main.py` file, which is the entry point for processing images and generating artifact predictions and explanations.

```bash
python main.py --folder_path <path_to_images> --tsv_file_path <path_to_tsv_file> --svm_path <path_to_svm_model> --cnn_model_base_path <path_to_cnn_model> --num_preds <num_predictions> --limit_flag <limit_flag> --clip_list_limit <clip_list_limit>
```

To ensure the final keys of the json output are in the correct case, run the below command

```bash
python .\json_fix.py -i <path_to_output.json> -o <path_to_final_output.json>
```
### Command-Line Arguments:
- `folder_path`: Path to the folder containing images.
- `tsv_file_path`: Path to the TSV file containing data for class labels and artifact mappings. 
- `svm_path`: Path to the pre-trained SVM model.
- `cnn_model_base_path`: Base path where CNN models (e.g., ResNet, DenseNet) are stored.
- `num_preds`: The number of predictions to make (default: 13).
- `limit_flag`: Determines if similar classes count towards the prediction limit (default: 0).
- `clip_list_limit`: Limits the size of the CLIP prediction list (default: 15).

## Example

Here is an example of how to run the `main.py` script:

```bash
python main.py --folder_path ./images --tsv_file_path ./Class_Artifact_Mapping.tsv --svm_path ./svm_cifar10_model.joblib --cnn_model_base_path ./cnn_weights --num_preds 13 --limit_flag 0 --clip_list_limit 15
```
Here is an example of how to run the `json_fix.py` script:

```bash
python json_fix.py -i ./output.json -o ./final_output.json
```
