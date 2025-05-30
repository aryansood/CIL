## Environment Setup

Create a conda environment and activate it:

```bash
conda create -n CIL python=3.11
conda activate CIL
```

Install the required dependencies (via `pip`):

```bash
conda install pip
pip install -r requirements.txt
```

## Model Training

The file `example_training.py` is an example containing the code for running a model. It is the same script used by us to achieve our best Kaggle score.

In the example file, the `begin_training_loop` function is called. That function is called during training to start the main model training process. The function's parameter `random_split` is a boolean value that should be `True` if you want to perform a random split of your data, so creating at runtime a training set and validation set at random. If `random_split = False`, then a custom split should be provided, so both the `train_split` and `val_split` parameters should be non-empty.

Here there is an example of such parameters using the training and validation splits created by clustering images based on their cosine similarity (as we did in our experiments):
- `train_split=pd.read_csv("train_split.csv")["file_name"].to_list()`
- `val_split=pd.read_csv("val_split.csv")["file_name"].to_list()`

Check the files `train_split.csv` and `val_split.csv` in the repository root folder to see how to format the files. Both files are formatted as _Comma-Separated Value_ (_.csv_) files with only one column, `file_name`, and the values contained in the column are alternating names of depth values (`.npy`) and RGB Images (`.png`) filenames. 

## Model Evaluation

To evaluate a previously trained model, run the script `test_evaluation.py` after setting the model weights, the path of your test data and the directory where the predictions should be saved (refer to the comments inside the evaluation file for more details on how to change the values to evaluate a specific model).

To use the same model weights and test data as done in our best experiment, do not change any variables in the script. Simply [download the model weights](https://drive.google.com/file/d/1iIvGZ2I5k3vbXvfXvBV6dONWtoDYHTlK/view) and save the downloaded checkpoint file in the root directory of this repository.  

## Submitting Results

After evaluation, to generate a Kaggle submission file, run the script `create_outputs.py`.

## Models

Inside the `models/` subfolder there are the files for the different models we tested.

### Models trained from scratch

Those are the models we trained from scratch by using the data referred in `train_split.csv`.

- _Base U-Net_: `models/large_unet.py`
- _UNet++_: `models/unetpp.py`
- _Swin transformer encoder with a U-Net decoder_: `models/unet_swin_depth_estimator.py` 
- _ViT transformer encoder with a U-Net decoder_: `models/unet_vit_depth_estimator.py`

### ResNet Model

The model created using a _ResNet encoder plus a U-Net decoder_ used pre-trained weights for the encoder and was fine-tuned on the training data referred in `train_split.csv`. The encoder's pre-trained weights were taken from `torchvision.models.ResNet50_Weights`.
The files detailing ResNet based models are:
- *ResNet*: `models/resnet_unet_decoder.py`
- *ResNet-Transformer*: `models/resnet_transformer_unet.py`

### SegFormer Model

The model created by using _SegFormer encoder plus SegFormer Decoder_ (modified by us to output a depth mask) used pre-trained weights for the encoder and was fine-tuned on the training data referred in `train_split.csv`.  The encoder's pre-trained weights were taken from [`nvidia/segformer-b5-finetuned-ade-640-640`](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640). 

For more details of the actual model's implementation refer to `models/segformer_depth.py`. 

### Mask2Former Model

The model created by using a _Mask2Former architecture plus double convolutional layer_ used pre-trained weights for the Mask2Former architecture and was fine-tuned on the training data as referred in `train_split.csv`. The fine-tuned weights used were taken from [`facebook/mask2former-swin-small-coco-instance`](https://huggingface.co/facebook/mask2former-swin-small-coco-instance). 

For more details of the actual model's implementation refer to: `models/Mask2Former_depth.py`.

**Table: Depth estimation performance across different architectures**

| Model                     | Validation Loss | Training Loss | Kaggle Public Score |
|--------------------------|-----------------|----------------|---------------------|
| **U-Net Based models**   |                 |                |                     |
| Base U-Net               | 0.27101         | 0.32242        | --                  |
| UNet++                   | 0.36963         | 0.44006        | --                  |
| **ResNet Based models**  |                 |                |                     |
| ResNet                   | 0.15886         | 0.09157        | 0.16063             |
| ResNet-Transformer       | 0.17421         | 0.11357        | 0.21408             |
| **Mask2Former model**     |                 |                |                     |
| mask2former-swin-small-coco-instance         | 0.12976         | 0.097775       | 0.13343             |
| **SegFormer models**     |                 |                |                     |
| segformer-b4-512-512     | 0.14236         | 0.11561        | 0.14048             |
| segformer-b5-640-640     | 0.11474         | 0.15951        | **0.12621**         |

> **Note:** The SegFormer based model achieved a better result compared to the other models, showing a 27.8% improvement over the ResNet based model and a 68.9% improvement over the Base-U-Net and the UNet++ model.

### Model Weights

To evaluate our results, we provide the model weights of the following models:
| **Model** | **File** |
| ---------|-----------|
| segformer-b4-512-512 | [polybox](https://polybox.ethz.ch/index.php/s/abqc8tgFegTScZ7) |
| segformer-b5-640-640  | [polybox](https://polybox.ethz.ch/index.php/s/6XXgN6bzNNbSj4D) |
| mask2former-swin-small-coco-instance  | [polybox](https://polybox.ethz.ch/index.php/s/Xf7wDprL3KaTj52) |
| ResNet50  | [polybox](https://polybox.ethz.ch/index.php/s/xG2Ea9xQQyjTyFL) |
