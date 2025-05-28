Create a conda environment, and install the dependencies inside requirements.txt.

Check example_training.py code to see how to train a model. \
In the begin_training_loop function choose random_split=True if you want to perform a random split of your data, to obtain a training set and validation set. \
Choose random_split=False, if you want to use a custom split, in this case
the value: \
train_split=pd.read_csv("train_split.csv")["file_name"].to_list() 
val_split=pd.read_csv("val_split.csv")["file_name"].to_list() 
\ 
should be changed.
\
Check train_split.csv and val_split.csv to see how to format the files.

To obtain evaluation, run test_evaluation.py. Put the weight of the model you want to run, and insert the path of you test data, see the comments inside the code
for further details and where to change the variables.
\

Optionally, if you want to submit run create_outputs.py, after running test_evaluation.py.

In the models sub-folder you can find the implementation of the different models we have trained.

The Unet (models/large_unet.py), Unet++ (models/unetpp.py), the Swin transformer encoder with the Unet decoder(models/unet_swin_depth_estimator.py) and the Vit transformer encoder with a Unet decoder(models/unet_vit_depth_estimator.py) were trained from scratch on the training data as presented inside train_split.csv.

The ResNet encoder plus Unet decoder used pre-trained weights for the encoder and was fine-tuned on the training data as presented inside train_split.csv. We used (torchvision.models.ResNet50_Weights) pre-trained weights.

The Segformer Encoder plus Segformer Decoder(modified by us to output a depth mask) used pre-trained weights for the encoder and was fine-tuned on the training data as presented inside train_split.csv. We used (nvidia/segformer-b5-finetuned-ade-640-640) fine-tuned weights.

The Maskformer architecture plus double convulutional layer used pre-trained weights for the Maskformer part and was fine-tuned on the training data as presented inside train_split.csv. We used (facebook/mask2former-swin-small-coco-instance) fine-tuned weights.

**Table: Depth estimation performance across different architectures**

| Model                     | Validation Loss | Training Loss | Kaggle Public Score |
|--------------------------|-----------------|----------------|---------------------|
| **U-Net Based models**   |                 |                |                     |
| Base U-Net               | 0.27101         | 0.32242        | --                  |
| UNet++                   | 0.36963         | 0.44006        | --                  |
| **ResNet Based models**  |                 |                |                     |
| ResNet                   | 0.15886         | 0.09157        | 0.16063             |
| ResNet-Transformer       | 0.17421         | 0.11357        | 0.21408             |
| **Maskformer model**     |                 |                |                     |
| mask2former-swin         | 0.12976         | 0.097775       | 0.13343             |
| **SegFormer models**     |                 |                |                     |            |
| segformer-b4-512-512     | 0.14236         | 0.11561        | 0.14048             |
| segformer-b5-640-640     | 0.11474         | 0.15951        | **0.12621**         |

> **Note:** The SegFormer based model achieved a better result compared to the other models, showing a 27.8% improvement over the ResNet based model and a 68.9% improvement over the Base-U-Net and the UNet++ model.
