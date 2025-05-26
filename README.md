Create a conda environment, and install the dependencies inside requirements.txt.

Check example_training.py code to see how to train a model. In the begin_training_loop function choose random_split=True,
if you want a random split of your training Data. Choose random_split=False, if you want to use a custom split, in this case
the value: 
train_split=pd.read_csv("train_split.csv")["file_name"].to_list(),
val_split=pd.read_csv("val_split.csv")["file_name"].to_list()
You should put the desired name of file of training data and of the validation data. Check train_split.csv and val_split.csv on how to format the files.

During evaluation, first run test_evaluation.py, put the weight of the model you want to run, and change insert the path of you test data.

Optionally, if you want to submit run create_outputs.py, after running test_evaluation.py.
