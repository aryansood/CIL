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
