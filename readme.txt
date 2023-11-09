truncate_data.py: this file is used to truncate the data in different time steps
usage: python truncate_data.py input_path output_path k(timestep)

train_model.py: used to train the model and save the output for the further use case
usage: python train_model.py input_path output_path 

hypothesis.py: used to have hypothesis test on two different datas
usage: hypothesis.py original_json predicted_json

data_retrieve.py: helper file used to retireve data for lsh.py

lsh.py: used for locality sensitive hashing
usage: lsh.py data_path

extractsCPC.py, extractsOutcome.py : used to extract CPCs and outcome from the .txt files
usage: extractsCPC.py data_path output.json
