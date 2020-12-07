from tqdm import tqdm
import numpy as np
import argparse
import pickle
import glob
import json
import os

def get_machine_id(file_name):
    file_meta = file_name.split('.')[0].split('/')[-1].split('_')
    #"pickle/train", machine_name, "id", id_number, snr
    # ID studture: machine_name_id (indices 1, 3 and 4)
    return '_'.join([file_meta[1], file_meta[3]])

def load_combine_files(files, id_map):
    file_data_list = []
    file_id_list = []
    for file_name in tqdm(files):
        file_data = pickle.load(open(file_name, "rb"))
        file_id_str = get_machine_id(file_name)
        file_id = id_map[file_id_str]
        file_id_array = np.full(len(file_data), file_id)
        file_data_list.append(file_data)
        file_id_list.append(file_id_array)
    file_data_concat = np.concatenate(file_data_list)
    file_id_concat = np.concatenate(file_id_list)
    return file_data_concat, file_id_concat

parser = argparse.ArgumentParser(description='Enter the pickle source directory and a target directory to write the data.')
parser.add_argument('--source_dir',
                   type=str,
                   help='Source directory where pickles are stored.')
parser.add_argument('--target_dir',
                   type=str,
                   help='Target directory where numpy files are to be saved')

args = parser.parse_args()


pickle_dir = args.source_dir
data_dir = args.target_dir 
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

if not os.path.exists(test_dir):
    os.mkdir(test_dir)

#all_test_files = sorted(glob.glob("pickle/eval_*_id_*_*.pickle"))
#all_test_files = [f for f in all_eval_files if ("files" not in f) and ("labels" not in f)]
train_files = sorted(glob.glob(os.path.join(pickle_dir, "train/train_*_id_*_*.pickle")))
val_files = sorted(glob.glob(os.path.join(pickle_dir, "val/val_*_id_*_*.pickle")))
test_files = sorted(glob.glob(os.path.join(pickle_dir, "eval/*_id_*_*.pickle")))
# test_files = sorted(glob.glob(os.path.join(pickle_dir, "eval/*_id_*_*.pickle")))
# Collect all the possible IDs from the traning set. 
machine_ids = np.array([get_machine_id(f) for f in train_files])
# There should be 8 different machine types, with 4 different IDs. 
unique_machine_id_strs = sorted(np.unique(machine_ids))
assert len(unique_machine_id_strs) == 16
# Get integer ID of each unique machine ID str representation. 
unique_machine_ids = [i for i in range(len(unique_machine_id_strs))]
# Create dictionary that maps string IDs to integer IDs. 
machine_ids_map = dict(zip(unique_machine_id_strs, unique_machine_ids))
# Save machine map as JSON in case it is needed later.
with open(os.path.join(data_dir, "machine_ids_map"), 'w') as f:
    json.dump(machine_ids_map, f)

train_data, train_machine_ids = load_combine_files(train_files, machine_ids_map)
train_random_idxs = np.random.permutation(len(train_data))
train_shuffled_data = train_data[train_random_idxs]
train_shuffled_machine_ids = train_machine_ids[train_random_idxs]
val_data, val_machine_ids = load_combine_files(val_files, machine_ids_map)

print("Writing individual contexts...")
for ID in np.unique(train_shuffled_machine_ids):
    ID_train_shuffled_data = train_shuffled_data[train_shuffled_machine_ids==ID]
    ID_val_machine_data = val_data[val_machine_ids==ID]
    ID_train_dir = os.path.join(train_dir, str(ID))
    ID_val_dir = os.path.join(val_dir, str(ID))
    if not os.path.exists(ID_train_dir): os.mkdir(ID_train_dir)
    if not os.path.exists(ID_val_dir): os.mkdir(ID_val_dir)
    np.save(os.path.join(ID_train_dir,"X.npy"), ID_train_shuffled_data)
    np.save(os.path.join(ID_val_dir, "X.npy"), ID_val_machine_data)

np.save(os.path.join(train_dir, "X.npy"), train_shuffled_data)
np.save(os.path.join(train_dir,"contexts.npy"), train_shuffled_machine_ids)
np.save(os.path.join(val_dir, "X.npy"), val_data)
np.save(os.path.join(val_dir, "contexts.npy"), val_machine_ids)


for test_file_name in tqdm(test_files):
    # Get file name
    test_machine_id_str = test_file_name.split('/')[-1]
    # Get meta data from file name
    file_snr, file_machine_name, _, file_machine_id, file_label, file_id  = test_machine_id_str.split('.')[0].split('_') 
    test_file_data = pickle.load(open(test_file_name, "rb"))
    file_label_int = 1 if file_label == "abnormal" else 0
    file_machine_id_int =  machine_ids_map["{}_{}".format(file_machine_name, file_machine_id)]
    file_test_dir = os.path.join(test_dir, "{}/{}/{}".format(file_snr, file_machine_name, file_machine_id))
    if not os.path.exists(file_test_dir):
        os.makedirs(file_test_dir)
    np.save(os.path.join(file_test_dir, "X_{}_{}_{}.npy".format(file_id, file_machine_id_int, file_label_int)), test_file_data)

