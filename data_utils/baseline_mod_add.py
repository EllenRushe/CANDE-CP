#!/usr/bin/env python
"""
MODIFIED VERSION OF THE FOLLOWING FILE BY THE FOLLOWING AUTHORS:
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection used experiment in [1].
 @author Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
 [1] Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.
"""


########################################################################
# load parameter.yaml
########################################################################
import yaml
with open("baseline.yaml") as stream: param = yaml.load(stream)
"""
The parameters are loaded as a dict-type.
# default value
base_directory : ./dataset
pickle_directory: ./pickle
model_directory: ./model
result_directory: ./result
result_file: result.yaml

feature:
  n_mels: 64
  frames : 5
  n_fft: 1024
  hop_length: 512
  power: 2.0

fit:
  compile:
    optimizer : adam
    loss : mean_squared_error
  epochs : 50
  batch_size : 512
  shuffle : True
  validation_split : 0.1
  verbose : 1
"""
########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging
logging.basicConfig(level = logging.DEBUG, filename = "baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
########################################################################


########################################################################
# import default python-library
########################################################################
import pickle
import os
import sys
import glob
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
numpy.random.seed(0) # set seed
import librosa
import librosa.core
import librosa.feature

# from import
from tqdm import tqdm

########################################################################
# version
########################################################################
__versions__ = "1.0.2"
########################################################################


########################################################################
# file I/O
########################################################################
# mkdir
def try_mkdir(dirname, silence = False):
    """
    make output directory.

    dirname : str
        directory name.
    silence : bool
        boolean setting for STD I/O.
    return : None
    """
    try:
        os.mkdir(dirname)
        if not silence:
            print("%s dir is generated"%dirname)
    except:
        if not silence:
            print("%s dir is exist"%dirname)
        else:
            pass

# pickle I/O
def save_pickle(filename, data):
    """
    picklenize the data.

    filename : str
        pickle filename
    data : free datatype
        some data will be picklenized

    return : None
    """
    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as f:
        pickle.dump(data , f)
        
def load_pickle(filename):
    """
    unpicklenize the data.

    filename : str
        pickle filename

    return : data
    """
    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# wav file Input
def file_load(wav_name, mono = False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for monoral data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr = None, mono = mono)
    except:
        logger.error( f'{"file_broken or not exists!! : {}".format(wav_name)}' )
        
def demux_wav(wav_name,  channel = 0):
    """
    demux .wav file.

    wav_name : str
        target .wav file
    channel : int
        target channel number

    return : numpy.array( float )
        demuxed monoral data

    Enabled to read multiple sampling rates.

    Enabled even one channel.
    """
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr,multi_channel_data

        return sr, numpy.array(multi_channel_data)[channel, :]

    except ValueError as f:
        logger.warning(f'{f}')
########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels = 64,
                         frames = 5,
                         n_fft = 1024,
                         hop_length = 512,
                         power = 2.0):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa (**kwargs == param["librosa"])
    sr,y=demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y = y,
                                                     sr = sr,
                                                     n_fft = n_fft,
                                                     hop_length = hop_length,
                                                     n_mels = n_mels,
                                                     power = power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vectorarray_size = len(log_mel_spectrogram[0,:]) - frames + 1

    # 05 skip too short clips
    if vectorarray_size < 1:
        return numpy.empty((0, dims), float)
    
    # 06 generate feature vectors by concatenating multiframes
    vectorarray = numpy.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[:, t : t + vectorarray_size].T
        
    return vectorarray

def list_to_vector_array(file_list, 
                         msg = "calc...",
                         n_mels = 64,
                         frames = 5, 
                         n_fft = 1024,
                         hop_length = 512,
                         power = 2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param @ tqdm.

    return : numpy.array( numpy.array( float ) )
        training dataset (when generate the validation data, this function is not used.)
        * dataset.shape = (total_dataset_size, fearture_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 loop of file_to_vectorarray
    for idx in tqdm(range(len(file_list)), desc = msg):

        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels = n_mels,
                                            frames = frames, 
                                            n_fft = n_fft,
                                            hop_length = hop_length,
                                            power = power)
        
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)

        dataset[vector_array.shape[0] * idx : vector_array.shape[0] * (idx + 1), :] = vector_array
        
    return dataset

def dataset_generator(target_dir, 
                      normal_dir_name = "normal", 
                      abnormal_dir_name = "abnormal", 
                      ext = "wav"):
    """
    target_dir : str
        base directory path of the dataset
    normal_dir_name : str (default="normal")
        directory name the normal data located in
    abnormal_dir_name : str (default="abnormal")
        directory name the abnormal data located in
    ext : str (default="wav")
        filename extension of audio files 

    return : 
        train_data : numpy.array( numpy.array( float ) )
            training dataset
            * dataset.shape = (total_dataset_size, fearture_vector_length)
        train_files : list [ str ]
            file list for training
        train_labels : list [ boolean ] 
            label info. list for training
            * normal/abnnormal = 0/1
        eval_files : list [ str ]
            file list for evaluation
        eval_labels : list [ boolean ] 
            label info. list for evaluation
            * normal/abnnormal = 0/1
    """
    logger.info("target_dir : {}".format(target_dir))

    # 01 normal list generate
    normal_files = sorted(glob.glob("{dir}/{normal_dir_name}/*.{ext}".format(
                   dir = target_dir, 
                   normal_dir_name = normal_dir_name, 
                   ext = ext))
                   )
    normal_labels = numpy.zeros(len(normal_files))
    if len(normal_files) == 0: logger.exception(f'{"no_wav_data!!"}')

    # 02 abnormal list generate
    abnormal_files = sorted(glob.glob( "{dir}/{abnormal_dir_name}/*.{ext}".format(dir = target_dir, abnormal_dir_name = abnormal_dir_name, ext = ext)))
    abnormal_labels = numpy.ones(len(abnormal_files))
    if len(abnormal_files) == 0: logger.exception(f'{"no_wav_data!!"}')
    # 03 separate train & eval
    '''MODIFICATION'''
    # train_files = normal_files[len(abnormal_files):]
    # train_labels = normal_labels[len(abnormal_files):]
    # We want to show a realistic imbalance of anomalies to normal examples. 
    train_files = normal_files[len(normal_files)//2:]
    train_labels = normal_labels[len(normal_files)//2:]
    num_val = round(len(train_files)*(int(param["validation_percentage"])/100))
    val_files = normal_files[:num_val]
    val_labels = normal_labels[:num_val]

    eval_files = numpy.concatenate((normal_files[:len(normal_files)//2], abnormal_files), axis=0)
    eval_labels = numpy.concatenate((normal_labels[:len(normal_files)//2], abnormal_labels), axis=0)
    '''-----------'''   
    logger.info("train_file num : {num}".format(num = len(train_files)))
    '''MODIFICATION'''
    logger.info("val_file num : {num}".format(num = len(val_files)))    
    '''-----------'''
    logger.info("eval_file  num : {num}".format(num = len(eval_files)))
    '''MODIFICATION'''
    return train_files, train_labels, val_files, val_labels, eval_files, eval_labels
    '''-----------'''
########################################################################

########################################################################
# main
########################################################################
if __name__ == "__main__":
    # make output directory
    try_mkdir(param["pickle_directory"])
    '''MODIFICATION'''
    try_mkdir(os.path.join(param["pickle_directory"], 'train'))
    try_mkdir(os.path.join(param["pickle_directory"], 'val'))
    try_mkdir(os.path.join(param["pickle_directory"], 'eval'))
    '''------------'''
    # load base_directory list
    dirs = sorted(glob.glob("{base}/*/*/*".format(base = param["base_directory"])))
    # loop of the base directory
    for num, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{num}/{total}] {dirname}".format(dirname = target_dir, num = num + 1, total = len(dirs)))
        # dataset param        
        db = os.path.split(os.path.split(os.path.split(target_dir)[0])[0])[1]
        machine_type = os.path.split(os.path.split(target_dir)[0])[1]
        machine_id = os.path.split(target_dir)[1]

        train_pickle = "{pickle}/train/train_{machine_type}_{machine_id}_{db}.pickle".format(
                        pickle = param["pickle_directory"], 
                        machine_type = machine_type, 
                        machine_id = machine_id, 
                        db = db
                        )
        ''' MODIFICATION'''
        val_pickle = "{pickle}/val/val_{machine_type}_{machine_id}_{db}.pickle".format(
                     pickle = param["pickle_directory"], 
                     machine_type = machine_type, 
                     machine_id = machine_id, 
                     db = db
                     )
        eval_dir = "{pickle}/eval".format(pickle = param["pickle_directory"])
        '''-------------'''
        # dataset generator
        print("============== DATASET_GENERATOR ==============")
        train_files, train_labels, val_files, val_labels, eval_files, eval_labels = dataset_generator(target_dir)         
        train_data = list_to_vector_array(train_files,
                                 msg = "generate train_dataset",
                                 n_mels = param["feature"]["n_mels"],
                                 frames = param["feature"]["frames"], 
                                 n_fft = param["feature"]["n_fft"],
                                 hop_length = param["feature"]["hop_length"],
                                 power = param["feature"]["power"])
       
        save_pickle(train_pickle, train_data)
        '''MODIFICATION'''
        val_data = list_to_vector_array(val_files,
                                 msg = "generate train_dataset",
                                 n_mels = param["feature"]["n_mels"],
                                 frames = param["feature"]["frames"],
                                 n_fft = param["feature"]["n_fft"],
                                 hop_length = param["feature"]["hop_length"],
                                 power = param["feature"]["power"])

        save_pickle(val_pickle, val_data)

        '''-------------'''
        for num, file_name in tqdm(enumerate(eval_files), total = len(eval_files)):
            try:
                # MODIFICATION ON FILE NAME. 
                eval_data = file_to_vector_array(file_name,
                                            n_mels = param["feature"]["n_mels"],
                                            frames = param["feature"]["frames"],
                                            n_fft = param["feature"]["n_fft"],
                                            hop_length = param["feature"]["hop_length"],
                                            power = param["feature"]["power"])
                '''MODIFICATION'''
                # ['', '/dataset/min6dB/valve/id_06/abnormal/00000119', 'wav'] 
                # /dataset/min6dB/valve/id_06/abnormal/00000119
                # ['', 'dataset', 'min6dB', 'valve', 'id_06', 'abnormal', '00000119'] 
                # ['min6dB', 'valve', 'id_06', 'abnormal', '00000119'] 
                # min6dB_valve_id_06_abnormal_00000119 
                eval_feature_file_name = '_'.join(file_name.split('.')[0].split('/')[-5:])
                save_pickle(os.path.join(eval_dir, eval_feature_file_name+".pickle"), eval_data) 
                '''------------'''
            except:
                logger.warning( "File broken!!: {}".format(file_name))
########################################################################
