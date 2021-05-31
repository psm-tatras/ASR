import os,json,csv,random,sys
import librosa
import numpy as np
import pandas as pd
import deepasr as asr

# features_extractor = asr.features.FilterBanks(features_num=161, winlen=0.02,winstep=0.025,winfunc=np.hanning)
alphabet_en = asr.vocab.Alphabet(lang='en')


def load_configuration(config_file):
    with open(config_file,'r') as f:
        config = json.load(f)
    return config

# reads all files in main dir and labels, 
# filters files < 5 seconds (deep asr condition)
# creates two time sorted csv files for train and test (helps padding)
def process_original_data(config):
    # first load labels in a dict
    labels = {}
    f = open(os.path.join(config["data_root"],config["label_file"]),"r")
    reader = csv.reader(f,delimiter="|")
    for row in reader:
        labels[row[0]] = row[-1].strip("\n").replace(",", " ").upper()
    print("Labels loaded")
    f.close()
    # now read all the audios from dir
    audio_dir = os.path.join(config["data_root"],config["audio_dir"])
    print("Reading audios from %s"%audio_dir)
    info = []
    for root,sd,files in os.walk(audio_dir):
        for filename in files:
            abs_name = os.path.join(root,filename)
            y, sr = librosa.load(abs_name)
            duration = librosa.get_duration(y=y, sr=sr)
            file_key = filename.replace(".wav","")
            label = labels[file_key]
            info.append([abs_name,duration,label])
            sys.stdout.write("\r%s,%0.2f,%s"%(filename,duration,label))
            sys.stdout.flush()
            # print(filename,duration)
    # now split into train and test
    random.shuffle(info)
    nb_files = len(info)
    nb_train = nb_files - int(nb_files * config["train_test_ratio"])
    train_info = info[:nb_train]
    test_info = info[nb_train:]
    # now sort on time duration
    train_info = sorted(train_info,key=lambda x:x[1]) # as the second element is time duration
    test_info = sorted(test_info,key=lambda x:x[1]) # as the second element is time duration
    # time to write in file
    f = open(os.path.join(config["data_root"],config["time_sorted_csv"])+"_train.csv","w")
    f.write("path,transcripts\n")
    for fn,time,label in train_info:
        if time < 5: # precondition for deep asr
            f.write("%s,%s\n"%(fn,label))
    f.close()
    # and the test data
    f = open(os.path.join(config["data_root"],config["time_sorted_csv"])+"_test.csv","w")
    f.write("path,transcripts\n")
    for fn,time,label in train_info:
        if time < 5 :
            f.write("%s,%s\n"%(fn,label))
    f.close()
    print("Original data processing complete. Train and Test ready")

   
# must be called before batch_generator
def initialize_batch_generator(csv_file,batch_size):
    f = open(csv_file,"r")
    reader = csv.reader(f,delimiter=",")
    next(reader) # skips header
    total = 0
    info = []
    for row in reader:
        filename = row[0]
        label = row[-1]
        info.append([filename,label])
        total += 1
    nb_batches = int(np.ceil(total/batch_size))
    return info,nb_batches,total

# Extract Features
# Pad feature array
# Pad label array
# returns train_x,train_y,audio_lens,label_lens , 4 arrays required for CTC training
def batch_generator(info,batch_size,total):
    start = 0
    while start < total:
        end = min(total,start + batch_size)
        data = info[start:end]
        audios = []
        audio_lens = []
        transcripts = []
        label_lens = []
        T_max  = 0
        for audio_file,transcript in data:
            audio, sr = librosa.load(audio_file)
            precomputed_mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128,fmax=8000)
            feature = librosa.power_to_db(precomputed_mel)
            feature = np.transpose(feature,[1,0])
            # feature = librosa.feature.mfcc(y=audio, sr=sr)
            T = feature.shape[0]
            L = len(transcript)
            if T > L:
                if T > T_max:
                    T_max = T
                audios.append(feature)
                audio_lens.append(T)
                transcripts.append(transcript)
                label_lens.append(L)
        train_y = alphabet_en.get_batch_labels(transcripts)
        audio_lens = np.expand_dims(audio_lens,-1)
        label_lens = np.expand_dims(label_lens,-1)
        # padding train_x
        N = len(audios)
        F = audios[0].shape[-1]
        train_x = np.zeros([N,T_max,F])
        for i in range(N):
            for t in range(len(audios[i])):
                train_x[i][t] = audios[i][t]
        yield train_x,train_y,audio_lens,label_lens,start,end
        start = end



if __name__ == "__main__":
    config_file = "/mnt/sdc5/Work/Tatras/ASR Experiments/config.json"
    config = load_configuration(config_file)
    """
    run the function process_original_data(config) only once to create train and test files
    """
    process_original_data(config)
    
