from model_utils import *

asr_model = LJASR()
"""
load these parameters from configuartion
"""
train_csv = '/mnt/sdc5/Work/Tatras/Common_Data/LJ Speech ASR Dataset/time_sorted_audio_list_train.csv'
batch_size = 64
EPOCHS = 100


# gather batch information
train_files,nb_batches,total = initialize_batch_generator(train_csv,batch_size)


# # train asr model
for e in range(EPOCHS):
    e_loss = 0
    e_acc = 0
    batches = batch_generator(train_files,batch_size,total)
    for b in range(nb_batches):
        train_x,train_y,audio_lens,label_lens,start,end = next(batches)
        inputs = {'the_input': train_x, 'the_labels': train_y, 'input_length': audio_lens, 'label_length': label_lens}
        # ctc output
        outputs = {'ctc': np.zeros([len(train_x)])}
        b_loss,b_acc = asr_model.train_step(inputs,outputs,batch_size,"")
        e_loss += b_loss
        e_acc += b_acc
        sys.stdout.write("\rBatch %d/%d [%d to %d] Loss %f Acc %f"%(b+1,nb_batches,start,end,b_loss,b_acc))
        sys.stdout.flush()
    avg_e_loss = e_loss / nb_batches
    avg_e_acc = e_acc / nb_batches
    print("\nEpoch %d/%d Loss %f Acc %f"%(e+1,EPOCHS,avg_e_loss,avg_e_acc))