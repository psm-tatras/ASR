import numpy as np
import pandas as pd
import tensorflow as tf
from data_utils import *

"""
TO DO
replace hard coded hyper parameters with configuration file values
implement predict method
implement model save routine using callback , cheeckpoint etc
"""

class LJASR:
    def __init__(self):
        self.model = asr.model.get_deepasrnetwork1(input_dim=128,output_dim=29,is_mixed_precision=True)
        self.optimizer = tf.keras.optimizers.Adam(lr=1e-4,beta_1=0.9, beta_2=0.999, epsilon=1e-8 )
        self.decoder = asr.decoder.GreedyDecoder()
        self.model = asr.model.compile_model(self.model,self.optimizer)
        self.model.summary()
    
    def train_step(self,inputs,outputs,batch_size,callbacks):
        history  = self.model.fit(inputs, outputs,batch_size = batch_size, epochs=1,verbose =0)
        b_loss = history.history['loss']
        b_acc = history.history['accuracy']
        return b_loss[0],b_acc[0]

"""---------------------------------------"""


# # history = pipeline.fit_generator(train_dataset = train_data, batch_size=32, epochs=500)

# pipeline.save('./checkpoint')
