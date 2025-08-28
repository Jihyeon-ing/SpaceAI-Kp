import json

import numpy as np
from PIL import Image
import os
import sklearn
import random
import glob
from dataloader import Dataloader
from models import *
from matplotlib import pyplot as plt

import IPython
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend as K
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        random.seed(self.args.seed)

        os.environ['KERAS_BACKEND'] = 'tensorflow'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except:
                pass

        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

        self.model_path = f'./models/{self.args.ex_name}/'
        os.makedirs(f"{self.model_path}", exist_ok=True)

        self.input_len = self.args.input_len
        self.n_features = self.args.n_features

#        self.model = model.model_v0417(self.input_len, self.n_features)

    def prepare_data(self):
        self.dataloader = Dataloader(self.args.flag, self.args.n_features)
        return self.dataloader

    def load_data(self):
        if self.args.flag == 'train':
            x_train, y_train = self.prepare_data().get_dataset()
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=777, shuffle=True)

            return x_train, y_train, x_val, y_val,

        else:
            x_test, y_test = self.prepare_data().get_dataset()
            return x_test, y_test

    def weighted_mse_loss(self, pred, tar):
        mse =  K.mean((pred - tar) ** 2, axis=0)
        mse_mean = K.mean(mse)
        mse_max = K.max(mse)
        loss = self.args.alpha * mse_mean + (1 - self.args.alpha) * mse_max
        return loss

    def train(self):
        x_train, y_train, x_val, y_val  = self.load_data()
        print(len(x_train))
#        x_train = x_train.reshape(-1, self.n_features * self.input_len)

        model = tsmixer(self.input_len, self.n_features, self.args.tar_len, self.args.hidden_dim, self.args.n_blocks, self.args.dropout)
        model.summary()
        optimizers = Adam(lr=self.args.lr)
        model.compile(loss='mse', optimizer=optimizers)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{self.model_path}best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)
        hist = model.fit(x_train, y_train, epochs=self.args.epochs, validation_data=[x_val, y_val], batch_size=self.args.batch_size, callbacks=[model_checkpoint])
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.title(np.argmin(hist.history['val_loss']))
        plt.show()
        plt.close()
        self.model_save(model)

    def model_save(self, model):
        model_json = model.to_json()
        with open(f"{self.model_path}model.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights(f"{self.model_path}model.h5")

        hparams = {
            'alpha': self.args.alpha,
            'hidden_dim': self.args.hidden_dim,
            'n_blocks': self.args.n_blocks,
        }
        with open(f"{self.model_path}hparams.json", "w") as json_file:
            json_file.write(json.dumps(hparams))

        print("Saved model to disk")

    def test(self):
        x_test, y_test = self.load_data()
        model_name = self.args.model_name

        # json_file = open(f'{self.model_path}model.json', 'r')
        #weight = f'{self.model_path}best_model.h5'
        weight = f'{self.model_path}model.h5'
        # loaded_model_json = json_file.read()
        # loaded_model = model_from_json(loaded_model_json)
        # json_file.close()
        # loaded_model = model_from_json(loaded_model_json, custom_objects={"iTransformerBlock": model_list.iTransformerBlock})
        loaded_model = tsmixer(self.input_len, self.n_features, self.args.tar_len, self.args.hidden_dim, self.args.n_blocks, self.args.dropout)
        loaded_model.summary()
        loaded_model.load_weights(weight)

        loss = 'mse'
        optimizer = Adam(lr=0.0001)
        loaded_model.compile(optimizer=optimizer, loss=loss)

        pred = loaded_model.predict(x_test, batch_size=1, verbose=1)

        savepath = f'./npy result/{self.args.ex_name}'
        os.makedirs(savepath, exist_ok=True)

        np.save(f'{savepath}/pred.npy', pred)


