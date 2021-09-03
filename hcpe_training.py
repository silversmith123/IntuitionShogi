import datetime
import glob
import argparse
import itertools
import logging
import numpy as np
import tensorflow as tf
import hcpe_data_loader
import hcpe_model


def learn(hcpe_num, epoch_num):
	logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)
	log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	model = hcpe_model.CNNModel()
	model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
	learn_set = hcpe_data_loader.load_hcpe(hcpe_num)
	learn_set = list(learn_set.batch(1000000).as_numpy_iterator())
	learn_set = learn_set[0]
	y = learn_set['y']
	del learn_set['y']


	model.fit(learn_set, y, epochs=epoch_num, callbacks=[tensorboard_callback])
	model.save('./model/hcpe_CNN.pb')
	model.summary()

parser = argparse.ArgumentParser()
parser.add_argument('--hcpe_num', type=int, default=5)
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()
learn(args.hcpe_num, args.epochs)
