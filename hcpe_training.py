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
	data_size = len(list(learn_set.as_numpy_iterator()))

	train_size = int(0.7 * data_size)
	train_set = learn_set.take(train_size)
	train_set = list(train_set.batch(train_size).as_numpy_iterator())[0]
	y_train = train_set['y']
	del train_set['y']

	test_size = int(0.15 * data_size)
	test_set = learn_set.take(train_size)
	test_set = list(test_set.batch(train_size).as_numpy_iterator())[0]
	y_test = test_set['y']
	del test_set['y']

	val_size = int(0.15 * data_size)
	val_set = learn_set.take(val_size)
	val_set = list(val_set.batch(val_size).as_numpy_iterator())[0]
	y_val = val_set['y']
	del val_set['y']

	history = model.fit(train_set, y_train, batch_size=64, epochs=epoch_num, validation_data=(val_set, y_val), callbacks=[tensorboard_callback])
	model.save('./model/hcpe_CNN.pb')
	print('\nhistory dict:', history.history)
	print('\n# Evaluate on test data')
	results = model.evaluate(test_set, y_test, batch_size=128)
	print('test loss, test acc:', results)

parser = argparse.ArgumentParser()
parser.add_argument('--hcpe_num', type=int, default=5)
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()
learn(args.hcpe_num, args.epochs)
