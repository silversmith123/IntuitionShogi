import datetime
import glob
import argparse
import itertools
import logging
import numpy as np
import tensorflow as tf
import hcpe_data_loader
import hcpe_model


def learn(hcpe_num, epoch_num, batch_size):
	logging.basicConfig(format='%(asctime)s:%(message)s', level=logging.DEBUG)
	log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	model = hcpe_model.CNNModel()
	model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
	#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	for learn_set, y in  hcpe_data_loader.load_hcpe(hcpe_num, batch_size):
		
		train_size = int(0.7 * batch_size)
		train_set = {}
		for key in learn_set.keys():
			train_set[key] = learn_set[key][:train_size]
		y_train = y[:train_size]

		test_size = int(0.15 * batch_size)
		test_set = {}
		for key in learn_set.keys():
			test_set[key] = learn_set[key][train_size:train_size + test_size]
		y_test = y[train_size:train_size + test_size]

		val_size = int(0.15 * batch_size)
		val_set = {}
		for key in learn_set.keys():
			val_set[key] = learn_set[key][train_size + test_size:]
		y_val = y[train_size + test_size:]

		#history = model.fit(train_set, y_train, batch_size=64, epochs=epoch_num, validation_data=(val_set, y_val), callbacks=[tensorboard_callback])
		history = model.fit(train_set, y_train, batch_size=64, epochs=epoch_num, validation_data=(val_set, y_val))
		print('\nhistory dict:', history.history)
		print('\n# Evaluate on test data')
		results = model.evaluate(test_set, y_test, batch_size=128)
		print('test loss, test acc:', results)
	model.save('./model/hcpe_CNN.pb')

parser = argparse.ArgumentParser()
parser.add_argument('--hcpe_num', type=int, default=5)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=500000)
args = parser.parse_args()
learn(args.hcpe_num, args.epochs, args.batch_size)
