from enum import unique
import pickle
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import string
from tqdm import tqdm
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import random

BATCH_SIZE = 64

def load_pairs(MAX=100):
	filepath = './spa-eng/spa.txt'
	f = open(filepath, 'r')
	english = []
	spanish = []
	count = 0
	lines = f.readlines()
	random.shuffle(lines)
	f.close()
	for line in lines:
		line = line.translate(str.maketrans('', '', string.punctuation)).lower()
		eng, span = line.split('\t')[0:2]

		if len(eng.split()) > 17 or len(span.split()) > 17:
			continue

		english.append(eng)
		spanish.append(span)
		
		count += 1
		if count >= MAX:
			break
	
	return english, spanish

def make_encoder(words):
	label_encoder = LabelEncoder()
	int_encoded = label_encoder.fit_transform(words)
	onehot_encoder = OneHotEncoder(sparse=False)
	onehot_encoder.fit(int_encoded.reshape(-1, 1))
	return label_encoder, onehot_encoder

def get_max(sentences):
	max_len = 0
	for sentence in sentences:
		senlen = len(sentence.split())
		if senlen > max_len:
			max_len = senlen

	return max_len

def make_model_1(output_size):

	model = models.Sequential()
	model.add(layers.GRU(256, return_sequences=True))
	model.add(layers.TimeDistributed(layers.Dense(1024, activation='relu')))
	model.add(layers.Dropout(0.5))
	model.add(layers.TimeDistributed(layers.Dense(output_size, activation='softmax')))

	model.compile(loss=sparse_categorical_crossentropy,
		optimizer=Adam(0.005), metrics=['accuracy'])

	return model

def make_model_2(english_vocab_size, output_size, max_size):

	model = models.Sequential()
	model.add(layers.Embedding(english_vocab_size, 256, input_length=max_size))
	model.add(layers.GRU(256, return_sequences=True))
	model.add(layers.TimeDistributed(layers.Dense(1024, activation='relu')))
	model.add(layers.Dropout(0.5))
	model.add(layers.TimeDistributed(layers.Dense(output_size, activation='softmax')))

	model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(0.005), metrics=['accuracy'])

	return model

def make_model_3(output_size):
	model = models.Sequential()
	model.add(layers.Bidirectional(layers.GRU(128, return_sequences=True)))
	model.add(layers.TimeDistributed(layers.Dense(1024, activation='relu')))
	model.add(layers.Dropout(0.5))
	model.add(layers.TimeDistributed(layers.Dense(output_size, activation='softmax')))

	model.compile(loss=sparse_categorical_crossentropy, 
		optimizer=Adam(0.003), metrics=['accuracy'])

	return model

def make_model_4(output_size, seq_length):
	model = models.Sequential()
	# encoder
	model.add(layers.GRU(256, go_backwards=True))
	model.add(layers.RepeatVector(seq_length))

	# decoder
	model.add(layers.GRU(256, return_sequences=True))
	model.add(layers.TimeDistributed(layers.Dense(1024, activation='relu')))
	model.add(layers.Dropout(0.5))
	model.add(layers.TimeDistributed(layers.Dense(output_size, activation='softmax')))

	model.compile(loss=sparse_categorical_crossentropy,
		optimizer=Adam(0.001), metrics=['accuracy'])
	
	return model

def make_model_5(english_vocab_size, output_size, max_size):
	model = models.Sequential()

	# Embedding
	model.add(layers.Embedding(english_vocab_size, 128, input_length=max_size))
	#model.add(layers.GRU(128, return_sequences=True))

	# Encoder
	model.add(layers.Bidirectional(layers.GRU(128)))
	model.add(layers.RepeatVector(max_size))

	# Decoder
	model.add(layers.Bidirectional(layers.GRU(128, return_sequences=True)))
	model.add(layers.TimeDistributed(layers.Dense(512, activation='relu')))

	model.add(layers.Dropout(0.5))
	model.add(layers.TimeDistributed(layers.Dense(output_size, activation='softmax')))

	model.compile(loss=sparse_categorical_crossentropy, optimizer=Adam(0.003), metrics=['accuracy'])

	return model

def generate_preprocess(english_sentences, spanish_sentences):
	i = 0
	while True:
		x_batch = []
		y_batch = []
		for _ in range(BATCH_SIZE):
			if i >= len(english_sentences):
				i = 0
			eng_sent = english_sentences[i].split()
			span_sent = spanish_sentences[i].split()
			if len(eng_sent) < FINAL_MAX:
				[eng_sent.append('NULL') for i in range(FINAL_MAX - len(eng_sent))]
			if len(span_sent) < FINAL_MAX:
				[span_sent.append('NULL') for i in range(FINAL_MAX - len(span_sent))]

			x_item = label_eng.transform(np.array(eng_sent))
			#x_item = onehot_eng.transform(x_item.reshape(-1, 1))
			
			y_item = label_span.transform(np.array(span_sent))
			#y_item = onehot_span.transform(y_item.reshape(-1, 1))

			x_batch.append(x_item)
			y_batch.append(y_item)
			i += 1
		
		x_batch = np.array(x_batch)
		y_batch = np.array(y_batch)
		#print('ybatch', y_batch.shape)
		yield x_batch, y_batch

if __name__ == "__main__":

	english_sentences, spanish_sentences = load_pairs(MAX=1000000)

	unique_english = ['NULL']
	unique_spanish = ['NULL']
	[unique_english.append(word) for sentence in english_sentences for word in sentence.split() if word not in unique_english]
	[unique_spanish.append(word) for sentence in spanish_sentences for word in sentence.split() if word not in unique_spanish]
	print('unique english', len(unique_english))
	print('unique spanish', len(unique_spanish))

	label_eng, onehot_eng = make_encoder(unique_english)
	label_span, onehot_span = make_encoder(unique_spanish)

	MAX_ENG = get_max(english_sentences)
	MAX_SPAN = get_max(spanish_sentences)
	print('max eng', MAX_ENG)
	print('max span', MAX_SPAN)
	FINAL_MAX = MAX_ENG if MAX_ENG > MAX_SPAN else MAX_SPAN
	print("final", FINAL_MAX)

	#model = make_model_1(len(unique_spanish))
	#model = make_model_2(len(unique_english), len(unique_spanish), FINAL_MAX)
	#model = make_model_3(len(unique_spanish))
	#model = make_model_4(len(unique_spanish), FINAL_MAX)
	model = make_model_5(len(unique_english), len(unique_spanish), FINAL_MAX)
	#model.build(input_shape=(1, x.shape[1], x.shape[2]))
	#print(model.summary())

	model.fit(generate_preprocess(english_sentences, spanish_sentences), epochs=7, steps_per_epoch = len(english_sentences) // BATCH_SIZE)

	model.save('models/model_1')

	with open('models/label_eng', 'wb') as f:
		pickle.dump(label_eng, f)
		f.close()
	with open('models/onehot_eng', 'wb') as f:
		pickle.dump(onehot_eng, f)
		f.close()
	with open('models/label_span', 'wb') as f:
		pickle.dump(label_span, f)
		f.close()
	with open('models/onehot_span', 'wb') as f:
		pickle.dump(onehot_span, f)
		f.close()

	#test_sentence = 'i am going to the shop now'
	test_sentence = "im glad you warned me before it was too late"

	test_sentence = test_sentence.split()
	if len(test_sentence) < FINAL_MAX:
		[test_sentence.append('NULL') for i in range(FINAL_MAX - len(test_sentence))]

	test_x = label_eng.transform(np.array(test_sentence))
	#test_x = onehot_eng.transform(test_x.reshape(-1, 1))

	test_x = test_x.reshape(1, test_x.shape[0], test_x.shape[1])
	test_y = model.predict(test_x)
	outsent = []
	for word_oh in test_y[0]:
		max_idx = np.argmax(word_oh)
		label = label_span.inverse_transform([max_idx])
		outsent.append(label[0])
	
	print(' '.join(outsent))
