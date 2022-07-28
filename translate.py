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
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

def load_pairs(MAX=100):
	filepath = './spa-eng/spa.txt'
	f = open(filepath, 'r')
	english = []
	spanish = []
	count = 0
	for line in f.readlines():
		line = line.translate(str.maketrans('', '', string.punctuation)).lower()
		eng, span = line.split('\t')[0:2]
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

	model.compile(loss=categorical_crossentropy,
		optimizer=Adam(0.005), metrics=['accuracy'])

	return model

if __name__ == "__main__":

	english_sentences, spanish_sentences = load_pairs(MAX=10000)

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

	x = []
	y = []
	for i in tqdm(range(len(english_sentences))):
		eng_sent = english_sentences[i].split()
		span_sent = spanish_sentences[i].split()
		if len(eng_sent) < FINAL_MAX:
			[eng_sent.append('NULL') for i in range(FINAL_MAX - len(eng_sent))]
		if len(span_sent) < FINAL_MAX:
			[span_sent.append('NULL') for i in range(FINAL_MAX - len(span_sent))]

		x_item = label_eng.transform(np.array(eng_sent))
		x_item = onehot_eng.transform(x_item.reshape(-1, 1))
		#x_item = np.concatenate((x_item, np.zeros_like(x_item, shape=(FINAL_MAX - x_item.shape[0], x_item.shape[1]))))
		
		y_item = label_span.transform(np.array(span_sent))
		y_item = onehot_span.transform(y_item.reshape(-1, 1))
		#y_item = np.concatenate((y_item, np.zeros_like(y_item, shape=(FINAL_MAX - y_item.shape[0], y_item.shape[1]))))

		x.append(x_item)
		y.append(y_item)
	
	x = np.array(x)
	y = np.array(y)
	print(x.shape)
	print(y.shape)

	model = make_model_1(len(unique_spanish))
	#model.build(input_shape=(1, x.shape[1], x.shape[2]))
	#print(model.summary())

	model.fit(x, y, batch_size=64, validation_split=0.2, epochs=10)

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
	test_sentence = "i will win NULL NULL NULL"

	test_x = label_eng.transform(np.array(test_sentence.split()))
	test_x = onehot_eng.transform(test_x.reshape(-1, 1))
	test_x = np.concatenate((test_x, np.zeros_like(test_x, shape=(FINAL_MAX - test_x.shape[0], test_x.shape[1]))))

	test_x = test_x.reshape(1, test_x.shape[0], test_x.shape[1])
	print('test x', test_x.shape)
	test_y = model.predict(test_x)
	print('test y', test_y.shape)

	outsent = []
	for word_oh in test_y[0]:
		max_idx = np.argmax(word_oh)
		print('max', max_idx)
		label = label_span.inverse_transform([max_idx])
		print(label)
		outsent.append(label[0])
	
	print(' '.join(outsent))