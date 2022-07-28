
import pickle
from tensorflow import keras
import numpy as np

if __name__ == "__main__":
	
	max_size = 8
	model = keras.models.load_model('models/model_1')
	label_eng = None
	onehot_eng = None
	label_span = None
	label_eng = None

	with open('models/label_eng', 'rb') as f:
		label_eng = pickle.load(f)
		f.close()
	with open('models/onehot_eng', 'rb') as f:
		onehot_eng = pickle.load(f)
		f.close()
	with open('models/label_span', 'rb') as f:
		label_span = pickle.load(f)
		f.close()
	with open('models/onehot_span', 'rb') as f:
		onehot_span = pickle.load(f)

	while True:
		print("Enter the text to translate ($quit$ for exit):")
		input_text = input('text: ')
		if '$quit$' in input_text:
			break

		print('input text is')
		print(input_text)

		input_text = input_text.strip().lower()
		x_val = label_eng.transform(np.array(input_text.split()))
		x_val = onehot_eng.transform(x_val.reshape(-1, 1))
		x_val = np.concatenate((x_val, np.zeros_like(x_val, shape=(max_size - x_val.shape[0], x_val.shape[1]))))

		x_val = x_val.reshape(1, x_val.shape[0], x_val.shape[1])

		translation = model.predict(x_val)

		outsent = []
		for word_oh in translation[0]:
			max_idx = np.argmax(word_oh)
			label = label_span.inverse_transform([max_idx])
			outsent.append(label[0])
		
		print(' '.join(outsent))