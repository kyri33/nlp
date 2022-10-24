# imports
import numpy as np
from lib.preprocessing import get_tokens, std_standardisation
from tensorflow import keras
from tqdm import tqdm
import tensorflow as tf

SEED = 42

class kylp:

	# Create an indexed vocab and inverse vocab 
	def inspect_vocab(self, corpus):
		corpus = corpus.flatten()
		tokens = []
		for sentence in corpus:
			tokens += get_tokens(sentence)
		
		self.vocab, index = {}, 1
		self.vocab['<pad>'] = 0
		for token in tokens:
			if token not in self.vocab:
				self.vocab[token] = index
				index += 1
		print("Vocab size", len(self.vocab))

		self.inverse_vocab = {index: token for token, index in self.vocab.items()}
		self.vocab_size = len(self.vocab)

		return self.vocab, self.inverse_vocab, self.vocab_size

	def sent_to_skipgram(self, sentence, vocab_size, window_size):

		sequence = [self.vocab[word] for word in get_tokens(sentence)]

		positive_skipgrams, _ = keras.preprocessing.sequence.skipgrams(
			sequence,
			vocab_size,
			window_size=window_size,
			sampling_table = self.sampling_table if hasattr(self, 'sampling_table') else None,
			negative_samples=0.0
		)
			
		return positive_skipgrams

	def generate_training_data(self, corpus, window_size, num_ns):

		targets, contexts, labels = [], [], []
		negative_samples = 4

		self.create_vocab(corpus)

		self.sampling_table = keras.preprocessing.sequences.make_sample_table(self.vocab_size)

		# TODO Clean data

		for sentence in tqdm(corpus):
			positive_skipgrams = self.sent_to_skipgram(self, sentence, vocab_size, window_size)

			for target_word, context_word in positive_skipgrams:
				context_class = tf.expand_dims(
					tf.constant([context_word], dtype="int64"), 1
				)
				negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
					true_classes=context_class,
					num_true=1,
					num_sampled=negative_samples,
					unique=True,
					range_max=self.vocab_size,
					seed=SEED,
					name='negative_sampling'
				)
				negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
				context = tf.concat([context_class, negative_sampling_candidates], 0)
				label = tf.constant([1] + [0]*negative_samples)
				
				targets.append(target_word)
				contexts.append(context)
				labels.append(label)
		
		return targets, contexts, labels



if __name__ == "__main__":

	corpus = np.array(["animals like the park dogs cats both like the park but not so much cats"])
	window_size = 1
	
	ky = kylp()
	vocab, inverse_vocab, vocab_size = ky.create_vocab(corpus)
	
	print(corpus[0])
	skip_grams = ky.sent_to_skipgram(corpus[0], vocab_size, window_size)
	for target, context in skip_grams:
		print(f"({target}, {context}): {inverse_vocab[target]}, {inverse_vocab[context]}")

	path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

	text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))
	
	vectorise_layer = keras.layers.TextVectorization(
		standardize = std_standardisation,
		max_tokens=ky.vocab_size,
		output_mode='int',
		output_sequence_length=12
	)

	vectorise_layer.adapt(text_ds.batch(1024))
	inverse_vocab = vectorise_layer.get_vocabulary()
	print(inverse_vocab[:20])