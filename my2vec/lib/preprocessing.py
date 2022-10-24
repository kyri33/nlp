import tensorflow as tf
import string
import re

def get_tokens(sentence):
	lower = sentence.lower()
	return lower.translate(str.maketrans('', '', string.punctuation))

def std_standardisation(input_data):
	lowercase = tf.strings.lower(input_data)
	return tf.strings.regex_replace(lowercase, '[%s]' % re.escape(string.punctuation), '')