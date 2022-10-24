import matplotlib.pyplot as plt
import string
from pprint import pprint

f = open('spa-eng/spa.txt', 'r')

english = []
spanish = []

lines = f.readlines()
f.close()
for line in lines:
	line = line.translate(str.maketrans('', '', string.punctuation)).lower()
	eng, span = line.split('\t')[0:2]
	english.append(eng)
	spanish.append(span)

englens = [0 for _ in range(71)]
spanlens = [0 for _ in range(71)]

for engsent in english:
	total_words = len(engsent.split(' '))
	englens[total_words] += 1

for spansent in spanish:
	total_words = len(spansent.split(' '))
	spanlens[total_words] += 1

plt.plot(englens)
plt.savefig('eng_length.png')

plt.plot(spanlens)
plt.savefig('span_length.png')

pprint(spanlens)