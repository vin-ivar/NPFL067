import math, functools, random
import numpy as np

class EntropyCalc:
	def __init__(self, text, prob=10, mess='none', padding=True):
		self.trigram, self.bigram, self.unigram = {}, {}, {}
		self.words = text.split("\n")
	
		temp = self.words[-60000:]
		self.train, self.test, self.dev = self.words[:-60000], temp[-20000:], temp[:-20000]
		
		# mess up
		self.charset = list(functools.reduce(lambda x, y: x.union(y), [set(i) for i in self.words]))
		if mess == 'char':
			self.words = [self.__mess_char(word, prob) for word in self.words]
		elif mess == 'word':
			self.words = [self.__mess_word(word, prob) for word in self.words]	
		else:
			self.words = self.words

		if padding:
			# pad only the start
			self.words = ["<s>"] + self.words

		self.init_dicts(self.words)
	
	def init_dicts(self, words):

		# don't really need these but for child class
		for i, j, k in zip(words, words[1:], words[2:]):
			if i not in self.trigram: self.trigram[i] = {}
			if j not in self.trigram[i]: self.trigram[i][j] = {}
			if k not in self.trigram[i][j]: self.trigram[i][j][k] = 1
			else: self.trigram[i][j][k] += 1

		for i, j in zip(words, words[1:]):
			if i not in self.bigram: self.bigram[i] = {}
			if j not in self.bigram[i]: self.bigram[i][j] = 1
			else: self.bigram[i][j] += 1

		for i in words:
			if i not in self.unigram: self.unigram[i] = 1
			else: self.unigram[i] += 1

		self.unigram["__sum__"] = sum(self.unigram.values())

	def __mess_char(self, word, prob=10):
		out = ""
		for char in word:
			if random.randint(1, 100 / prob) == 1:
				out += random.choice(self.charset)
			else:
				out += char
		return out

	def __mess_word(self, word, prob=10):
		if random.randint(1, 100 / prob) == 1:
			return random.choice(self.words)
		else:
			return word

	def show_words(self):
		print(self.words[:10])

	# P(i)
	def unigramProb(self, i):
		try: return self.unigram[i] / self.unigram["__sum__"]
		except: return 0

	# P(i, j)
	def bigramProb(self, i, j):
		return self.bigram[i][j] / self.unigram["__sum__"]

	# P(i, j, k)
	def trigramProb(self, i, j, k):
		return self.trigram[i][j][k] / self.unigram["__sum__"]

	# P(k | i, j) = c(i, j, k) / c(i, j)
	def trigramCond(self, k, i, j):
		try: return self.trigram[i][j][k] / self.bigram[i][j]
		except: return 0

	# P(j | i) = c(i, j) / c(i)
	def bigramCond(self, j, i):
		try: return self.bigram[i][j] / self.unigram[i]
		except: return 0

	def textEntropy(self):
		entropy = 0
		for i in self.bigram.keys():
			for j in self.bigram[i].keys():
				try:
					entropy -= self.bigramProb(i, j) * math.log2(self.bigramCond(j, i))
				except ValueError: pass

		return entropy

class Helpers:
	@staticmethod
	def normalise(args):
		return [i / sum(args) for i in args]

class Smoothing(EntropyCalc):
	def __init__(self, text):
		self.trigram, self.bigram, self.unigram = {}, {}, {}
		params = np.array([random.randint(1, 100) for i in range(4)])
		# params = np.array([0.25, 0.25, 0.25, 0.25])
		self.params = params / sum(params)
		self.words = text.split("\n")
		# create sets
		temp = self.words[-60000:]
		self.train, self.test, self.dev = self.words[:-60000], temp[-20000:], temp[:-20000]
		self.train = ["<ss>", "<s>"] + self.train
		self.test = ["<ss>", "<s>"] + self.test
		self.dev = ["<ss>", "<s>"] + self.dev

		super().init_dicts(self.train)

	def E(self, debug=False, alpha=0.1e-10):
		counts = np.array([0, 0, 0, 0], dtype='float64')

		for i, j, k in zip(self.dev, self.dev[1:], self.dev[2:]):
			curr = np.array([0.0, 0.0, 0.0, 0.0])

			try: curr[0] = self.params[0] / len(self.train)
			except: pass

			try: curr[1] = self.params[1] * super().unigramProb(k)
			except: pass

			try: curr[2] = self.params[2] * super().bigramCond(k, j)
			except: pass

			try: curr[3] = self.params[3] * super().trigramCond(k, i, j)
			except: pass

			curr /= sum(curr)
			counts += curr

		counts /= sum(counts)
		diff = self.params - counts
		self.params = counts

		if debug:
			print(counts)

		if (diff < alpha).all():
			return

		self.E(debug=debug, alpha=alpha)

	def smoothedProb(self, k, i, j):
		p0 = self.params[0] / len(self.train)
		p1 = self.params[1] * super().unigramProb(k)
		p2 = self.params[2] * super().bigramCond(k, j)
		p3 = self.params[3] * super().trigramCond(k, i, j)
		return p0 + p1 + p2 + p3

	def messTrigram(self, amt=10, cut=False):
		if cut:
			self.params[3] = self.params[3] * amt / 100
			self.params[0:3] = self.params[0:3] / (1 - self.params[3])
	
		else:
			diff = amt / 100 * (1 - self.params[3])
			self.params[3] += diff
			comp = 1 - self.params[3]
			self.params[0:3] = self.params[0:3] / comp

	def crossEntropy(self):
		self.E(debug=False, alpha=0.1e-5)

		print(self.params)
		# shallow copy
		orig = [i for i in self.params]

		# mess up
		for degree in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
			self.params = [o for o in orig]
			self.messTrigram(amt=degree)
			entropy = 0
			for i, j, k in zip(self.test, self.test[1:], self.test[2:]):
				entropy -= math.log2(self.smoothedProb(k, i, j))
			
			entropy /= len(self.test)
			print("{}\t{}".format(degree, entropy))

		# cut
		for degree in [90, 80, 70, 60, 50, 40, 30, 20, 10, 0]:
			self.params = [o for o in orig]
			self.messTrigram(amt=degree, cut=True)
			entropy = 0
			for i, j, k in zip(self.test, self.test[1:], self.test[2:]):
				entropy -= math.log2(self.smoothedProb(k, i, j))

			entropy /= len(self.test)
			print("{}\t{}".format(degree, entropy))
		
	def debug(self):
		print(super().trigramCond("organic", "of", "the"))


