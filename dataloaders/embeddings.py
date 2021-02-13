import numpy
import pickle

from os.path import exists, join, split, splitext

from environment import EMBEDDINGS_PATH


def load_embeddings(config):
	word_vectors = join(EMBEDDINGS_PATH, "{}.txt".format(
		config["embeddings_file"]))
	word_vector_size = config["embeddings_size"]
	print("  Loading word embeddings from vocabulary...")
	return load_word_vectors(word_vectors, word_vector_size)


def file_cache_name(file):
	"""
		Get cache file name from file basename
	"""
	head, tail = split(file)
	filename, ext = splitext(tail)
	return join(head, filename + ".p")


def load_cache_word_vectors(file):
	with open(file_cache_name(file), 'rb') as file:
		return pickle.load(file)


def write_cache_word_vectors(file, data):
	"""
		Write out a cache file optimized for later readings.

	Args:
		file: str, file name
		data: tuple of (word2idx, idx2word, embeddings)
	"""
	with open(file_cache_name(file), 'wb') as f:
		pickle.dump(data, f)


def load_word_vectors(file, dim):
	"""
		Read in word vectors from a vocabulary text file

	Args:
		file: str, name of the vocabulary file
		dim: int, embedding size

	Return:
		word2idx: dict, mapping word to index
		idx2word: dict, mapping index to word
		embeddings: numpy.ndarray, word embeddings matrix
	"""
	try:
		cache = load_cache_word_vectors(file)
		print('  Successfully loaded word embeddings from cache')
		return cache
	except OSError:
		print("  Did not find embeddings cache file {}".format(file))

	if exists(file):
		print("  Indexing file {}...".format(file))

		word2idx = {}
		idx2word = {}
		embeddings = []

		# We reserve first embeddings as a zero-padding
		# embedding with idx=0
		embeddings.append(numpy.zeros(dim))
		# Does the embeddings file have header?
		header = False

		with open(file, "r", encoding="utf-8") as f:
			for i, line in enumerate(f):
				if i == 0:
					if len(line.split()) < dim:
						header = True
						continue
				values = line.split(" ")
				word = values[0]
				try:
					vector = numpy.asarray(values[1:], dtype='float32')
				except ValueError:
					vector = numpy.array([float(x) for x in values[1:-1]])
					assert len(vector) == dim

				index = i - 1 if header else i

				idx2word[index] = word
				word2idx[word] = index
				embeddings.append(vector)

			# Add an <unk> token for OOV words
			if "<unk>" not in word2idx:
				idx2word[len(idx2word) + 1] = "<unk>"
				word2idx["<unk>"] = len(word2idx) + 1
				embeddings.append(
					numpy.random.uniform(low=-0.05, high=0.05, size=dim))

			print('Embeddings sizes found: ', set([
				len(x) for x in embeddings]))
			print("Found {} word vectors.".format(len(embeddings)))
			embeddings = numpy.array(embeddings, dtype='float32')

		write_cache_word_vectors(file, (word2idx, idx2word, embeddings))
		return word2idx, idx2word, embeddings

	else:
		raise OSError("{} not found!".format(file))

