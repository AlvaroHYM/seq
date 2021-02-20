"""
	Creation of training-test lists of x-vectors.
"""

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import os
import numpy
import pandas

from tqdm import tqdm
from rttm_converter import RttmConverter
from environment import DATA_DIR
from sklearn.model_selection import train_test_split

ROOT = "./../../DATA-ALVARO/"
print(os.path.isdir(ROOT))

WINDOW_SIZE = 5


get_name = lambda token: '_'.join([
			x for x in token.split('_')[:-1] if not x.isdigit()])
make_vec = lambda s: numpy.array(
	[float(x) for x in s[1:-2].split(' ') if x != ''])


def write_samples(file, samples):
	with open(file, 'w') as f:
			for s in samples:
				s = '\t'.join(s)
				f.write(s + '\n')


def read_xvec_file(file):
	"""Retrieve names, tokens and xvector from a xvector file."""
	data = list()
	with open(file, 'r') as f:
		for line in f:
			parts = line.split(' ' * 2)
			token = parts[0].lower()
			number = token.split('_')
			name = get_name(token)
			xvector = parts[1]
			data.append((name, token, xvector))

	return data


def retrieve_samples(dirnames):
	"""Given a list of dirnames, extract the samples' info and
	unify them into a single list of samples.

	Return:
		a list of tuple (name, token, xvector)
	"""
	out = []
	for dirname in dirnames:
		filenames = [x for x in os.listdir(os.path.join(
			ROOT, "xvector", dirname)) if x[0] != '.']
		for file in filenames:
			info = read_xvec_file(os.path.join(
				ROOT, "xvector", dirname, file))
			out.extend(info)
	return out


def merge_rttm_xvec(dirnames):
	"""
		Combine the processing of the rttm to get the 
		person at every xvector

	Return:
		a list of tuple (name, token, xvector)
	"""
	conv = RttmConverter(W=1.5, step=0.75)
	out = []
	for dirname in dirnames:
		filenames = [x for x in os.listdir(os.path.join(
			ROOT, "xvector", dirname)) if x[0] != '.']
		for file in tqdm(filenames):
			info = read_xvec_file(os.path.join(
				ROOT, "xvector", dirname, file))
			
			rttm_filepath = os.path.join(
				ROOT, "rttm", file.lower().replace('txt', 'rttm'))
			rttm = conv.run(rttm_filepath)
			
			for t, timestamp in enumerate(info):
				try:
					out.append((rttm[t, -1], timestamp[1], timestamp[2]))
				except IndexError:
					out.append(('no-spkr', timestamp[1], timestamp[2]))
			# break
	return out
def save_folds(folds, to_dir):
    """
        Save prepared lists of balanced folds datasets
    Args:
        folds: list of samples, labels and numbering
        to_dir: str, output directory
    """
    os.makedirs(to_dir, exist_ok=True)
    for f, fold in enumerate(folds):
        train = []
        val = []
        test = []
        [train.extend(i_fold) for i_fold in folds[:-2]]
        val.extend(folds[-2])
        test.extend(folds[-1])
        folds += [folds.pop(0)]
        filename = 'training-fold_{:d}_outof_{:d}.txt'.format(
            int(f + 1), len(folds))
        write_samples(os.path.join(to_dir, filename), train)
        filename = filename.replace('training', 'validation')
        write_samples(os.path.join(to_dir, filename), val)
        filename = filename.replace('validation', 'test')
        write_samples(os.path.join(to_dir, filename), test)

def make_lists(samples, export_dir, counter=0, test=False):
	"""
		Build sequences of samples and prepare for text writing
		Samples list is a list of (name, token, string xvector)

		If enrollment, 
	"""
	datalist = []
	for i, sample in tqdm(enumerate(samples), total=len(samples)):
		program = get_name(sample[1])
		name = sample[0]
		if "dev-" in name:
			name = name.replace('dev-', '')
		vec = make_vec(sample[2])
		if name != "null":
			
			# Uncomment for systems A and B
			if i - 2 < 0 or program != get_name(samples[i-2][1]):
				min2 = numpy.zeros(512)
			else:
				min2 = make_vec(samples[i-2][2])

			if i - 1 < 0 or program != get_name(samples[i-1][1]):
				min1 = numpy.zeros(512)
			else:
				min1 = make_vec(samples[i-1][2])

			if i + 1 >= len(samples) or program != get_name(samples[i+1][1]):
				more1 = numpy.zeros(512)
			else:
				more1 = make_vec(samples[i+1][2])

			if i + 2 >= len(samples) or program != get_name(samples[i+2][1]):
				more2 = numpy.zeros(512)
			else:
				more2 = make_vec(samples[i+2][2])

			mat = numpy.stack([min2, min1, vec, more1, more2])
			datalist.append(('{:09d}'.format(counter),
				name,
				os.path.join(ROOT, export_dir, sample[1] + '.npy'),
				sample[1]))
			numpy.save(os.path.join(ROOT, export_dir, sample[1]), mat)
		# print(os.path.join(ROOT, export_dir, sample[1]),
		# 	os.path.isfile(os.path.join(ROOT, export_dir, sample[1] + '.npy')))
			counter += 1
	return datalist

def _main_():
    # Directory names where xvector files are stored
    xvector_dirnames = ["dev", "enrollment-dev", "enrollment-test", "test"]

    # Intermediate enrollment sample list
    train_samples = retrieve_samples(["enrollment-dev", "enrollment-test"])
    print('Found {} enrolment-from samples'.format(len(train_samples)))
    number_count = 0
    os.makedirs(os.path.join(ROOT, "alvaro_seqs"), exist_ok=True)
    train_list = make_lists(train_samples, 'alvaro_seqs')
    os.makedirs(os.path.join(DATA_DIR, "IBERSPEECH20_alvaro"), exist_ok=True)
    # Merge processed RTTM files -> names and xvector files
    val_samples = merge_rttm_xvec(["dev"])
    print('Found {} dev-from samples'.format(len(val_samples)))
    val_list = make_lists(val_samples, 'alvaro_seqs', counter=len(train_list))
    datalist = train_list + val_list
    # train, val = train_test_split(datalist, test_size=0.3,
    #   random_state=1234, stratify=[x[1] for x in datalist])
    test_samples = merge_rttm_xvec(["test"])
    print('Found {} test-from samples'.format(len(test_samples)))
    test_list = make_lists(test_samples, 'alvaro_seqs',
        counter=len(datalist), test=False)
    datalist += test_list
    df = pandas.DataFrame(datalist,
        columns=["id", "label", "path", "token"])
    # print(df)
    if kfolds > 0:
        folds = [[] for f in range(kfolds)]
        for row, sample in tqdm(df.iterrows(), total=len(df)):
            fold = row % kfolds
            counter = '{:06d}'.format(row)
            folds[fold].append((
                counter,
                sample["label"],
                sample["path"],
                sample["token"]))
        save_folds(folds=folds,
            to_dir=os.path.join(DATA_DIR, "IBERSPEECH20_alvaro"))





if __name__ == "__main__":
	_main_()
