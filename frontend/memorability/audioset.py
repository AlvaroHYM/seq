import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

import numpy
import os
import pandas

from sklearn.model_selection import train_test_split

from environment import DATA_DIR

SCORES_FILE = "scores_v2.csv"
DEV_SCORES_FILE = "dev_scores.csv"


def get_score(data, short=True):
	"""
		Retrieve a list of memorability scores.

	Args:
		data: pandas.DataFrame with score fields
		short: bool, whether get long or short-term scores

	Return:
		a (N,) list of target values
	"""
	if short:
		data = data[['video_id', 'part_1_scores']]
		return data.rename(columns={'part_1_scores': "score"})
	data = data[['video_id', 'part_2_scores']]
	return data.rename(columns={'part_2_scores': "score"})


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
		[train.extend(i_fold) for i_fold in folds[:-1]]
		val.extend(folds[-1])

		folds += [folds.pop(0)]

		filename = 'training-fold_{:d}_outof_{:d}.txt'.format(
			int(f + 1), len(folds))
		write_samples(os.path.join(to_dir, filename), train)

		filename = filename.replace('training', 'validation')
		write_samples(os.path.join(to_dir, filename), val)


def write_samples(file, samples):
	with open(file, 'w') as f:
			for s in samples:
				s = '\t'.join(s)
				f.write(s + '\n')


def _main_():
	csv_dir = sys.argv[1]
	feat_dir = sys.argv[2]
	kfolds = int(sys.argv[3])

	scores = pandas.read_csv(os.path.join(
		csv_dir, SCORES_FILE))
	scores = pandas.concat([scores, pandas.read_csv(os.path.join(
		csv_dir, DEV_SCORES_FILE))], ignore_index=True)
	ids = list(set(scores['video_id'].values))

	train_paths = [[x, os.path.join(feat_dir, str(x) + '.npy')] for x in ids]
	train_list = set([str(x) + '.npy' for x in ids])
	df = pandas.DataFrame(train_paths, columns=['video_id', 'path'])
	train_df = pandas.merge(scores, df, on=['video_id'])

	files = set([f for f in os.listdir(feat_dir) if f[0] != '.'])
	test_files = files.difference(train_list)
	test_files = [os.path.join(feat_dir, x) for x in test_files]

	
	for t, term in enumerate(["long", "short"]):
		df_scores = get_score(scores, short=bool(t))
		df = pandas.merge(train_df, df_scores, on=['video_id'])

		if kfolds > 0:
			pass
			# TODO: ADAPT
			# folds = [[] for f in range(kfolds)]
			# for f, file in enumerate(feat_files):
			# 	filename = int(file.split('.')[0])
			# 	sample = scores[scores['video_id'] == filename].iloc[0]
			# 	fold = f % kfolds
			# 	s = sample['part_1_scores'] if term == "short" else sample['part_2_scores']
			# 	counter = '{:04d}'.format(f)
			# 	folds[fold].append(
			# 		(counter, str(s), os.path.join(feat_dir, file)))

			# save_folds(folds=folds, 
			# 	to_dir=os.path.join(DATA_DIR, "MEMAUDIO-" + term.upper()))
		else:
			df = df.sort_values(by="video_id")
			train, val = train_test_split(df, test_size=0.3, 
				random_state=1234)
			ds = []
			count = 0
			for row, sample in train.iterrows():
				counter = '{:05d}'.format(count)

				ds.append((counter, str(sample['score']),
					sample['path'], str(sample['video_id']) + '.mp4'))
				count += 1
			write_samples(os.path.join(
				DATA_DIR, 'MEMAUDIO-' + term.upper(), 'training.txt'), ds)

			ds = []
			for row, sample in val.iterrows():
				counter = '{:05d}'.format(count)
				ds.append((counter, str(sample['score']),
					sample['path'], str(sample['video_id']) + '.mp4'))
				count += 1
			write_samples(os.path.join(
				DATA_DIR, 'MEMAUDIO-' + term.upper(), 'validation.txt'), ds)

			ds = []
			for t, sample in enumerate(test_files):
				counter = '{:05d}'.format(count)
				ds.append((counter, str(0.0), 
					sample, sample.split('/')[-1]))
				count += 1
			write_samples(os.path.join(
				DATA_DIR, 'MEMAUDIO-' + term.upper(), 'test.txt'), ds)
			

if __name__ == "__main__":
	_main_()


