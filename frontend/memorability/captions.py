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

CAPTIONS_FILE = "text_descriptions.csv"
SCORES_FILE = "scores_v2.csv"
DEV_CAPTIONS_FILE = "dev_text_descriptions.csv"
DEV_SCORES_FILE = "dev_scores.csv"

TEST_CAPTIONS_FILE = "test_text_descriptions.csv"


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


def bow(data):
	"""
		Aggregate words within all captions for a given video ID in
		a Bag-Of-Words way.

	Args:
		data: pandas.DataFrame of videoID, url and descriptions

	Return:
		a pandas.DataFrame with 1 BOW sentence per video ID
	"""
	bow_df = []
	ids = sorted(list(set(data['video_id'])))
	for i in ids:
		subset_df = data.loc[data['video_id'] == i]
		caption = ' '.join(subset_df['description'].values)
		video_id = subset_df['video_id'].iloc[0]
		video_url = subset_df['video_url'].iloc[0]
		bow_df.append(
			(video_id, video_url, caption.lower()))

	bow_df = pandas.DataFrame(bow_df, 
		columns=['video_id', 'video_url', 'description'])
	return bow_df


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



def read_samples_file(captions_dir):
	"""
		Retrieve the sampels that are split among 2 files
	"""
	captions = pandas.read_csv(os.path.join(
		captions_dir, CAPTIONS_FILE))
	captions = pandas.concat([captions, pandas.read_csv(os.path.join(
		captions_dir, DEV_CAPTIONS_FILE))], ignore_index=True)
	scores = pandas.read_csv(os.path.join(
		captions_dir, SCORES_FILE))
	scores = pandas.concat([scores, pandas.read_csv(os.path.join(
		captions_dir, DEV_SCORES_FILE))], ignore_index=True)
	return captions, scores


def _main_():
	captions_dir = sys.argv[1]
	kfolds = int(sys.argv[2])
	captions, scores = read_samples_file(captions_dir)
	captions = bow(captions)

	test_captions = pandas.read_csv(os.path.join(
		captions_dir, TEST_CAPTIONS_FILE))
	test_captions = bow(test_captions)

	for t, term in enumerate(["long", "short"]):
		df_scores = get_score(scores, short=bool(t))
		df = pandas.merge(captions, df_scores, on=['video_id'])
		
		if kfolds > 0:
			folds = [[] for f in range(kfolds)]
			for row, sample in df.iterrows():
				fold = row % kfolds
				counter = '{:05d}'.format(row)
				folds[fold].append(
					(counter, str(sample['score']), 
						sample['description'], 
						str(sample['video_id']) + '.mp4'))
			save_folds(folds=folds, 
				to_dir=os.path.join(DATA_DIR, "MEMTEXT-" + term.upper()))

		else:
			train, val = train_test_split(df, test_size=0.3,
				random_state=1234)
			ds = []
			count = 0
			for row, sample in train.iterrows():
				counter = '{:05d}'.format(count)
				ds.append((counter, str(sample['score']), 
					sample["description"], 
					str(sample['video_id']) + '.mp4'))
				count += 1
			write_samples(os.path.join(
				DATA_DIR, 'MEMTEXT-' + term.upper(), 'training.txt'), ds)
			
			ds = []
			for row, sample in val.iterrows():
				counter = '{:05d}'.format(count)
				ds.append((counter, str(sample['score']), 
					sample["description"], 
					str(sample['video_id']) + '.mp4'))
				count += 1
			write_samples(os.path.join(
				DATA_DIR, 'MEMTEXT-' + term.upper(), 'validation.txt'), ds)
			
			ds = []
			for row, sample in test_captions.iterrows():
				counter = '{:05d}'.format(count)
				ds.append((counter, str(0.0), 
					sample["description"], 
					str(sample['video_id']) + '.mp4'))
				count += 1
			write_samples(os.path.join(
				DATA_DIR, 'MEMTEXT-' + term.upper(), 'test.txt'), ds)


if __name__ == "__main__":
	_main_()