"""
	Make diarization files from a csv with predictions
"""

import os
import sys
import numpy
import pandas

from tqdm import tqdm

get_codename = lambda s: s.split('_')[0]
get_times = lambda s: s.split('-')[-2:]


def write_rttm(codename, rttm_dict, out_dir):
	"""
		Write the lines of a rttm format file from a diarization
		as a dictionary of lists

	Args:
		codename: str, name of the program
		rttm_dict: a dict of (k: str, val: list of interventions)
		out_dir: str, output dir

	Return:
		Nothing
	"""
	# Prepare file
	os.makedirs(out_dir, exist_ok=True)
	# Write RTTM header
	try:
		spkrs = sorted(list(set(rttm_dict.keys())))
	except:
		spkrs = sorted([x for x in rttm_dict.keys() if isinstance(x, str)])
	header_line = 'SPKR-INFO ' + codename + ' 1 <NA> <NA> <NA> unknown '
	with open(os.path.join(out_dir, codename + '.rttm'), 'a') as f:
		for sp in spkrs:
			# print(header_line + sp + ' <NA>')
			f.write(header_line + sp + ' <NA>\n')

	# Write file itself
	line = 'SPEAKER ' + codename + ' 1 '
	with open(os.path.join(out_dir, codename + '.rttm'), 'a') as f:
		for sp in sorted(list(rttm_dict.keys())):
			for x in rttm_dict[sp]:
				init = str(x[0])
				dur = str(x[1])
				# print(line + init + ' ' + dur + ' <NA> <NA> ' + sp + ' <NA>')
				f.write(
					line + init + ' ' + dur + ' <NA> <NA> ' + sp + ' <NA>\n')


def _main_():
	preds_filepath = sys.argv[1]
	df = pandas.read_csv(preds_filepath, sep='\t')
	
	# Get list of program names from preds file
	df['codename'] = df['tag'].apply(get_codename)
	codename_list = list(set(df['codename'].values))

	for n, code in tqdm(enumerate(codename_list), total=len(codename_list)):
		partial_df = df.loc[df['codename'] == code]
		spkr = 'no-spkr'
		spkr_init = 0.0
		code_rttm = dict()
		for row, token in partial_df.iterrows():
			tstamps = get_times(token['tag'])
			now = float(tstamps[0]) / 100.
			final = float(tstamps[1]) / 100.
			current_spkr = token['y_hat']
			if current_spkr != spkr and current_spkr != 'no-spkr':
				if spkr not in code_rttm.keys():
					if now != 0.0:
						code_rttm[spkr] = []
						code_rttm[spkr].append((spkr_init, now - spkr_init))
				else:
					code_rttm[spkr].append((spkr_init, now - spkr_init))
				spkr = current_spkr
				spkr_init = now
		write_rttm(code, code_rttm, 
			os.path.join(os.path.dirname(preds_filepath), 'rttm'))


def _concatenate_():
	filesdir = sys.argv[1]
	raw = []
	files = [x for x in os.listdir(filesdir) if x[0] != '.']
	files = [x for x in files if '_vad' in x]
	total = 0
	for n, file in enumerate(files):
		print(n+1, file)
		rttm = pandas.read_csv(os.path.join(filesdir, file), sep=' ',
			header=None, dtype=str, na_filter=False)
		raw.append(rttm)
		total += len(rttm)
	glob = pandas.concat(raw, ignore_index=True, axis=0)
	glob = glob.drop_duplicates()
	# print(glob)
	# print(total)
	glob.to_csv(os.path.join(filesdir, 'vad.rttm'), header=None,
		index=False, sep=' ')


def _apply_lab_():
	labs_dir = sys.argv[1]	# VAD prediction dir 
	rttm_dir = sys.argv[2]	# RTTM prediction dir

	labs_files = [f for f in os.listdir(labs_dir) if f[0] != '.']
	for labfile in tqdm(labs_files, total=len(labs_files)):
		filename = os.path.splitext(labfile)[0]
		lab = pandas.read_csv(os.path.join(labs_dir, labfile), sep=' ',
			header=None)
		rttm = pandas.read_csv(os.path.join(
			rttm_dir, labfile.replace('lab', 'rttm')), sep=' ', header=None,
			na_filter=False)
		header = rttm[rttm[0] == 'SPKR-INFO']
		rttm = rttm[rttm[0] != 'SPKR-INFO']
		rttm['keep'] = [False] * len(rttm)
		for i, speech in lab.iterrows():
			tlab_0 = speech[0]
			tlab_f = speech[1]
			for n, turn in rttm.iterrows():
				trttm_0 = float(turn[3])
				trttm_f = trttm_0 + float(turn[4])
				if tlab_0 <= trttm_f and tlab_f >= trttm_0:
					rttm['keep'].loc[n] = True
					init = max(trttm_0, tlab_0)
					rttm[3].loc[n] = str(init)
					# turn[3] = max(trttm_0, tlab_0)
					end = min(trttm_f, tlab_f)
					rttm[4].loc[n] = end - init
					# turn[4] = end - turn[3]
		rttm = rttm[rttm['keep'] == True]
		df = pandas.concat([header, rttm], ignore_index=True)
		df = df.drop('keep', axis=1)
		df.to_csv(os.path.join(rttm_dir, filename + '_vad.rttm'),
			sep=' ', header=None, index=False)

		# print(labfile, os.path.isfile(
			# os.path.join(rttm_dir, labfile.replace('lab', 'rttm'))))

if __name__ == "__main__":
	# _main_()
	_concatenate_()
	# _apply_lab_()