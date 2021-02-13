"""
	Convert a RTTM diarization labelling to a sequence
	of windows of size W, each assigned with a name tag.

	Parameters to consider are input rttm file path,
	and window size. This window denotes the time length 
	spanned by every vector representation. For instance,
	DIHARD x-vectors are span 1.5 seconds of audio, and 
	therefore, window will be 1.5 seconds.

	Args:
		W: int, window size in seconds
"""

import os
import numpy
import pandas


W = 1.5
STEP = 0.75


class RttmConverter:
	"""
		Abstraction to convert RTTM diarization files
		to a classification-friendly list.

	Args:
		filepath: str, path to RTTM file
		W: float, window time in seconds
	"""

	def __init__(self, W, step):
		self.W = W
		self.step = step

		self.df = None

	def run(self, filepath):
		"""
			Convert rttm to windowed labels

		Args:
			filepath: str, path to rttm file

		Return:
			a numpy.ndarray of elements (init, end, name)
		"""
		df = self.read_rttm(filepath)
		max_T = max(df['end'])
		converted = []
		t = 0.0
		while t < max_T:
			t_f = t + self.W
			best_name = "null"
			for row, item in df.iterrows():
				if t <= item["end"] and t_f >= item["init"]:
					# print('Participation!')
					init = item["init"]
					end = item["end"]
					ov = self.get_overlap((t, t_f), (init, end))
					best_overlap = 0.0
					# Is participant sufficientely "into" the vector?
					if ov >= 0.5 and ov > best_overlap:
						best_overlap = ov
						best_name = item["name"]
			sample = (int(float(t) * 100), int(float(t_f) * 100), best_name)
			converted.append(sample)
			t += self.step
		return numpy.array(converted)

	@staticmethod
	def read_rttm(filepath):
		"""
			Read a rttm file. Retrieve the unique names

		Args:
			filepath: str, path to rttm file

		Return:
			a pandas.DataFrame of items program, init, end, name
		"""
		df = pandas.read_csv(filepath, header=None, sep=' ')
		df = df[df[0] != 'SPKR-INFO']
		data = []
		for row, item in df.iterrows():
			program = item[1]
			init = item[3]
			end = item[3] + item[4]
			name = item[7]
			data.append((program, init, end, name))
		data_df = pandas.DataFrame(data,
			columns=["program", "init", "end", "name"])
		return data_df

	@staticmethod
	def get_overlap(a, b):
		return max(0, min(a[1], b[1]) - max(a[0], b[0]))

