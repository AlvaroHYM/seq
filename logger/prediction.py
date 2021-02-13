import json
import math
import os
import pandas
import pickle
import time
import torch

from environment import TRAINED_PATH, EXPERIMENT_PATH


def log_inference(tester, name, description):
	"""
		Saves on disk inference results.

	Args:
		tester: a Tester object
		name: str, name to associate results to
		description: str, name description
	"""
	for dataset, output in tester.preds.items():
		results = pandas.DataFrame.from_dict(output)
		path = os.path.join(
			EXPERIMENT_PATH, tester.config["name"] + '-' + dataset)
		with open(path + ".csv", "w") as f:
			results.to_csv(f, sep="\t", encoding='utf-8', 
				float_format='%.3f', index=False)

		with open(path + "-predictions.csv", "w") as f:
			results[["tag", "y_hat"]].to_csv(
				f, index=False, float_format='%.3f', header=False)


def log_evaluation(tester, name, description):
	"""
		Saves on disk evaluation results.

	Args:
		testet: a Tester object
		name: str, name to associate results to
		description: str, name description
	"""
	for dataset, output in tester.preds.items():
		results = pandas.DataFrame.from_dict(output)
		path = os.path.join(
			EXPERIMENT_PATH, tester.config["name"] + '-' + dataset)
		with open(path + ".csv", "w") as f:
			results.to_csv(f, sep="\t", encoding='utf-8',
				float_format='%.3f', index=False)