"""
	Landing page within the experimentation pipeline.

	Read configuration file and training parameters,
	setting up the required bench of work for later steps.

	Usage:
		python3 run_train.py <config-file> [--options]

	Options:
		--pretrained    Path to pretrained model and config file
		--finetune      Layer to re-train up to
		--k-folds       Number of folds for Cross-Validation
		-h, --help	Display help message
"""
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')


import argparse
import json
import numpy
import os
import random
import sys
import torch

from train_tools import train_model


def set_random_seed(seed=1234):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	numpy.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True


def get_args():
	parser = argparse.ArgumentParser(description='Working bench preparation',
		formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("config",
		type=str,
		help="Path to config file denoting training parameters")
	parser.add_argument("--pretrained",
		type=str,
		default=None,
		help='Path to pretrained .model and .conf files')
	parser.add_argument("--finetune",
		action="store_true",
		help="Whether to finetune a pretrained model")
	parser.add_argument("--kfolds",
		type=int,
		default=None,
		help="Number of folds in Cross-Validation")

	args = parser.parse_args()
	if args.kfolds and args.kfolds < 2:
		parser.error('K-Folds requires at least K=2 folds')

	return parser.parse_args()


def display_params(args, params):
	print("SCRIPT: " + os.path.basename(__file__))
	print('Options...')
	for arg in vars(args):
		print('  ' + arg + ': ' + str(getattr(args, arg)))
	print('-' * 30)

	print('Config-file params...')
	for key, value in params.items():
		print('  ' + key + ': ' + str(value))
	print('-' * 30)


def load_json(json_file):
	with open(json_file, 'r') as file:
		jdata = json.load(file)
	return jdata


def _main_():
	args = get_args()
	
	params = load_json(args.config)
	display_params(args, params)

	train_model(params=params,
		pretrained=args.pretrained,
		finetune=args.finetune,
		kfolds=args.kfolds)


if __name__ == "__main__":
	_main_()
