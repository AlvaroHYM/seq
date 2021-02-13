"""
	Inference pipeline

	Load a pretrained model, as well as its configuration, and
	proceed to inference over a set of data.
	The structure follows that of the training processes.

	Usage:
		python3 run_inference.py <config-file> <pretrained>

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

from eval_tools import inference


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
	parser.add_argument("pretrained",
		type=str,
		help='Path to pretrained .model and .conf files')
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

	inference(params=params,
		pretrained=args.pretrained)


if __name__ == "__main__":
	_main_()

