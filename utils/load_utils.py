import pickle
import torch

from os.path import join

from environment import TRAINED_PATH


def get_pretrained(pretrained):
	if isinstance(pretrained, list):
		pretrained_models = []
		pretrained_config = []
		for pt in pretrained:
			pt_model, pt_conf = load_pretrained_model(pt)
			pretrained_models.append(pt_model)
			pretrained_config.append(pt_conf)
		return pretrained_models, pretrained_config

	return load_pretrained_model(pretrained)


def load_pretrained_model(name):
	model_path = join(TRAINED_PATH, "{}.model".format(name))
	conf_path = join(TRAINED_PATH, "{}.conf".format(name))

	try:
		model = torch.load(model_path)
	except:
		model = torch.load(model_path, map_location=torch.device('cpu'))

	model_conf = pickle.load(open(conf_path, 'rb'))
	return model, model_conf
