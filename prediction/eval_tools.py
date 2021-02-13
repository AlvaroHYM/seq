from os.path import basename

import utils.preprocessors as preproc

from environment import DEVICE
from environment import TRAINED_PATH
from dataloaders.datasets import get_data_list
from dataloaders.datasets import get_dataloaders
from dataloaders.embeddings import load_embeddings
from dataloaders.label_transformers import define_label_transformer
from nn.models import MonoModalModel
from logger.metrics import get_metrics
from logger.prediction import log_evaluation
from logger.prediction import log_inference
from prediction.tester import Tester
from utils.load_utils import get_pretrained
from utils.training import get_criterion
from utils.training import get_pipeline


def inference(params, pretrained):
	"""
		Perform inference over a set of data as defined in a config file.

	Args:
		params: dict of training and inference task parameters
		pretrained: str, path to .model and .conf
	"""
	model_config = params
	task_name = model_config["name"]
	desc_name = ""
	pt_name = basename(pretrained)
	desc_name += "-" + pt_name

	dataset_name = params["name"]
	datasets = {
		"test": get_data_list(dataset_name, key="test")
	}
	
	label_transformer = define_label_transformer(datasets["test"])

	tester = setup_tester(config=model_config,
		name=task_name,
		datasets=datasets,
		pretrained=pretrained,
		label_transformer=label_transformer,
		disable_cache=True)
	tester.inference()
	log_inference(tester, task_name, desc_name)


def evaluation(params, pretrained):
	"""
		Perform an evaluation step over a given set of data. Reports
		on metrics and valuable magnitudes

	Args:
		params: dict of evaluation task parameters
		pretrained: str, path to .model and .conf
	"""
	model_config = params
	task_name = model_config["name"]
	desc_name = ""
	pt_name = basename(pretrained)
	desc_name += "-" + pt_name

	dataset_name = params["name"]
	datasets = {
		"test": get_data_list(dataset_name, key="test")
	}
	
	label_transformer = define_label_transformer(datasets["test"])

	tester = setup_tester(config=model_config,
		name=task_name,
		datasets=datasets,
		pretrained=pretrained,
		label_transformer=label_transformer,
		disable_cache=True)
	tester.eval()
	log_evaluation(tester, task_name, desc_name)


def setup_tester(config, name, datasets, pretrained, label_transformer=None,
	disable_cache=False):
	"""
		Prepare everything needed to perform testing over datasets

	Args:
		config:
		name:
		datasets:
		pretrained:
		label_transformer:
		disable_cache:

	Return:
		a Tester object
	"""
	pretrained_model, pretrained_config = get_pretrained(pretrained)

	word2idx = None
	embeddings = None
	if config["embeddings_file"]:
		word2idx, idx2word, embeddings = load_embeddings(config)

	preprocessor = config["preprocessor"]
	try:
		preprocessor = getattr(preproc, preprocessor)
	except TypeError:
		preprocessor = preproc.dummy_preprocess

	loaders = get_dataloaders(datasets,
		batch_size=config["batch_size"],
		data_type=config["data_type"],
		name=name,
		preprocessor=preprocessor(),
		label_transformer=label_transformer,
		word2idx=word2idx,
		config=config)

	output_size, task = get_output_size(loaders["test"].dataset.labels)
	weights = None
	
	if embeddings is None:
		model = MonoModalModel(out_size=output_size,
			embeddings=embeddings,
			embed_dim=config["embeddings_size"],
			pretrained=pretrained_model,
			finetune=False,
			encoder_params=config["encoder_params"],
			attention_params=config["attention_params"]).to(DEVICE)
	else:
		model = MonoModalModel(out_size=output_size,
			embeddings=embeddings,
			pretrained=pretrained_model,
			finetune=False,
			embed_params=config["embeddings_params"],
			encoder_params=config["encoder_params"],
			attention_params=config["attention_params"]).to(DEVICE)

	criterion = get_criterion(task, weights)

	pipeline = get_pipeline(task, criterion)

	metrics, monitor_metric, mode = get_metrics(task)

	return Tester(model=model,
		loaders=loaders,
		task=task,
		config=config,
		pipeline=pipeline,
		metrics=metrics)


def get_output_size(labels):
	"""
		Gets the output layer's size of a model from
		the set of training labels

	Args:
		labels: list of labels

	Return:
		an int, the number of output units required
	"""
	if isinstance(labels[0], float):	# regression task
		return 1, "regression"
	
	classes = len(set(labels))
	out_size = 1 if classes == 2 else classes
	task = "binary" if classes == 2 else "multilabel"
	return out_size, task

