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
from logger.training import Checkpoint
from logger.training import EarlyStop
from logger.training import log_training
from logger.training import log_fold_training
from logger.training import log_K_folds_training
from utils.load_utils import get_pretrained
from utils.training import class_weights
from utils.training import get_criterion
from utils.training import get_optimizer
from utils.training import get_pipeline
from train.trainer import Trainer


def train_model(params, pretrained=None, finetune=False, kfolds=None):
	"""
		Train and validation pipeline, wrapped up in one function.

		Args:
			params: dict of training and task parameters
			pretrained: str, path to a pretrained model and conf file
			finetune: bool, whether to retrain whole model or last layer(s)
			kfolds: int, number of folds in Cross Validation

	"""
	model_config = params

	if kfolds:
		results = []
		task_name = model_config["name"]
		desc_name = ""
		if pretrained:
			desc_name = "PT-" + pretrained
			if finetune:
				desc_name += "-FT_" + str(finetune)

		for fold in range(kfolds):
			print("*" * 30)
			print("[FOLD {} OUT OF {}]".format(fold + 1, kfolds))
			print("*" * 30)

			fold_desc = "-fold_{}_outof_{}".format(fold + 1, kfolds)
			trainer_name = task_name + desc_name + fold_desc
			fold_pretrained_model = None
			if pretrained:
				fold_pretrained_name = pretrained + fold_desc
				# TODO: geto_topK_fold_result for rootname
				# fold_pretrained_model = get_top_Kfold_rootname(
				# 	fold_pretrained_name)
				if fold_pretrained_model == None:
					raise ValueError("Cannot find pretrained model")
			
			dataset_name = params["name"]
			datasets = {
				"train": get_data_list(dataset_name, 
					key="training" + fold_desc),
				"val": get_data_list(dataset_name, 
					key="validation" +  fold_desc)
			}

			label_transformer = define_label_transformer(datasets["train"])

			trainer = setup_trainer(config=model_config,
				name=trainer_name,
				datasets=datasets,
				monitor="val",
				pretrained=pretrained,
				finetune=finetune,
				label_transformer=label_transformer,
				disable_cache=True)
			trainer.train(model_config["epochs"])
			# log_training(trainer, task_name, desc_name + fold_desc)
			results.append(log_fold_training(trainer))

		log_K_folds_training(trainer, task_name, desc_name, results)

	else:
		task_name = model_config["name"]
		desc_name = ""
		if pretrained:
			pt_name = basename(pretrained)
			desc_name += "-PT-" + pt_name
			if finetune:
				desc_name += "-FT"

		dataset_name = params["name"]
		datasets = {
			"train": get_data_list(dataset_name, key="training"),
			"val": get_data_list(dataset_name, key="validation")
		}

		label_transformer = define_label_transformer(datasets["train"])
	
		trainer = setup_trainer(config=model_config,
			name=task_name,
			datasets=datasets,
			monitor="val",
			pretrained=pretrained,
			finetune=finetune,
			label_transformer=label_transformer,
			disable_cache=True)
		
		trainer.train(model_config["epochs"])
		log_training(trainer, task_name, desc_name)


def setup_trainer(config, name, datasets, monitor="val", pretrained=None,
	finetune=False, label_transformer=None, disable_cache=False):
	"""
		Prepare everything needed for a train + validation typical pipeline.

	Args:
		config:	dict, experiment parameters
		name:	str, name of the experiment
		datasets: dict, data for every data partition (X, y)
		monitor: str, partition to watch on learning time 
		pretrained: str, path to pretrained model and conf files
		finetune: bool, whether to finetune pretrained model
		label_transformer: Label transform function
		disable_cache: Whether to activate caching (TODO)

	Return:
		a Trainer object
	"""
	pretrained_model = None
	pretrained_config = None
	if pretrained:
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

	output_size, task = get_output_size(loaders["train"].dataset.labels)
	weights = None
	if task != "regression":
		weights = class_weights(loaders["train"].dataset.labels,
			to_pytorch=True).to(DEVICE)

	if embeddings is None:
		model = MonoModalModel(out_size=output_size,
			embeddings=embeddings,
			embed_dim=config["embeddings_size"],
			pretrained=pretrained_model,
			finetune=finetune,
			encoder_params=config["encoder_params"],
			attention_params=config["attention_params"]).to(DEVICE)
	else:
		model = MonoModalModel(out_size=output_size,
			embeddings=embeddings,
			pretrained=pretrained_model,
			finetune=finetune,
			embed_params=config["embeddings_params"],
			encoder_params=config["encoder_params"],
			attention_params=config["attention_params"]).to(DEVICE)

	criterion = get_criterion(task, weights)
	parameters = filter(lambda p: p.requires_grad, model.parameters())

	optimizer = get_optimizer(parameters, lr=config["lr"], 
		weight_decay=config["weight_decay"])

	pipeline = get_pipeline(task, criterion)

	metrics, monitor_metric, mode = get_metrics(task)

	model_dir = None
	if pretrained:
		model_dir = os.path.join(TRAINED_PATH, "TL")

	checkpoint = Checkpoint(name=name, 
		model=model, 
		model_conf=config,
		monitor=monitor, 
		keep_best=True, 
		timestamp=True, 
		scorestamp=True, 
		metric=monitor_metric, 
		mode=mode, 
		base=config["base"],
		model_dir=model_dir)

	early_stopping = EarlyStop(metric=monitor_metric,
		mode=mode,
		monitor=monitor,
		patience=config["patience"],
		min_change=config["min_change"])

	return Trainer(model=model,
		loaders=loaders,
		task=task,
		config=config,
		optimizer=optimizer,
		pipeline=pipeline,
		metrics=metrics,
		checkpoint=checkpoint,
		early_stopping=early_stopping)


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













