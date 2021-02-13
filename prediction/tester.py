import torch

from environment import TRAINED_PATH
from utils.prediction import predict


class Tester:
	"""
		Wrapper for typical testing pipelines of a DL model.
		Basically an abstraction for the whole testing process,
		namely inference and testing evaluation.

	Args:
		model: a nn.Module torch model
		loaders: dict of torch dataloaders by phase
		pipeline: a callback, output computing wrapper
		config: dict, parameters of training process
		task: str, task goal [regression|binary|multilabel]
		metrics: dict of metrics to observe
	"""
	def __init__(self, model, loaders, pipeline, config, task, metrics):
		self.model = model
		self.loaders = loaders
		self.pipeline = pipeline
		self.task = task
		self.config = config
		self.metrics = {} if not metrics else metrics

		self.running_loss = 0.0

		dataset_names = list(self.loaders.keys())
		metric_names = list(self.metrics.keys()) + ['loss']
		self.scores = {
			m: {d: [] for d in dataset_names} for m in metric_names}
		self.preds = {d: {} for d in dataset_names}
	
	def eval(self):
		"""
			Main evaluation function. Perform a metrics
			assessment over every dataset available to update
			metrics track recordings.
		"""
		for partition, loader in self.loaders.items():
			avg_loss, (y, y_hat), post, atts, tags = self.eval_loader(
				loader)
			scores = self.__calc_scores(y, y_hat)
			self.__log_scores(scores, avg_loss, partition)
			scores['loss'] = avg_loss

			for name, value in scores.items():
				self.scores[name][partition].append(value)

			self.preds[partition] = {
				'tag': tags,
				'y': y,
				'y_hat': y_hat
			}

	def inference(self):
		"""
			Main inference function. Given data loaders,
			output the main attributes from the model.
		"""
		for partition, loader in self.loaders.items():
			avg_loss, (y, y_hat), post, attentions, tags = self.eval_loader(
				loader)
			self.preds[partition] = {
				'tag': tags,
				'y': y,
				'y_hat': y_hat,
				# 'posteriors': post,
				# 'attentions': attentions
			}

	def eval_loader(self, loader):
		"""
			Evaluate over a specific dataloader

		Args:
			loader: torch.DataLoader instance
		"""
		return predict(model=self.model,
			pipeline=self.pipeline,
			dataloader=loader,
			task=self.task,
			mode="eval")


	def __calc_scores(self, y, y_hat):
		return {name: metric(y, y_hat) for \
			name, metric in self.metrics.items()}

	def __log_scores(self, scores, loss, tag):
		"""
			Display metrics on console

		Args:
			scores: dict of {metric_name: value}
			loss: float, epoch average loss over epoch samples
			tag: str, dataloader s name
		"""
		print("\t{:6s} - ".format(tag), end=" ")
		for name, value in scores.items():
			print(name, '{:.4f}'.format(value), end=", ")
		print(" Loss: {:.4f}".format(loss))