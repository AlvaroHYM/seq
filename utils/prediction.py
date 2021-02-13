import collections
import numpy


def _get_predictions(posteriors, task):
	"""
		Compute predicted labels from posteriors

	Args:
		posteriors: (N, out_size) np.ndarray of posteriors
		task: str, goal to seek [regression|binary|multilabel]

	Return:
		a (N,) np.ndarray of output predictions
	"""
	if task in ["binary", "multilabel"]:
		if posteriors.shape[1] > 1:
			predicted = numpy.argmax(posteriors, 1)
		else:
			predicted = numpy.clip(numpy.sign(posteriors),
				a_min=0, a_max=None)

	elif task == "regression":
		predicted = posteriors

	else:
		raise ValueError("Task not defined. Cannot compute preds")

	return predicted


def predict(model, pipeline, dataloader, task, mode="eval"):
	"""
		Pass a dataloader to the model and retrieve predictions

	Args:
		model:
		pipeline:
		dataloader:
		task:
		mode:

	Return:
		a int, average loss of over data
		a tuple of (N,) lists of ground truth and predicted labels
		a (N, out_size) np.ndarray of model s posteriors
		a (N, attention_dim) np.ndarray of attention values
	"""
	model.eval()
	if mode != "eval":
		model.train()

	posteriors = []
	y_hat = []
	y = []
	attentions = []
	tags = []
	total_loss = 0

	for i_batch, sample_batched in enumerate(dataloader, 1):
		outputs, labels, atts, loss = pipeline(model, sample_batched)
		batch_tags = sample_batched[-1]

		if loss is not None:
			total_loss += loss.item()

		posts_ = outputs.data.cpu().numpy()	# posteriors

		if len(posts_.shape) == 1:
			predicted = _get_predictions(numpy.expand_dims(
				posts_, acis=0), task)
		else:
			predicted = _get_predictions(posts_, task)

		labels = labels.data.cpu().numpy().squeeze().tolist()
		predicted = predicted.squeeze().tolist()
		posts_ = posts_.squeeze().tolist()
		if atts is not None:
			atts = atts.data.cpu().numpy().squeeze().tolist()

		if not isinstance(labels, collections.Iterable):
			labels = [labels]
			predicted = [predicted]
			posts_ = posts_
			if atts is not None:
				atts = [atts]

		if task != "regression":
			label_transformer = dataloader.dataset.label_transformer
			if label_transformer is not None:
				labels = numpy.array([
					label_transformer.inverse(x) for x in labels])
				predicted = numpy.array(
					[label_transformer.inverse(x) for x in predicted])

		y.extend(labels)
		y_hat.extend(predicted)
		posteriors.extend(posts_)
		if atts is not None:
			attentions.extend(atts)
		tags.extend(batch_tags)

	avg_loss = total_loss / i_batch
	return avg_loss, (y, y_hat), posteriors, attentions, tags
				

