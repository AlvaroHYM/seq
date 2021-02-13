# sequenceLearning repository

This repository provides a working bench for working with sequences of data. This is, data that naturally comes in the form of sequences of descriptors or features. For instance, words within a sentence or a series of embeddings representing the temporal evolution of an audio signal. 



## Install

After you setup a proper Python3 environment (either conda, virtualenv/venv, or any other serves), clone this repository by typing in terminal:
```
git clone https://github.com/affectivepixelsteam/sequenceLearning.git
```

Proceed to install all required dependencies:
```
cd sequenceLearning
pip3 install -r requirements.txt
```

## Structure

The structure contains the following modules:

- **conf**: json files describing an experiment. In order to make things clear and easy, please define a config file for every dataset you want to experiment on.
- **dataloaders**: Tools aimed at facilitating the Pytorch's data loading scheme from lists of data (**datasets**). Basically, scripts that allow to go from lists of data in text format to a torch-friendly data pipeline, including label transforms and vocabulary building (**embeddings**)
- **datasets**: Lists of data. For optimal performance, name the dataset directory accordingly to its corresponding configuration file.
- **embeddings**: Vocabulary lists. Helpful only if such a list of precomputed embeddings is needed.
- **frontend**: Tools aimed at building a dataset's list of data as will be saved in the **datasets** dir. For instance, one might want to compute MFCC features for an audio signal and build a dataset according to those MFCC features, rather from the raw signals. Particularly helpful in cases in which a heavy preprocessing of the data is required, since computing those transformations beforehand can ease the dataloader's processing.
- **logger**: Includes metrics definition, checkpoint and logging I/O operations.
- **nn**: Wrappers for torch-based neural models, and neural modules' definition.
- **scripts**: Bash scripts facilitating the experimentation process.
- **train**: Base functions for training streamdowns. Within it, Trainer is possibly the most importan object, since it includes the training and validation steps themselves and carries out the heaviest part of the procedure.
- **utils**: Complementary tools that can be substituted easily or that perform simple calculations and that otherwise would make more important classes prone to errors.

Additionally, the **environment.py** script defines the main paths used along the repository, as well as the device tu use (either CPU or GPU). Please make sure you have placed the required embeddings and datasets files in the locations specified in this script.

## Adapt to a custom dataset

Unless you require a new model, there are just 3 potential issues to consider and implement by yourself:

### Frontend

The format of the datasets as they will be read by the rest of the pipeline follows the following scheme:
```
<Number of the sample within corpus>    <label or score expected to learn>    	<sample, either a sequence of tokens or a path to a file>    <sample identifier>
```

Those 4 fields are separated by tabulations, every line denoting a new sample. Since the procedure required to adapt your dataset to this format is 200% problem-dependent, most likely you'll need to build your own application.

### Preprocessor

Even after you build up the dataset lists, there might be even more operations you want to perform on the original lists (mostly data augmentation routines, or filtering ones).

Therefore, implement your own preprocessor in the script `utils/preprocessors.py` and add the name of the method as argument to the `preprocessor`field in the corresponding configuration file.

This preprocessor must be defined as a wrapper function pointing to a callable. Oversimplifying, a function that returns the actual processing function. You can get some inspiration checking out the available ones.

In case you do not want to use any custom preprocessing function, a `dummy_preprocess` method is implemented. You can either set the config field as either this or `null`.

### Configuration file

Write a json file inside the `conf`. This configuration file is the cornerstone of the experimentation process. Introduce all the required information.

As important details, I would mention:
- `name`: Name of the experiment. Datasets will be sought in `datasets/<name>`.
- `data_type`: Either `token`for sequences of tokens as shown in datasets lists, or `path` if datasets point to array-like objects saved in disk.
- `preprocessor`: Not strictly needed, usually recommended.
- `embeddings_size`: Expected input is always a sequence in the form of either tokens or arrays (tokens can be considered as a special form of arrays in which the array is only retrieved from a lookup table after their token).

Please notice that the precise parameters needed may vary depending on the learning model and the particular problem requirements. 

## Repository communications

Please refer any inquiry about the code to the issues section. In case you want a feature of yours to be included in the main branch of this repository, make a pull request and we will evaluate it and provide feedback. 
# sequelizealv
# sequelizealv
# seq
# seq
