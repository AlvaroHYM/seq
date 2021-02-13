# dataloaders -- Data loading tools

## dataloaders.py

Datasets following a general torch structure. Two different standards are implemented, depending on whether your problem works on sequences of token or from sequences saved in disk in the form on `.npy` arrays.

The corresponding dataloader is selected automatically from the `data_type` field introduced in the corresponding configuration file.

## datasets.py

Includes tools to parse data from the lists contained in the `datasets/` directory. 

## embeddings.py

Helper functions related to cacheing, indexing and loading of word vectors.

## label_transformers.py

If we are tackling a multi-label classification problem, convert original string labels into integer indices, which is the preferable option in torch environments.