# conf -- Configuration files

Write a json file inside this directory. This configuration file is the cornerstone of the experimentation process. Introduce all the required information.

As important details, I would mention:
- `name`: Name of the experiment. Datasets will be sought in `datasets/<name>`. **Please make sure this field matches the dataset's name.**
- `data_type`: Either `token`for sequences of tokens as shown in datasets lists, or `path` if datasets point to array-like objects saved in disk.
- `preprocessor`: Not strictly needed, usually recommended.
- `embeddings_size`: Expected input is always a sequence in the form of either tokens or arrays (tokens can be considered as a special form of arrays in which the array is only retrieved from a lookup table after their token).

Please notice that the precise parameters needed may vary depending on the learning model and the particular problem requirements. 