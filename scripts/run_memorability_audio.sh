#!/bin/bash

########################################################
#                 MEMORABILITY-AUDIO                   #
########################################################

# This script gathers the experiments carried out in the
# context of predicting memorability scores from the processed
# audio signal in the MediaEval Media Memorability
# Challenge 2020.


Help(){	# Display help
	echo
	echo "Script to automatically download the MediaEval Media "
	echo "Memorability Challenge 2020 database."
	echo
	echo "Syntax: download.sh -[r|d|c|h]"
	echo "options:"
	echo "r    Change root path to CSV lists"
	echo "d    Download directory"
	echo "c    Config file"
	echo "h    Print this help message"
	echo
}


# GET SCRIPT OPTIONS
###############################
CSV_DIR="./csv/"
DOWNLOAD_DIR="./video/"
# CONFIG="conf/base_conf.json"


while getopts ":c:d:r:h:" o; do
	case "$o" in
		r)	
			CSV_DIR=${OPTARG}
			;; # Change original CSV from_dir
		d)
			DOWNLOAD_DIR=${OPTARG}
			;;	# Download directory
		c)
			CONFIG=${OPTARG}
			;;
		h | *)	
			Help
			;; # display help
	esac
done


# DOWNLOAD VIDEO DATA
###############################
# command="python3 ./frontend/memorability/download_bot.py $CSV_DIR $DOWNLOAD_DIR"
# echo "$command"
# eval "$command"

# CONVERT VIDEO TO AUDIO
###############################
# mkdir -p "${DOWNLOAD_DIR/video/wav}"
# for file in "$DOWNLOAD_DIR"/*.mp4; do
# 	basename "$file"
# 	f="$(basename -- $file)"
# 	audio_name="${f/mp4/wav}"

# 	ffmpeg -i "$file" "${DOWNLOAD_DIR/video/wav}/$audio_name"
# done

# SETUP OFFICIAL REPO
###############################
# Make sure to modify this part if anything changes.
# First clone and setup official VGGish repository
REPO_DIR="./frontend/memorability/audioset/"
# mkdir -p "$REPO_DIR"
# git clone -C "$REPO_DIR" https://github.com/tensorflow/models.git
# pip install numpy resampy tensorflow tf_slim six soundfile
# # Download pretrained models
VGGISH_DIR="./frontend/memorability/audioset/models/research/audioset/vggish"
# curl -L "https://storage.googleapis.com/audioset/vggish_model.ckpt" \
# 	-o "$REPO_DIR/vggish_model.ckpt"
# curl -L "https://storage.googleapis.com/audioset/vggish_pca_params.npz" \
# 	-o "$REPO_DIR/vggish_pca_params.npz"


# EXTRACT AUDIOSET EMBEDDINGS
#############################
FEAT_DIR="${DOWNLOAD_DIR/video/audioset}"
mkdir -p "$FEAT_DIR"
for entry in "${DOWNLOAD_DIR/video/wav}/"*.wav; do
	basename "$entry"
	f="$(basename -- $entry)"
	tf_file="${f%.*}"

	python3 "$VGGISH_DIR/vggish_inference_demo.py" \
	--wav_file "$entry" \
	--pca_params "$VGGISH_DIR/vggish_pca_params.npz" \
	--checkpoint "$VGGISH_DIR/vggish_model.ckpt" \
	--tfrecord_file "${DOWNLOAD_DIR/video/audioset}/$tf_file"
done

# command="python3 ./frontend/memorability/audioset.py $CSV_DIR $FEAT_DIR 5"
# echo "$command"
# eval "$command"


# EXPERIMENTATION - MODEL TRAINING
#################################
# command="python3 ./train/run_train.py $CONFIG --kfolds 5"
# echo "$command"
# eval "$command"

