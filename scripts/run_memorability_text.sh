#!/bin/bash

########################################################
#                 MEMORABILITY-TEXT                    #
########################################################

# This script gathers the experiments carried out in the
# context of predicting memorability scores from 
# caption descriptions in the MediaEval Media Memorability
# Challenge 2020.

# NOTE: Because this script is the first of its kind,
# it also serves as general recipe to structure other
# projects while writing out the rest of the documentation.

Help(){	# Display help
	echo
	echo "Script to automatically download the MediaEval Media "
	echo "Memorability Challenge 2020 database."
	echo
	echo "Syntax: download.sh -[r|c|h]"
	echo "options:"
	echo "r    Change root path to CSV lists"
	echo "c    Config file"
	echo "h    Print this help message"
	echo
}


# GET SCRIPT OPTIONS
###############################
CAPTIONS_DIR="./csv/"
DOWNLOAD_DIR="./video/"
CONFIG="conf/base_conf.json"


while getopts ":c:r:h:" o; do
	case "$o" in
		r)	
			CAPTIONS_DIR=${OPTARG}
			;; # Change original CSV from_dir
		c)
			CONFIG=${OPTARG}
			;;
		h | *)	
			Help
			;; # display help
	esac
done


# DATA PREPROCESSING - FRONTEND
###############################

# In order to work properly, we need to create our datasets
# in a particular format. Such format has the form:
# <#-of-sample>	'\t' <label> '\t' <sentence>/<path-to-sample>

# Because most of the datasets are not formatted that way
# it is highly advised to prepare a frontend proces Nonesor
# so data are converted into the specified format.

# command="python3 ./frontend/memorability/captions.py $CAPTIONS_DIR 5"
# echo "$command"
# eval "$command"

# You should find a directory within `datasets/` dir,
# with explicit fold lists of this dataset.


# EXPERIMENTATION - MODEL TRAINING
##################################

# command="python3 ./train/run_train.py $CONFIG --kfolds 5"
# echo "$command"
# eval "$command"


# INFERENCE - MAKE PREDICTIONS
###############################
# command="python3 ./frontend/memorability/captions.py $CAPTIONS_DIR 0"
# echo "$command"
# eval "$command"

# command="python3 ./train/run_train.py $CONFIG"
# echo "$command"
# eval "$command"

command="python3 ./prediction/run_inference.py $CONFIG MEMTEXT-SHORT--0.0173-2020_11_12_15_56"
echo "$command"
eval "$command"


