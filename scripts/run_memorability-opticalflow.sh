#!/bin/bash

########################################################
#               MEMORABILITY-OPTICAL FLOW              #
########################################################

# This script gathers the experiments carried out in the
# context of predicting memorability scores from 
# optical flows computed from the frames
# of the videos in the MediaEval Media Memorability
# Challenge 2020.

Help(){	# Display help
	echo
	echo "Script to automatically perform experiments on the MediaEval Media "
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

# CONVERT TO FRAMES
###############################
FRAMES_DIR="${DOWNLOAD_DIR/video/frames}"
for file in "$DOWNLOAD_DIR"*.mp4; do
	basename "$file"
	f="$(basename -- $file)"
	mkdir -p "$FRAMES_DIR/$f/"
	ffmpeg -i "$file" -vf "fps=3" "$FRAMES_DIR/$f/%04d.jpeg"
done

# PERFORM LITEFLOWNET OR OTHER OPTICAL FLOW COMPUTATION
#######################################################
FEAT_DIR="${DOWNLOAD_DIR/video/flo}"

# CONVERT IMAGE TO EMBEDDINGS
#############################
command="python3 ./frontend/memorability/liteflownet.py $CSV_DIR $FEAT_DIR"
echo "$command"
eval "$command"


