#!/usr/bin/env bash

# Make csv of absolute paths of files contained in a raw folder.
# $1: the name of the raw folder

SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
PROJECTPATH=$( cd "$SCRIPTPATH/../../" ; pwd )

DATASET_PATH="$PROJECTPATH/data/raw/$1"
INTERIM_PATH="$PROJECTPATH/data/interim/$1"
if ! [ -d $DATASET_PATH ]; then
    echo "Given raw data folder does not exist"
    exit 2
fi
if ! [ -d $INTERIM_PATH ]; then
    echo "Given interim data folder does not exist"
    exit 2
fi

#TODO fix obious case where files don't match or some files are missing
paste -d "," <(ls $DATASET_PATH/left/* | sort) <(ls $DATASET_PATH/disparity/* | sort) <(ls $INTERIM_PATH/confidence/* | sort)
