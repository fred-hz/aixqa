#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters. 1 is expected"
  exit 2
fi
echo "Running configure name: $1"
export "ALLENNLP_CONFIG_NAME=$1"
export "CONFIG_FILE=configs/$ALLENNLP_CONFIG_NAME.json"
if ! [ -f "$CONFIG_FILE" ]; then
  echo "Configuration file not exist"
  exit 2
fi
echo "Load config file from $CONFIG_FILE"
LOG_FOLDER="logs/$ALLENNLP_CONFIG_NAME-$(date '+%Y-%m-%d %H:%M:%S')"
echo "Store logs to $LOG_FOLDER"
rm -rf "$LOG_FOLDER"
allennlp train "$CONFIG_FILE" -s "$LOG_FOLDER" --include-package src
