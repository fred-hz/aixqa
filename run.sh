rm -rf output
allennlp train configs/train_on_sample.json -s ./output --include-package src