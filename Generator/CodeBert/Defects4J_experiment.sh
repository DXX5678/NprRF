#!/bin/bash

USE_TIMESTAMP=0
LOG_RESULT=1

while [ "$1" != "" ]; do
  case $1 in
    -l | --log-result )
      LOG_RESULT=1      
      ;;
    -t | --timestamp )
      USE_TIMESTAMP=1
      ;;
  esac
  shift
done

echo "Defects4J_single_hunk_Fix start"

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DEFECTS4J_DIR=$CURRENT_DIR/Defects4J_projects
PARENT_DIR=$(dirname "$CURRENT_DIR")
GRAND_DIR=$(dirname "$PARENT_DIR")
TOP=APPT/top5

echo "Creating directory 'Defects4J_projects'"
mkdir -p $DEFECTS4J_DIR
echo

if [ $USE_TIMESTAMP -eq 1 ]; then
  TIMESTAMP=`date +"%d%m%Y-%H%M"`
  DEFECTS4J_PATCHES_DIR=$CURRENT_DIR/Defects4J_patches/$TIMESTAMP
else
  DEFECTS4J_PATCHES_DIR=$CURRENT_DIR/Defects4J_patches
fi

echo "Creating directory 'Defects4J_patches'"
mkdir -p $DEFECTS4J_PATCHES_DIR/$TOP
echo

echo "Reading from Defects4J_method_singlehunk.csv"
while read -r line
do
  # Remove surrounding double quotes and split col4 by commas
  block2=${line#*\"}
  col4=${block2%\"*}
  block2=${block2#*\"}
  block1=${line%\"*}
  block1=${block1%\"*}
  block2=${block2#\,}  # Remove leading
  block1=${block1%\,}  # Remove trailing
  IFS=',', read -r col1 col2 col3 <<< "$block1"
  IFS=',', read -r col5 col6 <<< "$block2"

  BUG_PROJECT=${DEFECTS4J_DIR}/${col1}_${col2}
  mkdir -p $BUG_PROJECT
  echo "Checking out ${col1}_${col2} to ${BUG_PROJECT}"
  defects4j checkout -p $col1 -v ${col2}b -w $BUG_PROJECT &>/dev/null
  echo
  
  BUGGY_FILE_PATH=$BUG_PROJECT/$col3
  BUGGY_FILE_NAME=${BUGGY_FILE_PATH##*/}

  echo "Generating patches for ${col1}_${col2}"
  python $CURRENT_DIR/preprocess_Defects4J.py --temp_file=$CURRENT_DIR/src-test.txt --buggy_file=$BUG_PROJECT/$col3 --buggy_line=$col4
  echo

  echo "Runing the CodeBert for ${col1}_${col2}"
  python $CURRENT_DIR/run.py \
  --do_test \
  --do_defects4j \
  --model_type roberta \
  --model_name_or_path $CURRENT_DIR/codebert-base \
  --log_file_dir $CURRENT_DIR/logging/$TOP \
  --tokenizer_name $CURRENT_DIR/codebert-base  \
  --load_model_path $PARENT_DIR/saved_models/PPO_APPT_CodeBERT/checkpoint-best-reward/pytorch_model.bin \
  --test_filename $CURRENT_DIR/src-test.txt \
  --output_dir $DEFECTS4J_PATCHES_DIR/$TOP/${col1}_${col2} \
  --max_source_length 512 \
  --max_target_length 512 \
  --beam_size 5 \
  --eval_batch_size 1 \
  --buggy_file $BUG_PROJECT/$col3 \
  --buggy_line $col4
  echo

  echo "Generating diffs"
  for patch in $DEFECTS4J_PATCHES_DIR/$TOP/${col1}_${col2}/*; do
    diff -u -w $BUGGY_FILE_PATH $patch/$BUGGY_FILE_NAME > $patch/diff
  done
  echo

  echo "Running test on all patches for ${col1}_${col2}"
  python $CURRENT_DIR/validatePatch.py $DEFECTS4J_PATCHES_DIR/$TOP/${col1}_${col2} $BUG_PROJECT $BUG_PROJECT/$col3
  echo

  echo "Deleting ${BUG_PROJECT}"
  rm -rf $BUG_PROJECT
  rm $CURRENT_DIR/src-test.txt
  echo
done < $GRAND_DIR/Defects4J_method_singlehunk.csv

echo "Deleting Defects4J_projects"
rm -rf $DEFECTS4J_DIR
echo

if [ $LOG_RESULT -eq 1 ]; then
  CREATED=`find $DEFECTS4J_PATCHES_DIR/$TOP -name '*' -type d | wc -l | awk '{print $1}'`
  COMPILED=`find $DEFECTS4J_PATCHES_DIR/$TOP -name '*_compiled' | wc -l | awk '{print $1}'`
  PASSED=`find $DEFECTS4J_PATCHES_DIR/$TOP -name '*_passed' | wc -l | awk '{print $1}'`
  echo "$CREATED,$COMPILED,$PASSED,$TIMESTAMP" >> $CURRENT_DIR/logging/$TOP/CodeBert.log
fi

echo "Found $(find $DEFECTS4J_PATCHES_DIR/$TOP -name '*_passed' | wc -l | awk '{print $1}') test-suite adequate patches in total." >> $CURRENT_DIR/logging/$TOP/CodeBert.log
echo "Found passing patches for $(find $DEFECTS4J_PATCHES_DIR/$TOP -name '*_passed' -exec dirname {} \; | sort -u | wc -l | awk '{print $1}') projects" >> $CURRENT_DIR/logging/$TOP/CodeBert.log
echo "Defects4J_experiment.sh for CodeBert done" >> $CURRENT_DIR/logging/$TOP/CodeBert.log
echo
exit 0

