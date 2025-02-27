#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MODEL_TYPE="CodeT5"
PATCH_DIR=$CURRENT_DIR/CodeT5/Defects4J_patches_before

DEFECTS4J_DIR=$CURRENT_DIR/Defects4J_projects
echo "Creating directory 'Defects4J_projects'"
mkdir -p $DEFECTS4J_DIR
echo
OUTPUT_DIR=$CURRENT_DIR/${MODEL_TYPE}_Eva
echo "Creating directory '${MODEL_TYPE}'_Eva"
mkdir -p $OUTPUT_DIR
echo

echo "Reading from Defects4J_method_singlehunk.csv"
while read -r line
do
  # Remove surrounding double quotes and split col4 by commas
  block2=${line#*\"}
  block2=${block2#*\"}
  block1=${line%\"*}
  block1=${block1%\"*}
  block2=${block2#\,}  # Remove leading
  block1=${block1%\,}  # Remove trailing
  IFS=',', read -r col1 col2 col3 <<< "$block1"

  BUG_PROJECT=${DEFECTS4J_DIR}/${col1}_${col2}
  mkdir -p $BUG_PROJECT
  echo "Checking out ${col1}_${col2} to ${BUG_PROJECT}"
  defects4j checkout -p $col1 -v ${col2}b -w $BUG_PROJECT &>/dev/null
  echo

  BUGGY_FILE_PATH=$BUG_PROJECT/$col3
  PATCH_PROJECT=$PATCH_DIR/${col1}_${col2}
  for PATCH in "$PATCH_PROJECT"/*; do
    if [[ "$PATCH" == *"passed" ]]; then
      PATCH_FILE="$PATCH"
      for FILE in "$PATCH"/*; do
        if [[ "$FILE" == *".java" ]]; then
          PATCH_FILE="$FILE"
        fi
      done

      if [[ -f $PATCH_FILE ]]; then
        echo "Detecting ${PATCH_FILE}"
        TITLE=$(basename "$(dirname "$PATCH_FILE")")
        cp "$PATCH_FILE" "$BUGGY_FILE_PATH"
        echo "Create database ${MODEL_TYPE}_${col1}_${col2} for ${PATCH}"
        if find "$BUG_PROJECT" -type f -name build.xml | grep -q build.xml; then
          FIND_BUILD_FILE=$(find "$BUG_PROJECT" -type f -name build.xml)
          codeql database create $CURRENT_DIR/${MODEL_TYPE}_${col1}_${col2} --language=java --source-root=$BUG_PROJECT --command="ant -f ${FIND_BUILD_FILE}"
        elif find "$BUG_PROJECT" -type f -name pom.xml | grep -q pom.xml; then
          FIND_BUILD_FILE=$(find "$BUG_PROJECT" -type f -name pom.xml)
          codeql database create $CURRENT_DIR/${MODEL_TYPE}_${col1}_${col2} --language=java --source-root=$BUG_PROJECT --command="mvn clean install -f ${FIND_BUILD_FILE}"
        else
          FIND_BUILD_FILE=$(find "$BUG_PROJECT" -type f -name build.gradle)
          codeql database create $CURRENT_DIR/${MODEL_TYPE}_${col1}_${col2} --language=java --source-root=$BUG_PROJECT --command="gradle --no-daemon clean test -f ${FIND_BUILD_FILE}"
        fi
        echo
        echo "Analyze database ${MODEL_TYPE}_${col1}_${col2} for ${PATCH}"
        codeql database analyze --threads 16 $CURRENT_DIR/${MODEL_TYPE}_${col1}_${col2} /home/dxx/Codeql/ql/java/ql/src/codeql-suites/java-security-and-quality.qls --format=csv --output=$OUTPUT_DIR/${col1}_${col2}_${TITLE}.csv
        echo
        echo "Delete database ${MODEL_TYPE}_${col1}_${col2} for ${PATCH}"
        rm -rf $CURRENT_DIR/${MODEL_TYPE}_${col1}_${col2}
        echo
      fi
    fi
  done

  echo "Deleting ${BUG_PROJECT}"
  rm -rf $BUG_PROJECT
  echo
done < $CURRENT_DIR/Defects4J_method_singlehunk.csv

echo "Deleting Defects4J_projects"
rm -rf $DEFECTS4J_DIR
echo
