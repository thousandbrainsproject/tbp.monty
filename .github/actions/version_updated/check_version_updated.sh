#!/usr/bin/env bash

while IFS= read -r changed_file
do
  echo $changed_file
  if [[ $changed_file == src/tbp/monty/__init__.py ]]; then
    echo "version_updated=true" >> $GITHUB_OUTPUT
    exit 0
  fi
done < changed_files.txt
