#!/usr/bin/env bash

while IFS= read -r changed_file
do
  echo $changed_file
  if [[ $changed_file == src/tbp/monty/__init__.py ]]; then
    echo "version_updated=true" >> $GITHUB_OUTPUT
    # grep finds the version line. It will only have two quotes. Then awk uses
    # the quotes as separators and prints the middle value, which corresponds to
    # the version inside the quotes.
    version=$(grep '__version__\s*=\s*' src/tbp/monty/__init__.py | awk -F'"' '{print $2}')
    echo "version=$version" >> $GITHUB_OUTPUT
    exit 0
  fi
done < changed_files.txt
