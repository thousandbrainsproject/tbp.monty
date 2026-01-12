#!/usr/bin/env bash

while IFS= read -r changed_file
do
  echo $changed_file
  if [[ $changed_file == tools/* ]] ||
     [[ $changed_file == pyproject.toml ]] ||
     [[ $changed_file == .github/workflows/tools.yml ]]; then
    echo "should_run_tools=true" >> $GITHUB_OUTPUT
    exit 0
  fi
done < changed_files.txt
