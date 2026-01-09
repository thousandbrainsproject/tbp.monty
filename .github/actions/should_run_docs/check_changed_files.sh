#!/usr/bin/env bash

while IFS= read -r changed_file
do
  echo $changed_file
  if [[ $changed_file == .github/workflows/docs.yml ]] ||
     [[ $changed_file == benchmarks/* ]] ||
     [[ $changed_file == docs/* ]] ||
     [[ $changed_file == rfcs/* ]] ||
     [[ $changed_file == src/tbp/monty/__init__.py ]] ||
     [[ $changed_file == tools/github_readme_sync/* ]]; then
    echo "should_run_docs=true" >> $GITHUB_OUTPUT
    exit 0
  fi
done < changed_files.txt
