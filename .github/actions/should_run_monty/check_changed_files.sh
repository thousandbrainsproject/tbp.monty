#!/usr/bin/env bash

while IFS= read -r changed_file
do
  echo $changed_file
  if [[ $changed_file != .github/ISSUE_TEMPLATE/* ]] &&
     [[ $changed_file != .rsyncignore ]] &&
     [[ $changed_file != .vale/* ]] &&
     [[ $changed_file != .vscode/* ]] &&
     [[ $changed_file != docs/* ]] &&
     [[ $changed_file != rfcs/* ]] &&
     [[ $changed_file != tools/* ]] &&
     [[ $changed_file != CODE_OF_CONDUCT.md ]] &&
     [[ $changed_file != CONTRIBUTING.md ]] &&
     [[ $changed_file != LICENSE ]] &&
     [[ $changed_file != MAINTAINERS.md ]] &&
     [[ $changed_file != README.md ]]; then
    echo "should_run_monty=true" >> $GITHUB_OUTPUT
    exit 0
  fi
done < changed_files.txt
