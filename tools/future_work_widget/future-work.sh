CMD1="python -m tools.github_readme_sync.cli generate-index docs /tmp/index.json"
CMD2="python -m tools.future_work_widget.cli /tmp/index.json tools/future_work_widget/app"
CMD3="http-server tools/future_work_widget/app"

echo "$CMD1"
eval "$CMD1"

echo "$CMD2"
eval "$CMD2"

eval "$CMD3"
