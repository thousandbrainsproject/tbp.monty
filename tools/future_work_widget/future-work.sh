# DELETE ME
python -m tools.github_readme_sync.cli generate-index docs /tmp/index.json
python -m tools.future_work_widget.cli build /tmp/index.json tools/future_work_widget/app
http-server tools/future_work_widget/app