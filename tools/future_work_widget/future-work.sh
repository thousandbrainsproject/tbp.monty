# DELETE ME

# TODO 
# Move the docs under documentation/future-work.
# remove the build command as there is only one command.

echo "python -m tools.github_readme_sync.cli generate-index docs /tmp/index.json"
python -m tools.github_readme_sync.cli generate-index docs /tmp/index.json

echo "python -m tools.future_work_widget.cli build /tmp/index.json tools/future_work_widget/app"
python -m tools.future_work_widget.cli build /tmp/index.json tools/future_work_widget/app

http-server tools/future_work_widget/app
