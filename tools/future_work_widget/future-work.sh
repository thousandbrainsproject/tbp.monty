set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate tbp.monty
python -m tools.github_readme_sync.cli generate-index docs /tmp/index.json
python -m tools.future_work_widget.cli build /tmp/index.json tools/future_work_widget/app
http-server tools/future_work_widget/app