# Benchmarks
This folder contains the configs for all our benchmark experiments. Whenever a functional change is made to the code, these need to be re-run to make sure performance is not negatively affected. If results on these experiments change, the benchmark results table needs to be updated in the corresponding pull request. You can edit the [tables here](../docs/overview/benchmark-experiments.md)

You can find more information on how to run these experiments here: https://thousandbrainsproject.readme.io/docs/running-benchmarks

## Types of Benchmarks
- *pretraining*: These configs are used for running supervised pretraining experiments to generate the models used for the following benchmark evaluation experiments. These only need to be rerun if a functional change to the way a learning module learns is introduced. We keep track of version numbers for these.
- *ycb_experiments*: These are our main benchmarks. We test on the 77 YCB objects under a bunch of different conditions. More details can be found here: https://thousandbrainsproject.readme.io/docs/benchmark-experiments
- *monty_world_experiments*: These are experiment testing Monty on real-world data (moving a patch over a 2D RGBD image taken with an iPad camera).

## Follow-up Configs
If you are trying to debug something or simply want to learn more about what is happening during an experiment you can use the `make_detailed_follow_up_configs.py` script. This script will generate a config for rerunning one or several episodes of a previous experiment with detailed logging. You can then visualize and analyze the detailed logs. We do not recommend running an entire benchmark experiment with detailed logging since the log files will become prohibitively large.