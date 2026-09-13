---
title: Add More Wandb Logging for Unsupervised Learning
description: Improve logged information in unsupervised experiments (to wandb).
rfc: optional
estimated-scope: unknown
improved-metric: telemetry, learning
output-type: PR, monty-feature
skills: python, wandb
contributor: 
status: open
---

Our [benchmark experiments](../../overview/benchmark-experiments.md) that rely on pretrained models and only evaluate inference, already log a lot of statistics to Weights and Biases (wandb). We use these logs to update our benchmark tables and track changes in Monty's performance.

However, for the [unsupervised learning benchmarks](../../overview/benchmark-experiments.md#unsupervised-learning), we don't have good wandb logging set up and analysis of results needs to happen manually, based on the saved .csv file (see instructions [here](../../how-to-use-monty/running-benchmarks.md#where-to-find-the-results)). It would improve workflows significantly to also have the wandb logging handler send the necessary statistics to wandb during unsupervised learning and inference experiments.

> [!NOTE]
> For further details and discussion, see [this Discourse thread](https://forum.thousandbrains.org/t/interested-in-contributing-add-more-wandb-logging-for-unsupervised-learning/1136)