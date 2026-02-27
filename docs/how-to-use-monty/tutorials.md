---
title: Tutorials
---
These tutorials will help you get familiar with Monty. Topics and experiments increase in complexity as the tutorials progress, so we recommend going through them in order. You can also learn about contributing tutorials [here](../contributing/contributing-tutorials.md).

# Getting Started
The following tutorials are aimed at getting you familiar with the Monty code infrastructure, running experiments, and the general algorithm. They don't require you to write custom code, instead providing code for experiment configurations that you can follow along with.

- [Running Your First Experiment](tutorials/running-your-first-experiment.md): Demonstrates basic layout of an experiment config and outlines experimental training loops.
- [Pretraining a Model](tutorials/pretraining-a-model.md): Shows how to configure an experiment for supervised pretraining.
- [Running Inference with a Pretrained Model](tutorials/running-inference-with-a-pretrained-model.md): Demonstrates how to load a pretrained model and use it to perform object and pose recognition.
- [Unsupervised Continual Learning](tutorials/unsupervised-continual-learning.md): Demonstrates how to use Monty for unsupervised continual learning to learn new objects from scratch.
- [Multiple Learning Modules](tutorials/multiple-learning-modules.md): Shows how to perform pretraining and inference using multiple sensor modules and multiple learning modules.

# Advanced
The following tutorials assume that you are already a bit more familiar with Monty. They are aimed at people who understand our approach and now want to customize it for their own experiments and applications. We provide code and configurations to follow along but the tutorials also provide instructions for writing your own custom code.
- [Using Monty in a Custom Application](tutorials/using-monty-in-a-custom-application.md) [No Habitat Required]: Guides you through the process of letting Monty learn and perform inference in an environment of your choice. Talks about requirements, what needs to be defined, and which classes to customize.
- [Using Monty for Robotics](tutorials/using-monty-for-robotics.md) [No Habitat Required]: Building on the previous tutorial, this goes into depth on how to use Monty with physical sensors and actuators (aka robots :robot:). 