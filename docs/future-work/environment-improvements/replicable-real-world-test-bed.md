---
title: Set Up Replicable Real World Test Bed
description: Create a real-world test bed for Monty with instructions on how to replicate it.
rfc: optional
estimated-scope: large
improved-metric: real-world
output-type: experiments, testbed
skills: python, robotics-hardware
status: open
---
Currently we predominantly test Monty in simulation. However, working in simulation can miss important factors encountered in the real world. We occasionally venture out to test Monty with real sensors and robotics hardware (see [project showcase](../../community/project-showcase.md) for some examples), but these projects are not easily replicable or repeatable on a regular basis. 

It would be great to have a robotic setup that can easily be replicated by anyone. This means, all parts should be listed and obtainable, ideally at low cost. There should also be precisely specified experimental setups and evaluation pipelines (similar to our [benchmarks in simulation](../../overview/benchmark-experiments.md)).

> [!NOTE]
> For some help on using Monty in a robotic application, see our [Monty for robotics tutorial](https://docs.thousandbrains.org/docs/using-monty-for-robotics)

Some inspiration and existing test beds:

1. An example robotic setup using a RaspberryPi, LEGOs, and cameras, can be found here: https://github.com/thousandbrainsproject/everything_is_awesome However, this was set up using a variety of LEGO pieces bought at different stores and is therefore not easily replicable. One idea could be to put together a similar setup with 3D printed parts instead, where CAD models can be shared openly.
2. We have some "offline" real-world Monty benchmark experiments where we collected data with real sensors and share the dataset and experiments. Those are [Monty Meets World](https://github.com/thousandbrainsproject/everything_is_awesome) (using the iPad depth camera) and the [Ultrasound dataset](https://github.com/thousandbrainsproject/tbp.ultrasound_perception) (containing images collected with the Butterfly mobile ultrasound probe). One way to build on these would be to have a setup where the sensor explicitly moves in space (as it did in the ultrasound dataset), but where the input is RGB-D or stereoscopic RGB inputs (which would motivate exploring alternative depth-measurement methods). It may be possible to identify an already existing dataset of this kind in the literature, but note that it is essential that the sensor's pose is accurately tracked and stored as part of the dataset.
