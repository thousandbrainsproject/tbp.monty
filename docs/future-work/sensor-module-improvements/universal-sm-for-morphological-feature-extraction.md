---
title: Universal SM for Morphological Feature Extraction
description: Develop a sensor module that automatically extracts CMP-compliant morphological features, independent of input modality and dimensionality.
rfc: required
estimated-scope: large
improved-metric: features-and-morphology, transfer, deformations, generalization, real-world
output-type: prototype, monty-feature, PR, paper
skills: python, research, computer-vision
contributor: 
status: open
---

The aim here is to develop a universal sensor module that automatically extracts CMP-compliant morphological features, independent of input modality and dimensionality. The brain suggests that this may be possible. In particular, center-surround response properties appear to be key to sensory processing in vision, somatosensation, and auditory processing. This suggests that local changes across space may be what defines morphological features like a visual or tactile edge, as well as their potential auditory analogues, like a frequency tone.

In the roadmap for designing Monty, we have emphasized that learning modules will be general, while sensor modules need to be designed for their particular use case. The hope with this Future Work item is that we could design a universal SM that will also be general. This would significantly simplify deploying Monty in new domains such as sound (which Monty does not currently support), a "whisker" based somatosensory system, sensors detecting electromagnetic fields in 3D, and other, more exotic, settings.

Importantly, the "space" in "local changes across space" would ideally include 1D, 2D, and 3D space. As such, progress here might support other Future Work items, such as feature extraction in [a 1D sensor module](one-d-sensor-module.md). We also note that extracting *movement* in an appropriate dimensionality is a separate, but related, problem to extracting morphological features.

A proposed implementation could be tested by verifying that it recovers the morphological features that our current SMs already extract in [3D](https://docs.thousandbrains.org/docs/sm-for-rgbd-data) and [2D](https://docs.thousandbrains.org/docs/two-d-sensor-module) space given an RGB-D input, as well as the features extracted by the pipeline we developed for [ultrasound](https://github.com/thousandbrainsproject/tbp.ultrasound_perception).