---
title: Improve Monty's Performance on Ultrasound
description: Improve the ability of Monty to learn and infer 3D objects in ultrasound images.
rfc: not-required
estimated-scope: medium
improved-metric: real-world, accuracy, noise, numsteps
output-type: experiments, analysis, publication
skills: python, computer-vision, research
contributor: nleadholm
status: evergreen
---

We would like to continue to improve Monty's ability to perform learning and inference given real-world ultrasound data. Ultrasound is an inherently sensorimotor modality with wide applications in the real world. Achieving strong performance on ultrasound tasks despite limited training data is a major open challenge in machine learning.

We have developed a separate repository called [tbp.ultrasound_perception](https://github.com/thousandbrainsproject/tbp.ultrasound_perception), and we welcome contributions to improve Monty's performance in this domain. For example, there are various improvements that can be made to how ultrasound images are processed before information is sent to Monty, as well as ways to integrate the latest Monty capabilities into the ultrasound perception repository. To help track items of work, this Future Work item is marked as `evergreen`, and we use individual Issues in the `tbp.ultrasound_perception` repository.