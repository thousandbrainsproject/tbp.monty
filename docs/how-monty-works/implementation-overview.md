---
title: Implementation Overview
---
# Recap

The implementation of Monty contains all the basic building blocks that were described in the documentation section. To recap, the basic components of Monty are: Sensor modules (SM) to turn raw sensory data into a common language; learning modules (LM) to model incoming streams of data and use these models for interaction with the environment; motor system(s) to translate abstract motor commands from the learning module into the individual action space of agents; and an environment in which the system is embedded and which it tries to model and interact with. The components within the Monty model are connected by the Cortical Messaging Protocol (CMP) such that basic building blocks can be easily repeated and stacked on top of each other. Any communication within Monty is expressed as features at poses (relative to the body) where features can also be seen as object IDs and poses can be interpreted in different ways. For example, pose to the motor system is a target to move to, pose to another LMs input channel is the most likely pose, and poses to the vote connections are all possible poses. All these elements are implemented in Python in the git repository <https://github.com/thousandbrainsproject/tbp.monty> and will be described in detail in the following sections.

# Overview

The classes in the Monty code base implement the abstract concepts described above. **Each basic building block of Monty has its own customizable abstract class.** Additionally, we have an experiment class that wraps around all the other classes and controls the experiment workflow to test Monty.

![The Monty implementation contains several customizable classes representing the key concepts of Monty. Each box in the diagram is a class that can be subclassed and customized to the needs of the specific experiment. For instance, one can implement different types of learning modules or sensor modules while using the general experiment and Monty classes of the framework. Or one could test an existing Monty model set-up in a new environment by updating the environment interface. For a list of all currently available custom classes see the respective tables in each section. (grey arrow indicate that a class has a reference to an existing instance of another class (created by the black arrow connected to it)). In the future we may want the Monty class to be able to have multiple MotorSystem instances (one for each actuator)](../figures/how-monty-works/monty_class_diagram.png)


# Object and Pose Recognition in the Monty Framework

The main testbed for Monty is currently focused on object and pose recognition. This also involves learning models of objects and interacting with the environment but it all serves the purpose of recognizing objects and their poses. In the future, this focus might shift more towards the interaction aspect where recognizing objects is mostly fulfilling the purpose of being able to meaningfully interact with the environment.

# Overview of Monty's Current Default Setup
This video gives a high level overview of all the components in Monty and the custom versions of them that we currently use for our [benchmark experiments](../overview/benchmark-experiments.md). It goes into some more depth on the [evidence based learning module](learning-module/evidence-based-learning-module.md) (our current best LM implementation) and voting.

[block:embed]
{
  "html": "<iframe class=\"embedly-embed\" src=\"//cdn.embedly.com/widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%2FbkwY4ru1xCg%3Ffeature%3Doembed&display_name=YouTube&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DbkwY4ru1xCg&image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FbkwY4ru1xCg%2Fhqdefault.jpg&type=text%2Fhtml&schema=youtube\" width=\"854\" height=\"480\" scrolling=\"no\" title=\"YouTube embed\" frameborder=\"0\" allow=\"autoplay; fullscreen; encrypted-media; picture-in-picture;\" allowfullscreen=\"true\"></iframe>",
  "url": "https://www.youtube.com/watch?v=bkwY4ru1xCg",
  "title": "2023/01 - A Comprehensive Overview of Monty and the Evidence-Based Learning Module",
  "favicon": "https://www.youtube.com/favicon.ico",
  "image": "https://i.ytimg.com/vi/bkwY4ru1xCg/hqdefault.jpg",
  "provider": "https://www.youtube.com/",
  "href": "https://www.youtube.com/watch?v=bkwY4ru1xCg",
  "typeOfEmbed": "youtube"
}
[/block]
