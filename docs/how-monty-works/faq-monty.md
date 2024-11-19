---
title: FAQ - Monty
description: Frequently asked questions around the algorithm implemented in Monty.
---

# What is the relationship of Monty to the Free Energy Principle?

The Free-Energy Principle and Bayesian theories of the brain are interesting and broadly compatible with the principles of the Thousand Brains Theory. While they can be useful for neuroscience research, our view is that Bayesian approaches are often too broad and require problematic assumptions (such as modelling noise as Gaussian) for building practical, intelligent systems. While the concept of the neocortex as a system that predicts the nature of the of world is common to the Free-Energy Principle and the Thousand Brains Theory (as well as much older ideas going back to Hermann von Helmholtz), we want to emphasize the key elements that set the TBT apart, such as the use of a modular architecture with reference frames, where each module builds representations of entire objects.

# What does it mean to say that cortical columns in the brain model whole objects? Does the TBT claim there are models of coffee mugs in V1 (early visual cortex)?

One important prediction of the Thousand Brains Theory is that the intricate columnar structure found throughout brain regions, including early sensory areas like V1, supports computations much more complex than extracting simple features for recognizing objects.

To recap a simple version of the model (i.e. simplified to exclude top-down feedback or motor outputs):
We assume the simple features that are often detected experimentally in areas like V1 correspond to the feature input (layer 4) in a cortical column. Each column then integrates movement in L6, and uses features-at-locations to build a more stable representation of a larger object in L3 (i.e. larger than the receptive field of neurons in L4). L3’s lateral connections then support "voting", enabling columns to inform each-other’s predictions. Some arguments supporting this model are:

i) It is widely accepted that the brain is constantly trying to predict the future state of the world. A column can predict a feature (L4) at the next time point much better if it integrates movement, rather than just predicting the same exact feature, or predicting it based on a temporal sequence - bearing in mind that we can make arbitrary movements when we do things like saccade our eyes. Reference frames enable predicting a particular feature, given a particular movement - if the column can do this, then it has built a model of the object.

ii) Columns with different inputs need to be able to work together to form consensus about what is in the world. This is much easier if they use stable representations, i.e. a larger object in L3, rather than representations that will change moment to moment, such as a low-level feature in L4. Fortunately this role for lateral connections fits well with the anatomy.

We use a coffee mug as an illustrative example, because a single skin pad on a single finger can support recognizing such an object by moving over it. With all this said however, we don’t know exactly what the nature of the “whole objects” in the L2/L3 layers of V1 would be (or other primary sensory areas for that matter). As the above model describes, we believe they would be significantly more complex than a simple edge or Gabor filter, corresponding to statistically repeating structures in the world that are cohesive in their representation over time.

It’s also important to note that compositionality and hierarchy is still very important even if columns model whole objects. For example, a car can be made up of wheels, doors, seats, etc., which are distinct objects. Instead, we argue that a single column can do a surprising amount, more than what would be predicted by artificial neural-network (ANN) style architectures.

# What is the relationship of Monty to robotics algorithms that use maps, such as particle filters and Simultaneous Localization and Mapping (SLAM)?

There are some deep connections between the Thousand Brains Theory (TBT) and SLAM, or related methods like particle filters.

This relationship was discussed, for example, in Numenta’s 2019 paper by Lewis et al, during a discussion of grid-cells:

“To combine information from multiple sensory observations,
the rat could use each observation to recall the set of all
locations associated with that feature. As it moves, it could
then perform path integration to update each possible location.
Subsequent sensory observations would be used to narrow
down the set of locations and eventually disambiguate the
location. At a high level, this general strategy underlies a set of
localization algorithms from the field of robotics including Monte
Carlo/particle filter localization, multi-hypothesis Kalman filters,
and Markov localization (Thrun et al., 2005).”

This connection points to a deep relationship between the objectives that both engineers and evolution are trying to solve. Methods like SLAM emerged in robotics to enable navigation in environments, and the hippocampal complex evolved in organisms for a similar purpose. One of the arguments of the TBT is that the same spatial processing that supported representing environments was compressed into the 6-layer structure of cortical columns, and then replicated throughout the neocortex to support modelling *all* concepts with reference frames, not just environments. 

Furthermore, Monty's evidence-based learning-module has clear similarities to particle filters, such as its non-parametric approximation of probability distributions. However we have designed it to support specific requirements of the Thousand Brains Theory - properties which we believe neurons have - such as binding information to points in a reference frame.

So in some ways, you can think of the Thousand-Brains Project as leveraging concepts similar to SLAM or particle filters to model all structures in the world (including abstract spaces), rather than just environments. However, it is also more than this. For example, the capabilities of the system to model the world and move in it magnify due to the processing of many, semi-independent modelling units, and the ways in which these units interact.

# What is the relationship of Monty to swarm intelligence?

There are interesting similarities between swarm intelligence and the Thousand Brains Theory. In particular, thousand-brains systems leverage many semi-independent computational units, where each one of these is a full sensorimotor system. As such, the TBT is a recognition of the centrality of distributed, sensorimotor processing to intelligence. However, the bandwidth and complexity of the coordination is much greater in the cortex and thousand-brains systems than what could occur in natural biological swarms.

It might be helpful to think of the difference between prokaryotic organisms that may cooperate to some degree (such as bacteria creating a protective biofilm), vs. the complex abilities of eukaryotic organisms, where cells cooperate, specialize, and communicate in a much richer way. This distinction likely underlies the capabilities of swarming animals such as bees, which while impressive, do not match the intelligence of mammals. In the long-term, we imagine that Monty systems can use communicating agents of various complexity, number and independence as required.

# What is the relationship of Monty to reinforcement learning, including deep reinforcement learning?

- world models, model free and model-based

deep-RL sample efficiency; model-based challenging

Tolman and mice; learning to ride a bike

Can compare to dreamer based architecture, alphago (a success but not practical, recent deepmind playing a bunch of games, doom game, and O1 preview. Is alphazero any different?


# Why doesn't Monty make use of deep learning?

Why don't you use deep learning for various components: their may come a day when we use it, clearly powerful for many use cases. However we have found we have made the most conceptual progress when we set aside the black box of dnns and try to develop a system based on first principles? Could eventually be for subcortical, where current deep RL more resembles biology

See TODOist

May use for more sub-cortical processing, e.g. model-free policies or some sensory feature extraction.

Highlight that representations are object-centric at every layer of abstraction (ref slot attention in contrast)
- also continual leanring; structured representations (shape-based, no adversarial examples - see blog post).

?Add some references


# Connecion to GPT-o models and GPT models generally
- unsupervised learning
- use of search and therefore "type 2 thinking"


# Why are there no grid-cells or Hierarchical Temporal Memory (HTM) in this version of Monty?

?Make the neural research roadmap public --> can turn into a PDF; should make sure it looks good enough; e.g. Jeff hasn't reviewed it yet


# What about Sutton's bitter lesson? Most progress in AI was made just by scaling up existing methods right?

Evolution also discovered scale, but it's not everything.

Scaling of number of LMs vs. amount of body to model is clearly important, see e.g. primates and dolphins vs. whales. 

Does this mean scale doesn't matter - no, but matters what scaling. Also data shouldn't be key. Scaling this has given an illusion of generalisation