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

# Why is there Hierarchical Temporal Memory (HTM) or grid-cells in this version of Monty?

We are focused on ensuring that the first generation of thousand-brain systems are interpretable and easy to iterate upon. Being able to conceptually understand what is happening in the Monty system, visualize it, debug it, and propose new algorithms in intuitive terms is something we believe to be extremely valuable to fast progress. As such, we have focused on the core principles of the TBT, but have not yet included lower-level neuroscience componenets such as HTM or grid-cells. In the future, we will consider adding these elements where a clear case for a comparative advantage exists.

# Why doesn't Monty make use of deep learning?

Deep-learning is a powerful technology - we use large-language models ourselves on a daily basis, and systems such as AlphaFold are an amazing opportunity for biological research. However, we believe that there are many core assumptions in deep-learning that are inconsistent with the operating principles of the brain. It is often tempting when implementing a component in an intelligent system to reach for a deep-learning solution. However, we have made most conceptual progress when we have set aside the black box of deep-learning and worked from basic principles of known neuroscience and the problems that brains must solve.

As such, there may come a time where we leverage deep-learning componenets, particularly for more "sub-cortical" processing such as low-level feature extraction, and model-free motor policies (see below), however we will avoid these until they prove themselves to be absolutely essential. 

# What is the relationship of Monty to reinforcement learning, including deep reinforcement learning?

Reinforcement learning can be divided into two kinds, model-free and model-based. Model-free learning is used by the brain, for example, to help you balance on a bicycle by making fine adjustments in your actions in response to feedback. Current deep reinforcement learning algorithms are very good at this (refs, e.g. Break-out, and also Andrew Ng controlling helicopter). However, when you learnt to ride a bicycle, you watched your parent demonstrate how to do it, listened to their explanation, and had an understanding of the bicycle's shape and the concept of peddling before you even started moving on it. Without these deliberate, guided actions, it could take thousands of years of random movement in the vicinity of the bicycle until you figured out how to ride it, as positive feedback (the bicycle is moving forward) is rare.

All of these deliberate, guided actions you took as a child were "model-based", i.e. dependent on models of the world. These models are learned in an unsupervised manner, without reward signals. Mammals are very good at this, as demonstrated by [Tolman's classic experiments with rats in the 1940s](https://psycnet.apa.org/record/1949-00103-001). However, how to learn and then leverage these models in deep reinforcement learning is still a major challenge. For example, part of DeepMind's success with [AlphaZero (Silver et al, 2018)](https://www.science.org/doi/10.1126/science.aar6404) was the use of explicit models of the game-board states. However, for most things in the world, these models cannot be added to a system like the known structure of a Go-board, but need to be learned in an unsupervised manner.

While this remains an active area of research in deep-reinforcement learning ([Hafner et al, 2023](https://arxiv.org/pdf/2301.04104)), we believe that the combination of 3D, structured reference frames with sensorimotor loops will be key to solving this problem. In particular, thousand brains systems learn (as the name implies) thousands of semi-independent models of objects through unsupervised, sensorimotor exploration. These models can then be used to decompose complex tasks, where any given learning module can propose a desired "goal-state" based on the models that it knows about. This enables tasks of arbitrary complexity to be planned and executed, while constraining the information that a single module needs to learn about the world. In addition, the use of explicit reference frames increases the speed at which learning takes place, and enables following arbitrary sequences of actions. Like Tolman's rats, this is similar to how you can navigate around a room depending on what obstacles there are, such as an office chair that has been moved, without needing to learn it as a specific sequence of movements.

In the long term, there may be a role for something like deep-reinforcement learning to support the model-free, sub-cortical processing of thousand-brains systems. However the key open problem, and the one which we believe the TBT will be central to, is unlocking model-based learning in the cortex.


# Can't modern deep-learning systems like generative pre-trained transformers (GPTs) and diffusion-models learn "world models"?

We believe that there is limited evidence that deep-learning systems, including these generative architectures, can learn sufficiently powerful "world models" for true machine intelligence. For example, representations of objects in deep-learning systems tend to be highly entangled and divorced of concepts such as cause-and-effect (ref slot attention; ref Sora failure), in comparison to the object-centric representations that are core to how humans represent the world even from an extremely young age (ref bio). Representations are also often limited in structure, manifesting in the tendancy of deep-learning systems to classify objects based on texture more than shape (ref recent), a vulnerability to adversarial examples (ref), the hallucinations of information (ref), and the idiosynchrasies of generated images (such as inconsistent number of fingers on hands), when compared to the simpler, but much more structured drawings of children.

*refs in blog post likely useful

Instead, these systems appear to learn complex input-output mappings, which are capable of some degree of interpolation between observed points, but limited generalization beyond the training data. This makes them useful for many tasks, but requires training on enormous amounts of data, and limits their ability to solve benchmarks such as ARC-AGI (ref), or more importantly, make themselves very useful when physically embodied. This dependence on input-output mappings means that even approaches such as searching over the space of possible outputs (e.g. the recent o1 models (ref)), are more akin to searching over a space of learned "type-1" actions (ref type 1 vs type 2 thinking pape), rather than the true "type-2", model-based planning that is a marker of intelligence.


Pass through Grammarly