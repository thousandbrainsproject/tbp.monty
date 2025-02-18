---
title: FAQ - Monty
description: Frequently asked questions about the Thousand Brains Theory, Monty, thousand-brains systems, and the underlying algorithms.
---

Below are responses to some of the frequently asked questions we have encountered. However, this is not an exhaustive list, so if you have a question, please reach out to us and the rest of the community at our [Discourse page](https://thousandbrains.discourse.group/). We will also update this page with new questions as they come up.

# General

## What is the Difference Between the Thousand Brains Theory, Monty, and Thousand-Brains Systems?

We use these terms fairly interchangeably (particularly in our meetings), however the [Thousand Brains Theory](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00121/full) (TBT) is the underlying theory of how the neocortex works. Thousand-brains systems are artificial intelligence systems designed to operate according to the principles of the TBT. Finally, Monty is the name of the first system that implements the TBT (i.e. the first thousand-brains system), and is made available in our [open-source code](https://github.com/thousandbrainsproject/tbp.monty/).

# Neuroscience Theory

## What is the Relationship of the Thousand Brains Theory to the Free Energy Principle?

The Free-Energy Principle and Bayesian theories of the brain are interesting and broadly compatible with the principles of the Thousand Brains Theory (TBT). While they can be useful for neuroscience research, our view is that Bayesian approaches are often too broad for building practical, intelligent systems. Where attempts have been made to implement systems with these theories, they often require problematic assumptions, such as modeling uncertainty with Gaussian distributions. While the concept of the neocortex as a system that predicts the nature of the of world is common to the Free-Energy Principle and the Thousand Brains Theory (as well as several other ideas going back to Hermann von Helmholtz), we want to emphasize the key elements that set the TBT apart. This includes, for example, the use of a modular architecture with reference frames, where each module builds representations of entire objects, and a focus on building a learning system that can model physical objects, rather than on fitting neuroscience data.

## Do Cortical Columns in the Brain Really Model Whole Objects, Like a Coffee Mug in V1?

One important prediction of the Thousand Brains Theory is that the intricate columnar structure found throughout brain regions, including primary sensory areas like V1 (early visual cortex), supports computations much more complex than extracting simple features for recognizing objects.

To recap a simple version of the model (i.e. simplified to exclude top-down feedback or motor outputs):
We assume the simple features that are often detected experimentally in areas like V1 correspond to the feature input (layer 4) in a cortical column. Each column then integrates movement in L6, and uses features-at-locations to build a more stable representation of a larger object in L3 (i.e. larger than the receptive field of neurons in L4). L3’s lateral connections then support "voting", enabling columns to inform each-other’s predictions. Some arguments supporting this model are:

i) It is widely accepted that the brain is constantly trying to predict the future state of the world. A column can predict a feature (L4) at the next time point much better if it integrates movement, rather than just predicting the same exact feature, or predicting it based on a temporal sequence - bearing in mind that we can make movements in many directions when we do things like saccade our eyes. Reference frames enable predicting a particular feature, given a particular movement - if the column can do this, then it has built a 3D model of the object.

ii) Columns with different inputs need to be able to work together to form a consensus about what is in the world. This is much easier if they use stable representations, i.e. a larger object in L3, rather than representations that will change moment to moment, such as a low-level feature in L4. Fortunately this role for lateral connections fits well with anatomical studies.

We use a coffee mug as an illustrative example, because a single patch of skin on a single finger can support recognizing such an object by moving over it. With all this said however, we don’t know exactly what the nature of the “whole objects” in the L2/L3 layers of V1 would be (or other primary sensory areas for that matter). As the above model describes, we believe they would be significantly more complex than a simple edge or Gabor filter, corresponding to 3D, statistically repeating structures in the world that are cohesive in their representation over time.

It is also important to note that compositionality and hierarchy are still very important even if columns model whole objects. For example, a car can be made up of wheels, doors, seats, etc., which are distinct objects. The key argument is that a single column can do a surprising amount, more than what would be predicted by artificial neural network (ANN) style architectures, modeling much larger objects than their receptive fields would indicate.

## Why is There no Hierarchical Temporal Memory (HTM) in This Version of Monty?

We are focused on ensuring that the first generation of thousand-brain systems are interpretable and easy to iterate upon. Being able to conceptually understand what is happening in the Monty system, visualize it, debug it, and propose new algorithms in intuitive terms is something we believe to be extremely valuable for fast progress. As such, we have focused on the core principles of the TBT, but have not yet included lower-level neuroscience components such as HTM, sparse distributed representations (SDRs), grid-cells, and active dendrites. In the future, we will consider adding these elements where a clear case for a comparative advantage exists.

## Where Does the Hippocampal Formation Fit Into the Thousand Brains Theory?

Part of the underlying Thousand Brains Theory is that the hippocampal formation evolved to enable rapid episodic memory, as well as spatial navigation in animals, with grid cells forming reference frames of environments. Over time, evolution repurposed this approach of modeling the world with reference frames in the form of cortical columns, which were then replicated throughout the neocortex to model objects at any level of abstraction.

As such, we believe that the core computations that a cortical column performs are similar to the hippocampal formation. Since learning modules (LMs) are designed to capture the former, they also implement the core capabilities of the latter. This includes elements such as how objects change as a function of time (analogous to episodic memory), and recognizing specific vs. general instances of objects (analogous to pattern separation vs. completion). The key difference we believe is the time-scale over which learning happens, with the hippocampal complex laying down new information much more rapidly. In Monty, this could be implemented as a high-level LM that adds new information to object models extremely quickly.

It’s worth noting that the speed of learning in LMs is an instance where Monty might be “super-human”, in that within computers, it is easy to rapidly build arbitrary, new associations, while this is challenging for biology. Evolution has required innovations such as silent synapses and an excess of synapses in the hippocampal formation to achieve this ability. This degree of neural hardware to support rapid learning cannot be easily replicated in the compact space of a cortical column, so in biology the latter will always learn much more slowly, and very rapid learning is a more specialized domain of the hippocampal formation.

In the case of Monty, we might choose to enable most LMs to have fairly rapid learning, although we believe there are other reasons that we might restrict this, such as ensuring that low-level representations do not drift (change) too rapidly with respect to high-level ones.

Finally, it's worth noting that there may be other unique features of the hippocampal complex that would require details not found in standard LMs, however our aim is to re-use the implementation of LMs where possible.

## If "Cognitive Maps" are Found in the Hippocampal Formation, Why Do We Believe They Are Found in the Neocortex?

Neuroscientists have found evidence that grid cells in the hippocampal complex can encode abstract reference frames, such as the conceptual space of an object's properties ([Constantinescu et al, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5248972/)), what is often referred to as "cognitive maps" ([O'Keefe and Nadel, 1978](https://www.cmor-faculty.rice.edu/~cox/neuro/HCMComplete.pdf)). The Thousand Brains Theory does *not* argue that abstract, structured representations cannot be found in the hippocampal formation. As noted above, this structure has been highly developed so as to enable very rapid learning, which would account for its important role in learning structured representations in mammals. Furthermore, the fact is that this neural hardware exists, so even if cortical columns learn structured representations, it is natural that these would also emerge in the hippocampal formation. This would result in a degree of redundancy between cortical columns and the hippocampal formation, at least early in learning. With this view, evidence for structured representations in the hippocampal formation does not imply that such representations cannot be found in cortical columns.

More generally, we believe the learning that takes place in the reference frames of cortical columns happens more slowly, which would make it more challenging to measure experimentally. If you are an experimental neuroscientist interested in structured representations of objects, we would love to discuss ways that it might be possible to measure the emergence of such representations in the cortical columns.

## Does the "What" vs "Where" Pathway Exist in Monty?

In short, not at the moment, and it's unclear how essential this would be. 

It has been proposed that the *ventral* and *dorsal* streams of the visual cortex correspond to "what" vs "where" pathways respectively. In this context, the terms “what” vs. “where” can be misleading, as spatial computation is important throughout the brain, including within cortical columns in the what pathway. This is central to the TBT claim that every column leverages a reference frame, and so "what" should not be interpreted as there being a part of the brain that does not care about spatial relations. Even the alternative naming of the ventral and dorsal streams as a “perception” vs. “action” stream can be misleading, as all columns have motor projections. For example, [eye control can be mediated by columns in the ventral stream projecting to the superior colliculus, as well as by other sensory regions](https://pubmed.ncbi.nlm.nih.gov/6096414/).

However one distinction that might exist, at least in the brain, is the following: for columns to meaningfully communicate spatial information with one another, there needs to be some common reference frame. Within a column, the spatial representation is object-centric, but a common reference frame comes into play when different columns interact.
- One choice for this common reference frame is a body-centric coordinate system, which is likely at play in the dorsal ("where") stream. This would explain its importance for directing complex motor actions, as in [the classic Milner and Goodale study](https://pubmed.ncbi.nlm.nih.gov/1374953/) that spawned the two-stream framing of function.
- An alternative choice is an “allocentric” reference frame, which could be some temporary landmark in the environment, such as the corners of a monitor, or a prominent object in a room. This may be utilized in the ventral ("what") pathway.

In Monty, the between-column computations, such as voting, have made use of an ego/body-centric shared coordinate system. However, this might change in the future, where motor coordination would benefit from egocentric coordinates, and reasoning about object interactions might benefit from allocentric coordinates. If ever implemented, this could be analogous to separate "what" and "where" pathways.

# Alternative Approaches to Intelligence

## What is the Relationship of the Thousand Brains Theory to Robotics Algorithms That Use Maps?

There are deep connections between the Thousand Brains Theory and Simultaneous Localization and Mapping (SLAM), or related methods like particle filters. This relationship was discussed, for example, in [Numenta’s 2019 paper by Lewis et al](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2019.00022/full), in a discussion of grid-cells:

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

This connection points to a deep relationship between the objectives that both engineers and evolution are trying to solve. Methods like SLAM emerged in robotics to enable navigation in environments, and the hippocampal complex evolved in organisms for a similar purpose. One of the arguments of the TBT is that the same spatial processing that supported representing environments was compressed into the 6-layer structure of cortical columns, and then replicated throughout the neocortex to support modeling *all* concepts with reference frames, not just environments. 

Furthermore, Monty's evidence-based learning-module has clear similarities to particle filters, such as its non-parametric approximation of probability distributions. However we have designed it to support specific requirements of the Thousand Brains Theory - properties which we believe neurons have - such as binding information to points in a reference frame.

So in some ways, you can think of the Thousand-Brains Project as leveraging concepts similar to SLAM or particle filters to model all structures in the world (including abstract spaces), rather than just environments. However, it is also more than this. For example, the capabilities of the system to model the world and move in it magnify due to the processing of many, semi-independent modeling units, and the ways in which these units interact.

## What is the Relationship of the Thousand Brains Theory to Swarm Intelligence?

There are interesting similarities between swarm intelligence and the Thousand Brains Theory. In particular, thousand-brains systems leverage many semi-independent computational units, where each one of these is a full sensorimotor system. As such, the TBT recognizes the value of distributed, sensorimotor processing to intelligence. However, the bandwidth and complexity of the coordination is much greater in the cortex and thousand-brains systems than what could occur in natural biological swarms. For example, columns share direct neural connections that enable them to form compositional relations, including decomposing complex actions into simpler ones.

It might be helpful to think of the difference between prokaryotic organisms that may cooperate to some degree (such as bacteria creating a protective biofilm), vs. the complex abilities of eukaryotic organisms, where cells cooperate, specialize, and communicate in a much richer way. This distinction underlies the capabilities of swarming animals such as bees, which, while impressive, do not match the intelligence of mammals. In the long-term, we imagine that Monty systems can use communicating agents of various complexity, number and independence as required.

## Why Does Monty Not Make Use of Deep Learning?

Deep learning is a powerful technology - we use large-language models ourselves on a daily basis, and systems such as AlphaFold are an amazing opportunity for biological research. However, we believe that there are many core assumptions in deep learning that are inconsistent with the operating principles of the brain. It is often tempting when implementing a component in an intelligent system to reach for a deep learning solution. However, we have made most conceptual progress when we have set aside the black box of deep learning and worked from basic principles of known neuroscience and the problems that brains must solve.

As such, there may come a time where we leverage deep learning components, particularly for more "sub-cortical" processing such as low-level feature extraction, and model-free motor policies (see below), however we will avoid this until they prove themselves to be absolutely essential. This is reflected in our [request that code contributions are in Numpy, rather than PyTorch](https://thousandbrainsproject.readme.io/docs/style-guide#numpy-preferred-over-pytorch).

## What is the Relationship of the Thousand Brains Theory to Reinforcement Learning?

Reinforcement learning (RL) can be divided into two kinds, model-free and model-based. Model-free RL can be used by the brain, for example, to help you proficiently and unconsciously ride a bicycle by making fine adjustments in your actions in response to feedback. Current deep reinforcement learning algorithms are very good at this ([Mnih et al, 2015](https://www.nature.com/articles/nature14236)). However, when you learned to ride a bicycle, you likely watched your parents give a demonstration, listened to their explanation, and had an understanding of the bicycle's shape and the concept of pedaling before you even started moving on it. Without these deliberate, guided actions, it could take thousands of years of random movement in the vicinity of the bicycle until you figured out how to ride it, as positive feedback (the bicycle is moving forward) is rare.

All of these deliberate, guided actions you took as a child were "model-based", i.e. dependent on models of the world. These models are learned in an unsupervised manner, without reward signals. Mammals are very good at this, as demonstrated by [Tolman's classic experiments with rats in the 1940s](https://psycnet.apa.org/record/1949-00103-001). However, how to learn and then leverage these models in deep reinforcement learning is still a major challenge. For example, part of DeepMind's success with [AlphaZero (Silver et al, 2018)](https://www.science.org/doi/10.1126/science.aar6404) was the use of explicit models of game-board states. However, for most things in the world, these models cannot be added to a system like the known structure of a Go or chess board, but need to be learned in an unsupervised manner.

While this remains an active area of research in deep-reinforcement learning ([Hafner et al, 2023](https://arxiv.org/pdf/2301.04104)), we believe that the combination of 3D, structured reference frames with sensorimotor loops will be key to solving this problem. In particular, thousand brains systems learn (as the name implies) thousands of semi-independent models of objects through unsupervised, sensorimotor exploration. These models can then be used to decompose complex tasks, where any given learning module can propose a desired "goal-state" based on the models that it knows about. This enables tasks of arbitrary complexity to be planned and executed, while constraining the information that a single module needs to learn about the world. Finally, the use of explicit reference frames increases the speed at which learning takes place, and enables planning arbitrary sequences of actions. Like Tolman's rats, this is similar to how you can navigate around a room depending on what obstacles there are, such as an office chair that has been moved, without needing to learn it as a specific sequence of movements.

In the long term, there may be a role for something like deep-reinforcement learning to support the model-free, sub-cortical processing of thousand-brains systems. However, the key open problem, and the one that we believe the TBT will be central to, is unlocking the model-based learning of the neocortex.

## How Can Monty Learn Without Back-Propagation for Credit Assignment?

Attempts to explain how back-propagation might exist in the brain require problematic assumptions, such as 1:1 associations between neurons that are not observed experimentally. Furthermore, systems reliant on back-propagation display undesirable learning characteristics, such as catastrophic forgetting, and the requirement for large amounts of training data.

In Monty, we make use of associative learning, together with a strong spatial inductive bias, to enable rapid learning without the use of back-propagation. Although we do not currently use modifiable "weights", one way to conceptualize this is as a form of Hebbian learning in a Hopfield (associative memory) network, where memories are embedded in reference frames, rather than a single, homogeneous population of neurons.

As we introduce hierarchy with potentially distant dependencies, there are a few properties of the Thousand Brains Theory that should make learning less dependent on long-range credit assignment as compared to traditional neural networks:
1. Columns throughout the brain (and therefore Monty learning modules) have direct sensory input and motor output, and are therefore able to build predictive models of the world using information locally available to them. As such, a significant amount of learning involves a very flat hierarchy.
2. The vast majority of learning is based on predicting the nature of the world in an unsupervised manner, i.e. it is a very dense learning signal, unlike trying to learn sensorimotor systems end-to-end using reinforcement learning.
3. The use of a strong, spatial inductive bias significantly reduces the number of samples required to build a representation that can generalize.

It is also interesting to consider how learning occurs in humans when "credit-assignment" *is* required. For example, long-distance, sparse associations can occur after achieving a reward at the end of a complex task. However, learning in this setting involves the use of explicit, causal world models to understand what helped - for example, "I flipped a particular switch, which opened the door, and then I was able to climb onto the platform. Based on my causal knowledge of the world, the fact that I was humming a tune while I did this probably did not contribute to my success."

Such explicit credit-assignment is not limited to reinforcement learning. When you learn a new, compositional object, you develop a representation by iteratively learning shallow levels of composition, and building upon these. This is why it is important when children learn to read that they first learn to recognize the individual letters, which are composed of strokes. Once letters are recognized, children can learn how letters form words, and so on. This learning does not take place by showing children blocks of text, and expecting them to passively develop representations in a deep hierarchy through a form of error propagation.

This is different from the implicit, model-free credit assignment that neural networks use. It is also why, even *with* biologically implausible error transport (i.e. back-prop for credit-assignment), deep learning models need to train on orders of magnitude more data than humans to achieve comparable performance.

## Can't Deep Learning Systems Learn "World Models"?

We believe that there is limited evidence that deep learning systems, including generative pre-trained transformers (GPTs) and diffusion models, can learn sufficiently powerful "world models" for true machine intelligence. For example, representations of objects in deep learning systems tend to be highly entangled and divorced of concepts such as cause-and-effect (see e.g. [Brooks, Peebles et al, 2024](https://openai.com/index/video-generation-models-as-world-simulators/)), in comparison to the object-centric representations that are core to how humans represent the world even from an extremely young age ([Spelke, 1990](https://www.harvardlds.org/wp-content/uploads/2017/01/Spelke1990-1.pdf)). Representations are also often limited in structure, manifesting in the tendency of deep learning systems to classify objects based on texture more than shape ([Gavrikov et al, 2024](https://arxiv.org/html/2403.09193v1)), an entrenched vulnerability to adversarial examples ([Szegedy et al, 2013](https://arxiv.org/abs/1312.6199)), the tendency to hallucinate information, and the idiosyncrasies of generated images (such as inconsistent numbers of fingers on hands), when compared to the simpler, but much more structured drawings of children.

Instead, these systems appear to learn complex input-output mappings, which are capable of some degree of composition and interpolation between observed points, but limited generalization beyond the training data. This makes them useful for many tasks, but requires training on enormous amounts of data, and limits their ability to solve benchmarks such as [ARC-AGI](https://github.com/fchollet/ARC-AGI), or more importantly, make themselves very useful when physically embodied. This dependence on input-output mappings means that even approaches such as chain-of-thought or searching over the space of possible outputs (e.g. the [recent o1 models](https://openai.com/index/introducing-openai-o1-preview/)), are more akin to searching over a space of learned "Type 1" actions, rather than the true "Type 2" ([Stanovich and West, 2000](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/abs/individual-differences-in-reasoning-implications-for-the-rationality-debate/2906AEF620B36C10018DD291F790BE97)), model-based thinking that is a marker of intelligence.

## How Does Hierarchical Composition Relate to the Hierarchical Features in Convolutional Neural Networks?

In CNNs, and deep-learning systems more generally, there is often a lack of “object-centric” representations, which is to say that when processing a scene with many objects, the properties of these tend to be mixed up with one another. This is in contrast to humans, where we understand the world as being composed of discrete objects with a degree of permanence, and where these objects have the ability to interact with one another.

Furthermore, any given object in our brain is represented spatially, where the shape of the object - i.e. the relative arrangement of features - is far more important than low-level details like a texture that might correlate with a particular object class. This is how we see a person in the famous Vertumnus painting by Arcimboldo, despite all the local features being pieces of fruit. Again, this is different from how CNNs and other deep-learning systems learn to represent objects.

![Vertumnus by Arcimboldo, 1591, oil on canvas](../figures/how-monty-works/painting_vertumnus.png#width=300px)

Importantly, such object-centric and spatially-structured representations do not just exist at one level of abstraction, but throughout various levels of hierarchy, from how you understand something like an iris or a fingernail, all the way up to your representation of a person. This continues to extend upwards to abstract concepts like society, where representations continue to be discrete, structured objects.

So while there is hierarchy in both CNNs and the human visual system, the former can be thought of more as a bank of filters that detect things like textures and other correlations between input pixels and output labels. We believe that in the brain, every level of the hierarchy represents discrete objects with their own structure and associated motor policies. These can be rapidly composed and recombined, enabling a wide range of representations and behaviors to emerge.

## Is the Thousand Brains Theory Related to "Symbolic" Artificial Intelligence?

Concepts from symbolic AI have some similarities to the Thousand Brains Theory, including the importance of discrete entities, and mapping how these are structurally related to one another. However, we believe that it is important that representations are grounded in a sensorimotor model of the world, whereas symbolic approaches typically begin at high levels of abstraction.

However, the approach we are adopting contrasts to some "neuro-symbolic" approaches that have been proposed. In particular, we are not attempting to embed entangled, object-impoverished deep-learning representations within abstract, symbolic spaces. Rather, we believe that object-centric representations using reference frames should be the representational substrate from the lowest-level of representations (vision, touch) all the way up to abstract concepts (languages, societies, mathematics, etc.). Such a commonality in representation is consistent with the re-use of the same neural hardware (the cortical column) through the human neocortex, from sensory regions to higher-level, "cognitive" regions.

# Applications of Monty

## Can Monty be Used for a Scientific Problem, Like Mapping the Genetic Sequence of a Protein to its 3D Structure?

This depends a lot on how data is available. Given a static dataset of genetic sequences and their mapping onto the 3D structures of their proteins, Monty is not going to work well, while this is where a function-approximation algorithm like deep-learning can excel.

Where Monty would eventually shine is when it is able to control experimental devices that allow it to further probe information about the world, i.e. a sensorimotor environment. For example, Monty might have some hypotheses about a structure, and want to test these through various experiments, probing the space in which it is uncertain. We embed representations in 3D coordinates that can take on any kind of graph structure necessary which is <= 3D space (strings, graphs defined by edges, or 3D point-clouds), so in theory these kinds of entities can all be represented.

How Monty would *learn* to generalize a mapping between these levels of representations remains outstanding, and relates to how Monty can learn a mapping between different spaces (e.g. meeting a new family, and mapping these people onto an abstract family-tree structure). We are still figuring out exactly how this would work in a simpler case like the family-tree. In the protein case, the rules are much more complex, and so learning this is definitely not something that Monty can do now. However, just like a human scientist, the ultimate aim would be for Monty to learn to do this mapping based on a causal understanding of the world, informed by the above mentioned experiments, and in contrast to an end-to-end black-box function approximation.

You can also read more about [applications of Monty under the suggested criteria](../overview/application-criteria.md).