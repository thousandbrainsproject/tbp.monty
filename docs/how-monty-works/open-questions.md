---
title: Open Questions
description: I'm just starting to collect a new list of open questions here. Still a WIP
---

# Learning Modules/Modeling

## Object Behaviors

- How are object behaviors represented?
- How are the recognized?

## Object Models
- Where do we have a model of general physics? Can every LM learn the basic physics necessary for the objects it models (e.g. fluid-like behavior in some, cloth-like behavior in others)? Or are some LMs more specialized for this?

## Object Transformations

- How are the represented?
- How are they recognized?

## Scale

- How is an object recognized irrespective of its scale?
- How do we know the scale of an object?

## Learning
- How does it work that I can learn an object using vision but then recognize it using touch? (in Monty the Sensor Module can simply be switched out but how would it work in the brain? Or how would it work in a multimodal Monty system without rewiring the SM to LM connections or explicitly copying models?)
- Should there be some form of memory consolidation?
- How do we learn where an object begins and where it ends? What defines a sub-component of an object?

# Sensor Modules


# Policies
## Model-Based Policies
- How are goal states decomposed into sub-goals?

## Model-Free Policies and the Motor System
- How does the motor system decide which goal state to carry out (given that every/many LMs produce goal states)?

# CMP

## Routing

- What is the routing mechanism? 
- How does attention come into play here?

## Voting
- Can we assume that models of the same object in different LMs were learned using the same reference frame/in the same orientation?