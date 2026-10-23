- Start Date: 2026-06-13
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# RFC: Improve WandB Logging for Unsupervised Learning

## Summary

This RFC proposes improving the visibility of unsupervised continual learning experiments by exposing additional experiment statistics through Weights & Biases (WandB).

The goal is to make it easier to monitor learning progress, memory growth, matching behavior, and graph formation during training and evaluation without changing the underlying learning algorithms.

## Motivation

The current roadmap contains the item:

> Add More Wandb Logging for Unsupervised Learning

The Monty framework already computes and stores a variety of useful statistics related to object matching, graph memory, and learning progress. However, not all of these statistics appear to be readily available through WandB dashboards.

Improved WandB logging would help contributors and researchers:

* Monitor unsupervised learning runs more effectively.
* Compare experiments more easily.
* Debug unexpected learning behavior.
* Better understand graph-memory growth over time.

## Proposed Approach

Initially focus on exposing existing metrics rather than introducing new experiment logic.

Candidate metrics include:

* Graph-memory statistics (e.g. graph/object relationships).
* Matching statistics.
* Possible match counts.
* Episode-level learning statistics.
* Additional learning-module statistics already available in experiment outputs.

The exact set of metrics can be refined based on maintainer feedback.

## Non-Goals

This RFC does not propose:

* Changes to Monty's learning algorithms.
* Changes to matching logic.
* Changes to graph construction.
* New evaluation procedures.

The focus is strictly on experiment instrumentation and observability.

## Open Questions

1. Which unsupervised-learning metrics are currently considered most important by maintainers?
2. Are there existing WandB dashboards or conventions that new metrics should follow?
3. Should the initial implementation focus on parity with existing CSV outputs, or should it introduce additional WandB-specific visualizations?

## Expected Outcome

Researchers and contributors will have improved visibility into unsupervised learning behavior through WandB with minimal impact on the existing codebase.
