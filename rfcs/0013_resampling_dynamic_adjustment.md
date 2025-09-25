- Start Date: 2025-07-22
- RFC PR: [#390](https://github.com/thousandbrainsproject/tbp.monty/pull/390)

# Summary

This RFC extends [RFC-9](https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0009_hypotheses_resampling.md), which introduced hypotheses resampling as a hypotheses updater function in `EvidenceGraphLM`.

Instead of using fixed heuristics (e.g.,`hypotheses_count_multiplier` and `hypotheses_existing_to_new_ratio`) to
determine the amount of resampling and hypotheses deletion at every step, this RFC proposes to have the
resampling informed by more intelligent runtime heuristics, such as Monty's confidence in its prediction
and step-wise prediction error. Dynamic adjustment of the hypothesis space will enable Monty to sample
hypotheses only as needed, which is expected to improve Monty's run time as the hypothesis space can be smaller.

Consider a scenario where, after a few steps of observing an object, Monty is confident in its prediction of the object
ID and its pose. In such a case, continuing to resample will only slow down the recognition as it adds more computational
overhead due to processing the unnecessarily resampled hypotheses at every step. However, if this high confidence is
challenged with an unexpected observation (e.g., object swap), Monty should be smart enough to sample more hypotheses
as it tries to make sense of this high prediction error.

This RFC proposes the following:
1. Redefine the resampling parameters to decouple the amount of resampled hypotheses from deleted hypotheses. At the moment,
these are tightly coupled to satisfy the hypothesis space size multiplier parameter.
2. Allow resampling to be informed by Monty's step-wise performance metrics (e.g., confidence and prediction error), this will
enable dynamic resizing of the hypothesis space.


## Resampling Parameters Decoupling

The resampling procedure is currently defined by a few parameters that tightly couple resampling to hypotheses deletion.
In other words, the user defines the `hypotheses_count_multiplier` parameter to communicate how the hypothesis space size
should be scaled every step, and defines the `hypotheses_existing_to_new_ratio` to communicate how much of the existing
hypotheses should be resampled. This design has several limitations:
- We do not want fixed, user-defined resampling. These parameters fully define the resampling behavior, leaving no room for
resampling to be informed by Monty. We want Monty to control these parameters during runtime.
- Removing a multiplier parameter that requires the hypothesis space to be of a specific size (i.e., `hyp_space_size * multiplier`)
allows for the possibility of these objects hyp. spaces to scale independently of each other. For instance, if deletion is
based on a fixed evidence slope threshold, more hypotheses would be deleted from unlikely objects' spaces. This
independent scaling is not possible with a global multiplier parameter applied to all hypothesis spaces.
- While the existing parameters give us control over how to scale the hypothesis space, they cannot express all the desired resampling
behaviors. For example, for a space of 100 hypotheses and a multiplier of 0.5 (desired
hypothesis space of 50), we can use `hypotheses_existing_to_new_ratio` to control what portion of the 50 we want to be resampled
vs. extracted from existing hypotheses, but we cannot e.g. choose to resample 80 and delete 130. Decoupling allows for defining those
quantities independently.
- A fixed user-defined multiplier (!= 1) is not ideal as it leads to continuous growth or decay of the hypothesis space size each step until
it becomes quite large or diminishes to 0 hypotheses. The size will also vary exponentially because it's based off
the size from the previous time step rather than a fixed value (i.e., number of nodes in the object graph).

Therefore, we propose a new strategy for parameterizing the resampling procedure. Instead of controlling the hypotheses count multiplier, we
can directly define parameters to specify how many hypotheses to be resampled and which hypotheses are to be deleted.
Additionally, since this approach removes the concept of a fixed target hyp. space size, we must define minimum and maximum hypothesis
space sizes to prevent unbounded growth or collapse.


## Monty-Informed Resampling

The resampling procedure consists of two main components: (1) how many new hypotheses to sample, and (2) which hypotheses
to delete. Unlike the current implementation, we propose to have Monty control these details as follows:


#### The resampling parameter

This parameter would rely on Monty's confidence of the existing hypotheses and its prediction
errors to determine how many new hypotheses are to be resampled.
Confidence can be considered a long-term integration signal. It is built up over time as hypotheses
accumulate evidence. This cumulative score, when paired with something like `x_percent_threshold`,
can provide a strong estimate of Monty's belief.
On the other hand, a prediction error is a short-term signal. It conveys how well Monty's current
hypotheses match the latest observation, or the last few observations (to smooth out noise).
One useful idea is to combine these two quantities into a single metric; Monty’s surprise (placeholder name).
This "surprise" metric captures how unexpected the current observation is, given Monty’s current belief:

- A high prediction error in high-confidence hypotheses (i.e., high surprise)
should trigger aggressive resampling.
- A high prediction error in low-confidence hypotheses (i.e., low-to-moderate
surprise) might require only mild resampling.
- A low prediction error in high-confidence hypotheses could mean no resampling is
needed.


We can use existing `x_percent_threshold` logic to calculate confidence, whereas the prediction error can 
easily be derived from the evidence score based on morphological and non-morphological features match. Note
that the exact formula for calculating "Monty’s surprise" is considered an implementation detail and is
therefore out of scope for this RFC.


#### The deletion parameter

This will likely be a fixed threshold on the evidence slope. A fixed threshold
would allow different objects to scale their hypothesis spaces independently, which is a desirable effect.
That said, we could also consider more adaptive approaches. For example, the threshold could
be adjusted dynamically to maintain a fixed number of top-performing hypotheses as a
function of the object graph size (e.g., always keep the top 70 hypotheses for an object graph with
100 nodes). These adaptive alternatives are worth exploring, but the simplest fixed-threshold option is the
my preferred starting point.

Another possible approach is to also use Monty’s surprise to guide hypothesis deletion.
In this case, low surprise could serve as a signal to prune the hypothesis space.
While this does introduce some coupling between deletion and resampling, both responding to
the same surprise signal, it still allows for independent heuristics to guide each process.
This is fundamentally different from the current implementation, where deletion and resampling hypotheses
counts are tightly linked through shared parameters like the `hypothesis_count_multiplier`.

##### Note on biological plausibility

Biological neurons have different time constants for integrating synaptic inputs.
Different neuron types can have different integration properties, for example, inhibitory tend to integrate and spike
2-3 times faster[^1] than excitatory neurons.
In addition, different types of synaptic channels (e.g. AMPA, NMDA) can have different temporal integration windows compared
to one-another, as well as to the neuron as a whole.
Finally, there is the added flexibility of dendritic compartments which might have different integration properties[^2] ,
or having a separate population of neurons integrating (pooling) a summary representation.
All-in-all its both unsurprising that we are finding ourselves in need of at-least two measures of evidence at different
time-scales, and this seems perfectly consistent with what biology could implement.


# Plan of Work

This is how I've planned the subtasks of this RFC, if approved.

1. Decouple resampling parameters
    - Attempt to reproduce existing benchmarks by setting deletion and resampling to 10%
2. Change deletion parameter to a fixed threshold and resampling parameter to Monty's surprise.
    - Run unsupervised inference benchmarks and hyperparameter sweep on YCB benchmarks

# Complementary Work on Prediction Error and Resampling

This work is complementary to parallel work on resampling.
In particular, The [RFC on Intelligent Resampling in Monty](https://github.com/thousandbrainsproject/tbp.monty/pull/366) is
working out the conceptual details of how prediction errors resulting from observations off an object can inform resampling.
These observations correspond to seeing another object, or not sensing anything at all (e.g. the finger
held in empty space), resulting in a high prediction error.

Notably, this parallel work is exploring how the CMP signal from a SM should be formulated and processed
such that these kinds of observations result in a prediction error.
The RFC here however considers how prediction errors should be integrated over time, and the resulting resampling
strategies that follow from them.


[^1]: Markram, Henry, et al. "Interneurons of the neocortical inhibitory system." Nature reviews neuroscience 5.10 (2004): 793-807.

[^2]: Wright, William J., Nathan G. Hedrick, and Takaki Komiyama. "Distinct synaptic plasticity rules operate across dendritic compartments in vivo during learning." Science 388.6744 (2025): 322-328.
