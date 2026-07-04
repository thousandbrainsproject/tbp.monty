---
title: Clean up and Simplify Voting
description: Simplify some of the voting code to make it more efficient and easier to understand, and explore filtering by object orientations.
rfc: optional
estimated-scope: medium
improved-metric: speed, accuracy, learning-experience
output-type: PR, documentation
skills: python, monty, refactoring
contributor: 
status: open
---

The way we do voting is a bit nested and overly complex in the code. We could simplify it and add some clarifying documentation around it.

Additionally, we could use the hypothesized orientation earlier on to filter out how votes are  applied, thereby speeding up the process. The existing voting process relies on a K-D tree search method to identify hypotheses that are near each other in space. This accounts for the hypothesized orientation of objects, but is computationally quite expensive. One way to ameliorate this would be to first compare the hypothesized object orientations across the LMs. Only for votes where there is agreement at the orientation level would we then proceed to the K-D tree search. Importantly, the voting check for orientation agreement should be computationally cheap; this will likely be even more so once we move to better representations of symmetry.

Voting on rotations before sharing hypothesis locations may also improve accuracy. The intuition is that with our current voting approach, all of the votes exist in a single, shared space after being transformed by the orientations and relative sensor displacements. This means that votes can still align with hypotheses even where their object orientations are inconsistent. In other words, the vote could have a spurious neighbor in the receiving LMs hypothesis space, and thereby - incorrectly - provide positive evidence during the voting process. This is much like if we allowed voting across reference frames (a mug vs. a banana model) - the sheer number of points means that there is a chance of a collision occurring. This is particularly risky where thousands of votes are involved, which will be likely as the scale of Monty grows to more LMs.

Finally, the existing voting algorithm requires the construction of a K-D tree search whenever voting occurs. Through appropriate formulation of the voting process, including filtering by orientations, it may be possible to eliminate this requirement, and thereby reduce the computational cost of voting.