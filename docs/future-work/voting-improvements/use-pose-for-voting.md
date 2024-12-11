---
title: Use Pose for Voting
---

Currently we do not send out pose hypotheses when we are voting, however we believe it will be an important signal to use. One complication is that the poses stored for any given LM's object models are arbitrary with respect to other LM's models, as each uses an object-centric coordinate system.

This relates to [Generalize Voting To Associative Connections](./generalize-voting-to-associative-connections.md), which faces a similar challenge.

To make this more efficient, it would also be useful to improve the way we represent symmetry in our object models (see [Improve Handling of Symmetry](../learning-module-improvements/improve-handling-of-symmetry.md)), as this will significantly reduce the number of associative connections that need to be learned for robust generalization.