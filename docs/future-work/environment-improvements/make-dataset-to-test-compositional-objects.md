---
title: Make Dataset to Test Compositional Objects
---

We have developed an initial dataset based on setting a dinner-table with a variety of objects. For example, the objects can be arranged in a normal setting, or aligned in a row (i.e. not a typical dinner-table setting). Similarly, the component objects can be those of a modern dining table, or those from a "medieval" time-period. As such, this dataset can be used to test the ability of Monty systems to recognize compositional objects based on the specific arrangement of objects, and to test generalization to novel compositions.

By using explicit objects to compose multi-part objects, this dataset has the advantage that we can learn on the component objects in isolation, using supervised learning signals if necessary.

However, we would eventually expect compositional objects to be learned in an unsupervised manner. When this is consistently possible, we can consider more diverse datasets where the component objects may not be as explicit. At that time, the challenges described in [Figure out Performance Measure and Supervision in Heterarchy](figure-out-performance-measure-and-supervision-in-heterarchy.md) will become more relevant.