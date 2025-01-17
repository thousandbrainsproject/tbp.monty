---
title: Model-Based Policy to Recognize an Object Before Moving onto a New Object
---

When there are multiple objects in the world (including different parts of a compositional object), it is beneficial to recognize the object currently observed (i.e. converge to high confidence) before moving onto a new object.

Such a policy could have different approaches, such as moving back if an LM believes it has moved onto a new object (reactive), or using the model of the most-likely-hypothesis to try to stay on the object (pro-active), or a mixture of these. In either case, these would be instances of model-based policies.

