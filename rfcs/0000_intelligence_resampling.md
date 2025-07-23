- Start Date: 2025-07-08
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Intelligent Resampling in Monty

This is a high-level RFC on intelligence resampling in Monty, considering the below three questions:

1. How can we re-anchor hypotheses to model points for robustness to noise and distortions?
2. How can we implement and test resampling informed by out-of-reference-frame observations?
3. How can we use prediction errors in off-object observations to eliminate hypotheses? 

## How can we re-anchor hypotheses to model points for robustness to noise and distortions?

"Re-anchoring" means updating the `locations` or `poses` of the `Hypotheses` object to **an existing point** in the object model. The re-anchoring mechanism should be informed by feature observations. 

An example of "distortion" is like the bent TBP logo vs. "standard" TBP logo. In the case of a distortion, a feature / object part / morphology is present at a different location and rotation than would be expected according to the original model. (TODO: include image for clarity and reference for readers). By noise, we refer to errors in location estimates arising from imperfect path integration (TODO: also include image). For example, if relying on optic flow, proprioception, or inertial measurement units, then all of these will result in imperfect estimates of movement displacement and direction.

### Current Status and Objective

Below is the `Hypotheses` class implemented in `tbp.monty==0.8.0`. 
```python
@dataclass
class Hypotheses:
    """Set of hypotheses consisting of evidence, locations, and poses.

    The three arrays are expected to have the same shape. Each index corresponds to a
    hypothesis.
    """

    evidence: np.ndarray # numpy array of confidence scores
    locations: np.ndarray # numpy array of 3D positions in the object's reference frame
    poses: np.ndarray # numpy array of 3x3 rotation matrices, shape
```

Currently, when a hypothesis is initialized, its `poses` attribute is fixed and does not change. While we can create hypotheses with different poses, these new hypotheses will start from zero evidence and will take time to accumulate evidence. So, _we want to implement a mechanism to update the location or pose of a hypothesis without affecting the evidence value_.

### Matching Features 

At a high level, re-anchoring a hypothesis will involve trying to match the **observed features** to **stored features** in object models. If a unique match is found, then we may simply update the location and pose of the hypothesis to that stored in the object model, e.g.

```python
# updating the i-th hypothesis
hypothesis.locations[i] = object_model_location 
hypothesis.poses[i] = object_model_pose
```

Below we work through some questions and implications:

- **How can we meaningfully compare features?**

This depends on specific feature, but for now, we can leverage the existing code in `check_feature_change()` in `FeatureChangeSM` class to compare features. Specifically, as of `tbp.monty==0.8.0`, we have the following definition of distances for `hsv` and `pose_vectors`:

```python
# HSV (line 717)
hue_d = min(abs(current_hue - last_hue), 1 - abs(current_hue - last_hue))

# Pose Vectors (line 729)
angle_between = get_angle(last_feat[0],current_feat[0],)
```

To be considered a "match", _all_ $N$ features should be similar within some threshold. _Why all?_ If, let's say, `hsv` matches but `pose vectors` do not, then this is not informative for an object that has a uniform color all over (e.g. red mug). 

**Note**: Any subsequent extension of observed features in Monty should consider a meaningful distance metric. Note that the above two are examples where L2 distance is not suitable.  

- **What if there are multiple matches?**

This means that the observed set of features exists in multiple locations of the object model, i.e. the observed features are not very distinct. The larger the number of features we are measuring (i.e. larger $N$), the less likely "collisions" are likely to occur. It is also not necessary to re-anchor at every step - we may choose not to re-anchor if there are multiple matches, and only re-anchor if there is a unique match when the sensor if observing a patch containing a unique feature. 

**Note**: We may wish to "mark" these nodes in graph object model when a match has been made to indicate that it contains a unique feature. These may be potentially important nodes or landmarks we may wish to keep, especially if we want to learn a sparse object model. It may also help with computational complexity (see Question below) of the task.

- **What if there are no matches?**

I think this scenario is possible during inference when we add sensor noise that is greater than some threshold for a match. I think there are a couple of ways to proceed, depending on the evidence score:

1. Case 1: Evidence of the hypothesis is low
- Perhaps the hypothesis is uncertain about the object and is actually incorrect, hence leading to no match. 
2. Case 2: Evidence for the hypothesis is high
- We may want to temporarily increase the threshold to see if we get a match. The larger we have to increase for a match, the larger we could penalize the evidence. 

The simplest implementation shouldn't worry about handling a "no match" case (i.e. just do not perform re-anchoring), but note it could potentially be used as a tool to decrease evidence scores.

- **If our models are sparse, then it might actually throw us off from the correct estimate of the location. How can we mitigate this?**

- **If we are doing unsupervised learning, what happens if we re-anchor?**

- **Computational Complexity**

Because re-anchoring involves comparing of features, this could be costly as a function of: (1) number of features we are observing (the more features we are observing, the more "distance" calculations we will need to do), and (2) number of nodes in object model (the more number of nodes, the more number of comparisons to make).

Currently, case 1 is not much of a concern as we only have `hsv` and `pose_vectors`, but future implementions should try to consider efficient distance calculations AND prioritizing to compare a subset of features. The intuition for the latter is that for an object like a red mug, there is no real benefit of comparing `hsv` between observed and stored because they will always likely match. 

Case 2 is much more of a concern now until we have a more sparse model. We should first benchmark how long it takes to do comparison against ~2,000 points in an object model. One heuristic we could use is just to compare to points within some $\epsilon$-radius of the location, assuming we are re-anchoring to a nearby point. This may be valid if we frequently re-anchor and update the hypotheses, so we never accumulate large errors. Finally, we could try to prioritize the search by first comparing to nodes that were "marked" as unique. 





