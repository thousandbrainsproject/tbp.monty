---
title: Results from Alternative Implementations
---
# FeatureGraphLM (Old LM)

For comparison, the following results were obtained with the previous LM version (FeatureGraphLM). These results have not been updated since September 22nd 2022. Results here were obtained with more densely sampled models than results presented for the evidence LM. This means it is less likely for new points to be sampled. With the current, more sparse, models used for the EvidenceLM the FeatureGraphLM would have reduced performance.  
Runtimes are reported on laptop with 8CPUs and no parallelization.

| Experiment                     | # objects | tested rotations | new sampling | other                          | Object Detection Accuracy | Rotation Error | Run Time        |
| ------------------------------ | --------- | ---------------- | ------------ | ------------------------------ | ------------------------- | -------------- | --------------- |
| full_rotation_eval_all_objects | 77 YCB    | 32 (xyz, 90)     | no           |                                | 73.62%                    |                | 4076min (68hrs) |
| full_rotation_eval             | 4 YCB     | 32 (xyz, 90)     | no           |                                | 98.44%                    | 0.04 rad       | 5389s (89min)   |
| partial_rotation_eval_base     | 4 YCB     | 3 (y, 90)        | no           |                                | 100%                      | 0 rad          | 264s (4.4min)   |
| sampling_learns3_infs5         | 4 YCB     | 3 (y, 90)        | yes          |                                | 75%                       | 0.15 rad       | 1096s (18.3min) |
| sampling_3_5_no_pose           | 4 YCB     | 3 (y, 90)        | yes          | don't try to determine pose    | 100%                      | -              | 1110s (18.5min) |
| sampling_3_5_no_pose_all_rot   | 4 YCB     | 32 (xyz, 90)     | yes          | don't try to determine pose    | 96.55%                    | -              | 1557s (25.9min) |
| sampling_3_5_no_curv_dir       | 4 YCB     | 3 (y, 90)        | yes          | not using curvature directions | 91.67%                    | 0.03 rad       | 845s (14min)    |

## Explanation of Some of the Results

- Why `full_rotation_eval_all_objects` so much worse than full_rotation_eval?  
  The difference is that we test 77 objects instead of just 4. There are a lot of objects in the YCB dataset that are quite similar (i.e. a_cups, b_cups, ..., e_cups) and if we have all of them in memory there is more chance for confusion between them. Additionally, the 4 objects in full_rotation_eval are quite distinguishable and have fewer symmetries within themselves than some of the object objects do (like all the balls in the YCB dataset).
- Why is `full_rotation_eval_all_objects` so slow?  
  In this experiment, we test all 77 YCB objects. This means that we also have to store models of all objects in memory and check sensory observations against all of them. At step 0 we have to test `#possible_objects` `#possible_locations`  `#possible_rotation_per_location` which in this case is roughly 78 x 33.00 x 2 = 514.800. If we only test 4 objects this is just 26.400. Additionally, we test all rotation combinations along all axes which are 32 combinations.
- Why do new sampling experiments work better if we don't determine the pose?  
  First, the algorithm comes to a solution faster if the terminal condition does not require a unique pose. This makes it less likely to observe an inconsistent observation caused by the new sampling. Second, we don't have to rely much on the curvature directions (used to inform possible poses otherwise) which can be quite noisy and change fast with new sampling.
- Why do the new sampling experiments take longer to run?  
  The experiments reported here use step size 3 to learn the object models and step size 5 to recognize them. This ensures that we sample completely new points on the object. However, building a model with step size 3 leads to them containing a lot more points. This scales the number of hypotheses that need to be tested (approx. 4 x 12.722 x 2 = 101.776)