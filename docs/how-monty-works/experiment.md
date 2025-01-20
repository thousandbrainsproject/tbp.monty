---
title: Experiment
---
During an experiment in the Monty framework an agent is collecting a sequence of observations by interacting with an environment. We distinguish between training (internal models are being updated using this sequence of observations) and evaluation (the agent only performs inference using already learned models but does not update them). The MontyExperiment class implements and coordinates this training and evaluation of Monty models.

In reality an agent interacts continuously with the world and time is not explicitly discretized. For easier implementation we use steps as the smallest increment of time. Additionally, we divide an experiment into multiple episodes and epochs for easier measurement of performance. Overall, we **discretize time in the three ways** listed below.

![Three ways time is discretized in Monty is into steps (one movement and one observation), episodes (take as many steps as needed to reach the terminal condition of the environment such as recognizing an object or completing a task), and epoch (cycle through all objects/scenarios in the dataset once).](../figures/how-monty-works/step_episode_epoch.png)


<br />

- **Step:** taking one action and receiving one observation. There are different types of steps that track more specifically whether learning module updates were performed either individually for each LM or more globally for the Monty class.

  - **monty_step** (model.episode_steps total_steps): number of observations sent to the Monty model. This includes observations that were not interesting enough to be sent to an LM such as off-object observations. It includes both matching and exploratory steps.

  - **monty_matching_step** (`model.matching_steps`): At least one LM performed a matching step (updating its possible matches using an observation). There are also exploratory steps which do not update possible matches and only store an observation in the LMs buffer. These are not counted here.

  - **num_steps** (`lm.buffer.get_num_matching_steps`): Number of matching steps that a specific LM performed.

  - **lm_step** (`max(num_steps)`): Number of matching steps performed by the LM that took the most steps.

  - **lm_steps_indv_ts** (`lm.buffer["individual_ts_reached_at_step"]`): Number of matching steps a specific LM performed until reaching a local terminal state. A local terminal state means that a specific LM has settled on a result (match or no match). This does not mean that the entire Monty system has reached a terminal state since it usually requires multiple LMs to have reached a local terminal state. For more details see section [Terminal Condition](doc:evidence-based-learning-module#terminal-condition)

- **Episode:** putting a single object in the environment and taking steps until a terminal condition is reached, like recognizing the object or exceeding max steps.

- **Epoch:** running one episode on each object in the training or eval set of objects.

In the long term, we might remove the episode and epoch chunking and simply have the agent continuously interact with a given environment without resetting it. Removing the step discretization of time will probably not be possible (maybe with neuromorphic hardware?) but we can simulate continuous time by making the step increments tiny and utilizing the different step types (like only sending an observation to the LM if a significant feature change was detected).

| List of all experiment classes                 | Description                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MontyExperiment**                            | Abstract experiment class from which all other experiment classes inherit. The experiment class handles the initialization of all Monty components. It also takes care of logging and the highest level calls to _train_, _evaluate_, _pre_epoch_, _run_epoch_, _post_epoch_, _pre_episode_, _run_episode_, _post_episode_, _pre_step_, and _post_step_. |
| **MontyObjectRecognitionExperiment**           | Experiment class to test object recognition. The current default class for Habitat experiments. Saves the target object and pose for logging. Also contains some custom terminal condition checking and online plotting.                                                                                                                                 |
| **MontyGeneralizationExperiment**              | Same as previous but removes the current target object from the memory of all LMs. Can be used to test generalization to new objects.                                                                                                                                                                                                                    |
| **MontySupervisedObjectPretrainingExperiment** | Here we provide the object and pose of the target to the model. This way we can make sure we learn exactly one graph per object in the dataset and use the correct pose when merging graphs. This class is used for model pre-training.                                                                                                                  |
| **DataCollectionExperiment**                 | Just runs exploration and saves results as .pt file. No object recognition is performed.                                                                                                                                                                                                                                                                 |
| **ProfileExperimentMixin**                     | Can be added to any experiment class to add a profiler.                                                                                                                                                                                                                                                                                                  |