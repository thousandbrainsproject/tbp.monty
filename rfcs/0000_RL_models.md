- Start Date: 2025-01-04
- RFC PR: 

NOTE: While this RFC process is document-based, you are encouraged to also include visual media to help convey your ideas.

# Summary

> A brief explanation of the feature.
- Learn Policy using RL

**Idea** - to make learning and task execution more brain-like using graphs and without focusing on deep neural networks.

The brain has a model of the world and almost always "models" and predicts something, so let's clarify the terms model-free and model-based in RL. In RL terms, "model-free" doesn't mean "no models at all," but "no explicit model of the world's transitions used for planning at the moment of choice." When the task is familiar, the choice is almost a direct "situation → action" without explicit planning.

Let's define terms:  
⦁ World model: an explicit representation of transitions and rewards (s, a→s2, r) that can be scrolled forward for "mental" planning.  
⦁ Policy / Value model: a learned direct approximation of Q(s,a) that allows one to choose an action without scrolling into the future.

**Models** are stored as graphs, where the vertices are states and the edges are transition actions.

**State** (s) - this is a code/key as a concatenation of context  
    What can be included in the state code:  
    ⦁ Sensory cues of scene objects.  
    These can include images, sounds, smells, kinesthetics, etc.
    The scene code is formed as a concatenation of the codes of the sensory components included in the scene.  
    ⦁ Spatial code (scenes / locations / coordinates / distances).  
    For testing, coordinates / distances can be taken from an external environment emulator, such as MuJoCo. Use a primitive scenegraph based on scene object relationships (left_of / right_of / above / below / inside / overlaps).  
    ⦁ Temporal/ordinal context (e.g., recent k steps).   
    For testing, use a sequence of recent states and actions.  
    ⦁ Internal state: goal, motivation.  
    The goal is the desired state as an object of sensory input, a target image, a vocal description of what to do.
    Motivation is a personal, subjective factor that can be defined as a novelty factor.
    The weight of the components is flexibly modulated.  
    Each context component has its own encoding implementation. For testing, existing neural network implementations can be used.  
    For the project, a transition to Monty Cortical Messaging Protocol, State class, SDR or similar will be necessary. 

**Action** (a) - An action from a given state that transitions the agent to another state. For example, an action in an external environment emulator format (MuJoCo and similar).


**Reward** (r) - An immediate reward for a completed action.
First, the reward is updated from the real environment after the action is completed.
Reward calculation depends on the context (task, situation). For example, in robotics, this is usually the distance from the grasper to the target. A heuristic can be developed to calculate the reward, or the reward can be taken from an environment emulator.
During planning, the reward is taken from the model, which was previously saved from the real environment.	

**World model**: transition graph (s, a→s2, r).
Transitions are formed as a result of interaction with the external environment.
Nodes are states, edges are actions from one state that lead to another state, and reward is an attribute of an action edge.

**Policy / Value model**: Action evaluation graph Q(s,a). The current evaluation of Q(s,a) is the current "prediction" of the value of action a in state s.
Q(s,a) is updated during the planning process (Dyna style) and as a result of executing actual actions.
The update occurs according to the Bellman equation, when we pull Q(s,a) to the target value obtained from the reward r and the evaluation of the next state s'.
The policy model is the same graph as the world model, only with different attributes.

**Search and update (s,a)**  
To search and update the model, previously saved states and actions (s,a) are taken into account.
An absolute match is not a good approach, but rather a search for the most similar states based on their state codes.
For fast approximate nearest neighbor search (ANN), an index can be built using the HNSW (Hierarchical Navigable Small World Graph) algorithm.  
Search algorithm for (s,a):  
Find the most similar past states above a certain threshold using the index.  
If none are found, add the state to the graph and index the state code for fast proximity searching.  
If there are any, iterate over the actions performed in this state and select the most similar one. Save/update the average state code in the graph and in the index.  
The required parameters are added as attributes to the graph vertices and edges.


**Planning**  
For new/complex goals, planning is used using a world model: "what will happen if..." and selecting a step with the best expected outcome. It's a "model-based" approach.  
If a state has ready-made actions with a high Q (higher than the threshold), then there is high confidence, and no planning is needed. It's a "model-free" approach.

Below are the steps and pseudocode where we combine planning and interaction with the real environment:  
```
1) Start from the initial state s_real = s0_real        
for step in range(steps): # steps - number of interaction steps
    for iter in range(iters): # iters - the number of planning iterations for a specific interaction step
        s_plan = s_real
        for depth in range(depth_max): 
            # In a loop, we move depth-first and breadth-first.
            The prefrontal cortex (PFC) generates multiple candidates.
            To select actions during planning, we determine the possible actions in the current state. We use all actions that have ever been encountered from this state s in the world model. If the model doesn't yet know any actions, we can't plan yet.
            We use the metrics N(s), N(s,a) — the number of visits and executions of an action from a state.
            For the planning iteration, we take unused actions and select the top_k most promising ones according to the heuristic metric adv(s). For example, adv(s) could be the absolute value of the inverse distance from the grabber to the target. 
            2) For each child, there is a separate rollout and backup.
            The hippocampus "plays out" outcomes.
            for _, a_child, s2_child in top_k:
                The orbitofrontal cortex (OFC) and ventromedial PFC (vmPFC) transform information about action options into "subjective value".
                r_child = reward_model(s, a_child, s2_child)
                # We are moving towards a state according to the model with greater adv(s)
                G_tail = rollout_value(s2_child) 
                # backup in the model
                G_ret = r_child + gamma * G_tail
                delta = G_ret - Q[(s, a_child)]
                Q[(s, a_child)] += lr * delta
                Nsa[(s, a_child)] += 1
                # Add to offline replay
                push_replay(abs(delta), s, a_child)
                Prioritized offline replay: a queue with priority |delta| is maintained;
                After each online step and plan backups, candidates are added;
                offline_replay runs the K with the largest prediction errors |delta|,
                propagating changes to ancestors along the known graph or through learned predecessors.
                This is analogous to backpropagation in neural networks.
                
            3) Select the best action in state (s) taking into account backups according to the world model
                The basal ganglia (BG) decide which plan to proceed with, when to slow down/stop, and how to adapt future choices based on rewards and errors.
                for a in acts:
                    s2 = transition_model_step(s, a)
                    Selection criteria:
                    3.1) q = Q[(s, a)]
                    3.2) ucb = self.beta * math.sqrt(math.log(1 + Ns[s]) / (1 + Nsa[(s, a)]))
                    # UCB (Upper Confidence Bound) bonus is an addition to the action score that implements “optimism under uncertainty” and balances exploration/exploitation.
                    3.3) adv = alpha * adv(s2)
                    score = q + ucb + adv

            4) Perform the best action, make a transition based on the world model, and calculate the reward.
            s2 = transition_model_step(s, a)
            r = reward_model(s, a, s2)
            Adding action and reward to the path (s, a, r)
            
            5) Move on to the next step in depth
            s_plan = s2_plan 
            We go through all the steps of the leaves in depth. 
            
        6) Completing the planning iteration
        Dopamine (VTA) changes the weights in corticostriatal synapses so that the desired response is strengthened in the appropriate context.
        Rollout from the last leaf and backup along the entire path
        G_tail = rollout_value(s_plan)
        G_ret = G_tail
        for (ss, aa, rr) in reversed(path):
            G_ret = rr + gamma * G_ret
            delta = G_ret - Q[(ss, aa)]
            Q[(ss, aa)] += lr * delta
            Nsa[(ss, aa)] += 1
            push_replay(abs(delta), ss, aa)
            

    7) After preliminary planning, selecting the action based on the Policy model with the maximum value Q(s, a)
    The basal ganglia (BG) decide which plan to proceed with, when to slow down/stop, and how to adapt future choices based on rewards and errors.
    
    8) Performing an action in a real environment, an example from MuJoCo
    obs, r, terminated, truncated, info = env.step(a)
    
    9) After the action in the real environment we execute:
    Dopamine (VTA) changes the weights in corticostriatal synapses so that the desired response is strengthened in the appropriate context.
    # One-step TD update
    target = r
    nxt_acts = available_actions(s2)
    target += gamma * max(Q[(s2, aa)] for aa in nxt_acts)
    delta = target - Q[(s, a)]
    Q[(s, a)] += lr * delta
    Ns[s] += 1
    Nsa[(s, a)] += 1
    push_replay(abs(delta), s, a)

    # Prioritized backups of predecessors according to the current World model
    Also, during sleep, the brain “replays” (reactivates) recent experiences and, due to this, strengthens the necessary connections.
    cnt = 0
    while replay_pq and cnt < limit:
        negp, (s, a) = heapq.heappop(replay_pq)
        cnt += 1
        # Update Q(s,a) as one step backup
        s2 = transition_model_step(s, a)	
        r = reward_model(s, a, s2)
        target += gamma * max(Q[(s2, aa)] for aa in nxt_acts)
        delta = target - Q[(s, a)]
        Q[(s, a)] += lr * delta
        # Add the predecessors of s - these are (sp, ap), leading to s
        preds = model.get_predecessors(s)
        # For each predecessor, we estimate the priority (new |δ|)
        for (sp, ap) in preds:
            delta_pred = target_pred - Q[(sp, ap)]
            push_replay(abs(delta_pred), sp, ap)
    
    10) Move on to the next step in a real environment
    s_real = s2_real

    11) Continue until reach the goal or reach the maximum number of steps.
```

This proposal is not completed approach / implementation and there are many open questions. I've made some toy testing but of course It needs more research and testing iterations.  


# Motivation

> Why are we doing this? What use cases does it support? What is the expected outcome? Which metrics will this improve? What capabilities will it add?

https://thousandbrainsproject.readme.io/docs/learn-policy-using-rl

>
> Please focus on explaining the motivation so that if this RFC is not accepted, the motivation could be used to develop alternative solutions. In other words, enumerate the constraints you are trying to solve without coupling them too closely to the solution you have in mind.

# Guide-level explanation
> Explain the proposal as if it was already included in Monty and you were teaching it to another Monty user. That generally means:
>
> - Introducing new named concepts.
> - Explaining the feature largely in terms of examples.
> - Explaining how Monty developers should *think* about the feature and how it should impact the way they use Monty. It should explain the impact as concretely as possible.
> - If applicable, provide sample error messages, deprecation warnings, or migration guidance.
> - If applicable, describe the differences between teaching this to existing Monty users and new Monty users.
> - If applicable, include pictures or other media if possible to visualize the idea.
> - If applicable, provide pseudo plots (even if hand-drawn) showing the intended impact on performance (e.g., the model converges quicker, accuracy is better, etc.).
> - Discuss how this impacts the ability to read, understand, and maintain Monty code. Code is read and modified far more often than written; will the proposed feature make code easier to maintain?
>
> Keep in mind that it may be appropriate to defer some details to the [Reference-level explanation](#reference-level-explanation) section. 
>
> For implementation-oriented RFCs, this section should focus on how developer contributors should think about the change and give examples of its concrete impact. For administrative RFCs, this section should provide an example-driven introduction to the policy and explain its impact in concrete terms.

It seems that the reinforcement learning (RL) module can work alongside existing Monty functionality.  
Integration can be done at the level of Monty Cortical Messaging Protocol.  
RL input will be context/state in the form of Monty State class.  
RL outputs actions, which are passed to and executed in Monty Motor System.  
Unique RL entities not present in the current Monty include World and Policy models.  

For testing toy examples, I used the external MuJoCo environment without Monty integration.
Simulator_habitat looks resource-intensive and could not be installed on my local Ubuntu.


# Reference-level explanation

> This is the technical portion of the RFC. Explain the design in sufficient detail that:
>
> - Its interaction with other features is clear.
> - It is reasonably clear how the feature would be implemented.
> - Corner cases are dissected by example.
>
> The section should return to the examples from the previous section and explain more fully how the detailed proposal makes those examples work.

# Drawbacks

> Why should we *not* do this? Please consider:
>
> - Implementation cost, both in terms of code size and complexity
> - Whether the proposed feature can be implemented outside of Monty
> - The impact on teaching people Monty
> - Integration of this feature with other existing and planned features
> - The cost of migrating existing Monty users (is it a breaking change?)
>
> There are tradeoffs to choosing any path. Please attempt to identify them here.

# Rationale and alternatives

> - Why is this design the best in the space of possible designs?
> - What other designs have been considered, and what is the rationale for not choosing them?
> - What is the impact of not doing this?

# Prior art and references

> Discuss prior art, both the good and the bad, in relation to this proposal.
> A few examples of what this can include are:
>
> - References
> - Does this functionality exist in other frameworks, and what experience has their community had?
> - Papers: Are there any published papers or great posts that discuss this? If you have some relevant papers to refer to, this can serve as a more detailed theoretical background.
> - Is this done by some other community and what were their experiences with it?
> - What lessons can we learn from what other communities have done here?
>
> This section is intended to encourage you as an author to think about the lessons from other frameworks and provide readers of your RFC with a fuller picture.
> If there is no prior art, that is fine. Your ideas are interesting to us, whether they are brand new or adaptations from other places.
>
> Note that while precedent set by other frameworks is some motivation, it does not on its own motivate an RFC.
> Please consider that Monty sometimes intentionally diverges from common approaches.

# Unresolved questions

> Optional, but suggested for first drafts.
>
> What parts of the design are still TBD?

# Future possibilities

> Optional.
>
> Think about what the natural extension and evolution of your proposal would
> be and how it would affect Monty and the Thousand Brains Project as a whole in a holistic way.
> Try to use this section as a tool to more fully consider all possible
> interactions with the Thousand Brains Project and Monty in your proposal.
> Also consider how this all fits into the future of Monty.
>
> This is also a good place to "dump ideas" if they are out of the scope of the
> RFC you are writing but otherwise related.
>
> If you have tried and cannot think of any future possibilities,
> you may simply state that you cannot think of anything.
>
> Note that having something written down in the future-possibilities section
> is not a reason to accept the current or a future RFC; such notes should be
> in the section on motivation or rationale in this or subsequent RFCs.
> The section merely provides additional information.