- Start Date: 2025-07-28
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Conducting Research While Building a Stable Platform

[The vision of the Thousand Brains Project is to develop a platform for building AI and robotics applications using the same principles as the human brain.](https://thousandbrainsproject.readme.io/docs/vision-of-the-thousand-brains-project) To achieve this vision, we pursue the platform work while we continue to conduct active research with [open questions still to be answered](https://thousandbrainsproject.readme.io/docs/open-questions) and already [planned future work](https://thousandbrainsproject.readme.io/docs/project-roadmap).

We believe that the research and the platform are essential to our vision. We want to succeed at both.

## Inherent challenge

There lies an inherent challenge to achieving our vision. From the perspective of supply and demand competition in the marketplace, the activities of research and platform building differ in their characteristics. Some of what makes for good research, directly conflicts with what makes for a good platform, and vice versa.

| Characteristic / Property | Research | Platform (today) | Platform (future) |
|---|---|---|---|
| _Ubiquity_ | rare / slowly increasing consumption | rare / slowly increasing consumption | rapidly increasing consumption / widespread and stabilizing |
| _Certainty_ | poorly understood / rapid increases in learning | poorly understood / rapid increases in learning | rapid increases in use / fit for purpose / commonly understood |
| _Writings about_ | describe the wonder of TBT / focus on building, constructing, awareness, and learning | focus on building, constructing, awareness, and learning | maintenance / operation / installation / features / focus on  use |
| _Market_ | undefined market | undefined / forming market | growing / mature market |
| _Knowledge management_ | uncertain / learning on use | learning on use / operation | learning on operation / known / accepted |
| _Market perception_ | chaotic / domain of experts | chaotic / domain of experts | increasing expectation of use / ordered / trivial |
| _User perception_ | different / confusing / exciting / surprising / leading edge / emerging | different / confusing / exciting / surprising / leading edge / emerging | common / disappointed if not available / standard / expected |
| _Perception in industry_ | competitive advantage / unpredictable / unknown | competitive advantage / unpredictable / unknown | competitive advantage / ROI / advantage through implementation / features |
| _Focus of value_ | high future worth | high future worth | seeking profit / ROI / high profitability |
| _Understanding_ | poorly understood / unpredictable / increasing understanding / development of measures | increasing understanding / development of measures / increasing education / constant refinement of needs and measures | constant refinement of needs and measures / well defined / stable / measurable |
| _Comparison_ | constantly changing / a differential / unstable | learning from others / testing the waters / some evidential support | feature difference / essential / operational advantage |
| _Failure_ | high / tolerated / assumed | moderate / unsurprising but disappointed | not tolerated, focus on constant improvement / operational efficiency and surprised by failure |
| _Market action_ | gambling / driven by gut | gambling / driven by gut | exploring "found" value / market analysis / listening to customers |
| _Efficiency_ | reducing the cost of change (experimentation and prototyping) | reducing the cost of waste (learning and stability) | reducing the cost of waste (learning and stability) / reducing the cost of deviation (volume operations) |
| _Decision drivers_ | culture / analysis & synthesis | culture / analysis & synthesis | analysis & synthesis / previous experience |

While the characteristics of the platform in the future will be quite different from ongoing research in general (which we're planning for), today's platform already differs in two significant aspects: _Failure_ and _Efficiency_.

### Failure

Because of the nature of research, we expect experiments to not work, at least initially. Our tolerance for failure is high. For a platform to be useful, it cannot fail at the same rate as research. While initially, users may be willing to tolerate some failure due to our growing pains, ultimately, the goal is for failure to become surprising and not generally tolerated.

### Efficiency

How we organize our activity can have consequences at how successful we are in execution.

For research, we want to be able to minimize the cost of change. In research, ideas can change rapidly. For an example, one can trace a history and the timeline of research into [Modeling Object Behaviors](https://www.youtube.com/playlist?list=PLXpTU6oIscrn_v8pVxwJKnfKPpKSMEUvU). We want to be able to rapidly try new experiments, new theories, new implementations. We want minimal friction between an idea and experimental data. Since experiments can have a high failure rate, when we write prototype code, it is possible that most or some of it will be thrown away or significantly altered based on what we learn from the experiments.

For a platform, there is only so much change that platform users can absorb. While we intend for the platform to always stay up to date with the latest capabilities discovered through research, those capabilities need a level of stability before being incorporated in the the platform. Even if we were to rapidly iterate on platform internals, any externally facing interface benefits from being stable. When the improvements from a change are significant, change may be easier to adopt. In general, the platform users are focused on solving their own business problems and want to minimize the cost of using our platform in toil, time, and treasure.

### Conflicting Goals

TODO: continue the RFC...

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
