---
title: Ways to Contribute to Code
---
> [!NOTE]
>
> For an architecture overview see the [Architecture Overview](../overview/architecture-overview.md) page. Each of the major components in the architecture can be customized.

There are many ways in which you can contribute to the code. The list below is not comprehensive but might give you some ideas.

- **Create a Custom Sensor Module**: Sensor Modules are the interface between the real-world sensors and Monty. If you have a specific sensor that you would like Monty to support, consider contributing a Sensor Module for it. Also, if you have a good idea of how to extract useful features from a raw stream of data, this can be integrated as a new Sensor Module. For information on how to do this see our [Sensor Module Contributing Guide](ways-to-contribute-to-code/contributing-sensor-modules.md)
- **Create a Custom Learning Module**: Learning Modules are the heart of Monty. They are the repeating modeling units that can learn from a stream of sensorimotor data and use their internal models to recognize objects and suggest actions. What exactly happens inside a Learning Module is not prescribed by Monty. We have some suggestions, but you may have a lot of other ideas. As long as a Learning Module adheres to the Cortical Message Protocol and implements the abstract functions defined [here](../../src/tbp/monty/frameworks/models/abstract_monty_classes.py), it can be used in Monty. It would be great to see many ideas for Learning Modules in this code base that we can test and compare. For information on how to implement a custom Learning Module, see our [Learning Module Contributing Guide](ways-to-contribute-to-code/contributing-learning-modules.md)
- **Write a Custom Motor Policy**: Monty is a sensorimotor system, which means that action selection and execution are important aspects. Model-based action policies are implemented within the Learning Module's [Goal State Generator](../../src/tbp/monty/frameworks/models/goal_state_generation.py), but model-free ones, as well as the execution of the suggested actions from the Learning Modules, are implemented in the motor system. Our Thousand Brains Project team doesn't have much in-house robotics experience so we value contributions from people who do. For instructions on how to implement custom motor systems, see our [Motor System Contributing Guide](ways-to-contribute-to-code/contributing-motor-systems.md)
- **Improve the Code Infrastructure**: Making the code easier to read and understand is a high priority for us, and we are grateful for your help. If you have ideas on how to refactor or document the code to improve this, consider contributing. Please [create an RFC](./request-for-comments-rfc.md) before working on any major code refactor.
- **Optimize the Code**: We are always looking for ways to run our algorithms faster and more efficiently, and we appreciate your ideas on that. Just like the previous point, PRs around this should not change anything in the outputs of the system.
- **Work on an open Issue**: If you came to our project and want to contribute code but are unsure of what, the [open Issues](https://github.com/thousandbrainsproject/tbp.monty/issues) are a good place to start.  See our guide on [how to identify an issue to work on](ways-to-contribute-to-code/identify-an-issue-to-work-on.md) for more information.

# How to Contribute Code

Monty integrates code changes using Github Pull Requests. To start contributing code to Monty, please consult the [Contributing Pull Requests](pull-requests.md) guide.