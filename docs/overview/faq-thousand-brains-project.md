---
title: FAQ - Thousand Brains Project
description: Frequently asked question about the project.
---
# What Kind of Applications Will This Project Tackle?

We aim to build a **general-purpose system** that is not optimized for one specific application. Think of it as similar to Artificial Neural Networks (ANNs), which can be applied to all kinds of different problems and are a general tool for modeling data. Our system will be aimed at **modeling sensorimotor data, not static datasets**. This means the input to the system should contain sensor and motor information. The system then models this data and can output motor commands. 

The most natural application is **robotics**, with physical sensors and actuators. However, the system should also be able to generalize to **more abstract sensorimotor setups, such as navigating the web or conceptual space**. As another example, **reading and producing language** can be framed as a sensorimotor task where the sensor moves through the sentence space, and action outputs could produce the letters of the alphabet in a meaningful sequence. Due to the messaging protocol between the sensor and learning module, the system can **effortlessly integrate multiple modalities** and even **ground language in physical models learned through other senses like vision or touch**.

For more details, see [Application Criteria](application-criteria.md) and [Capabilities of the System](./vision-of-the-thousand-brains-project/capabilities-of-the-system.md).

# What is Special about This Project?

Much of today's AI is based on learning from giant datasets by training ANNs on vast clusters of GPUs. This not only burns a lot of energy and requires a large dataset, but it is also fundamentally different from how humans learn. We know that there is a more efficient way of learning; we do it every day with our brain, which uses about as much energy as a light bulb. So why not use what we know about the brain to make AI more efficient and robust?

This project is an ambitious endeavor to **rethink AI from the ground up**. We know there is a lot of hype around LLMs, and we believe that they will remain useful tools in the future, but they are not as efficient nor as capable as the neocortex. In the Thousand Brains Project, we want to build an open-source platform that will catalyze a new type of AI. **This AI learns continuously and efficiently through active interaction with the world**, just like children do. 

# How Will You Keep Track of Progress?

We aim to make our conceptual progress available quickly by **publishing recordings of all our research meetings on YouTube**. Any engineering progress will automatically be available as part of our **open-source code base.** We keep track of our system's capabilities by frequently running a suite of **benchmark experiments** and evaluating the effectiveness of any new features we introduce. The results of these will also be visible in our GitHub repository.

In addition to making all incremental progress visible, we will publish more succinct write-ups of our progress and results at **academic conferences and in peer-reviewed journals**. We also plan to produce more condensed informational content through a **podcast and YouTube videos**.

# Do You Think LLMs Will Become Obsolete?

**No**, LLMs are incredibly useful and powerful for various applications. However, we believe that the current approach most researchers and companies employ of incrementally adding small features to ANNs/LLMs will lead to diminishing returns over the next few years. **Developing genuine human-like intelligence demands a bold, innovative departure from the norm.** As we rethink AI from the ground up, we anticipate a longer period of initial investment with little return that will eventually compound to unlock potential that is unreachable with today’s solutions. We believe that this more human-like artificial intelligence will be what people think of when asked about AI in the future. At the same time, **we see LLMs as a tool that will continue to be useful for specific problems, much like the calculator is today**. 

# What are the Advantages of This System (over ANNs/LLMs)?

The system we are building in the Thousand Brains Project has many advantages over current popular approaches. For one, it is much **more energy and data-efficient.** It can **learn faster and from less data** than deep learning approaches. This means that it can **learn from higher-quality data and be deployed in applications where data is scarce**. It can also **continually add new knowledge** to its models without forgetting old knowledge. The system is always learning, actively testing hypotheses, and improving its current models of whatever environment it is learning in.

Another advantage is the system's **scalability and modularity**. Due to the modular and general structure of the learning and sensor modules, one can build **extremely flexible architectures tailored to an application's specific needs**. A small application may only require a single learning module to model it, while a large and complex application could use thousands of learning modules and even stack them hierarchically. The cortical messaging protocol makes **multimodal integration possible and effortless**.

Using reference frames for modeling allows for **easier generalization, more robust representations, and more interpretability**. To sum it up, the system is **good at all the things that humans are good at**, but current AI is not.

# How is the Thousand Brains Project (TBP) Different from Hierarchical Temporal Memory (HTM)?

The TBP and HTM are both based on years of neuroscience research at Numenta and other labs across the world. They both implement principles we learned from the neocortex in code to build intelligent machines. However, they are **entirely separate implementations and differ in which principles they focus on**. While **HTM focuses more on the lower-level computational principles** such as sparse distributed representations (SDR), biologically plausible learning rules, and sequence memory, the **TBP focuses more on the higher-level principles** such as sensorimotor learning, the cortical column as a general and repeatable modeling unit, and models structured by reference frames. 

In the TBP, we are building a general framework based on the principles of the thousand brains theory. We have sensor modules that convert raw sensor data into a common messaging protocol and learning modules, which are general, sensorimotor modeling units that can get input from any sensor module or learning module. Importantly, **all communication within the system involves movement information, and models learned within the LMs incorporate this motion information into their reference frames.** 

There can be many types of learning modules as long as they adhere to the messaging protocol and can model sensorimotor data. This means **there could be a learning module that uses HTM** (with some mechanism to handle the movement data, such as grid cells). However, the learning modules **do not need to use HTM**, and we usually don't use HTM-based modules in our current implementation.

# How can I Help? How can I Join the Project?

We are excited that you’re interested in this project, and we want to build an active open-source community around it. There are different ways you can get involved. If you are an engineer or researcher with ideas on improving our implementation, we would be delighted to have you **contribute to our code base or documentation.** Check out details on [ways to contribute](../contributing/ways-to-contribute-to-code.md) here. 

Second, if you have a specific sensorimotor task you are trying to solve, we would love for you to **try our approach**. We will work on making an easy-to-use SDK so you can just plug in your sensors and actuators, and our system does the modeling for you. If you would like to see some examples of how other people used our code in their projects, check out our [project showcase](../community/project-showcase.md). 

We will start hosting regular **research and industry workshops** and would be happy to have you join. 

Follow our [meetup group](https://www.meetup.com/thousand-brains-project/) for updates on upcoming events.

We are also planning to host a series of **invited speakers** again, so please let us know if you have research that you would like to present and discuss with us. Also, if you have ideas for potential **collaborations**, feel free to reach out to us at [info@thousandbrains.org](mailto:info@thousandbrains.org).

# How is the Thousand Brains Project Funded?

The Thousand Brains project and our research **continues to be funded by Jeff Hawkins** and now **also in part by the Gates Foundation**.  Our funding is focused on fundamental research into this new technology but will also facilitate exchanges with related research groups and potential applications.

# What are Numenta’s Goals for the Thousand Brains Project?

From our inception, Numenta has had two goals: first, to **understand the human brain and how it creates intelligence**, and second, to **apply these principles to enable true machine intelligence**. The Thousand Brains Project is aligned with both of these goals and adds a third goal to make the technology accessible and widely adopted.

Numenta has developed a framework for understanding what the neocortex does and how it does it, called the Thousand Brains Theory of Intelligence. It is a collaborative open-source framework dedicated to creating a new type of artificial intelligence that pushes the current boundaries of AI. Numenta’s goals for the Thousand Brains Project are to **build an open-source platform for intelligent sensorimotor systems and to be a catalyst for a whole new way of thinking about machine intelligence**.