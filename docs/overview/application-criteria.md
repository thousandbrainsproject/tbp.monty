---
title: Application Criteria
description: What can you use Monty for?
---
# Requirements
**Monty is designed for sensorimotor applications.** It is **not designed to learn from static datasets** like many current AI systems are, and so it may not be a drop-in replacement for an existing AI use case you have. Our system is made to learn and interact using similar principles to the human brain, the most intelligent and flexible system known to exist. The brain is a sensorimotor system that constantly receives movement input and produces movement outputs. In fact, the only outputs from the brain are for movement. Our system works the same way. It **needs to receive information about how the sensors connected to it are moving in space** in order to learn structured models of whatever it is sensing. The inputs are not just a bag of features, but instead features at poses (location + orientation). Have a look at our [challenging preconceptions page](./vision-of-the-thousand-brains-project/challenging-preconceptions.md) for more details.

# Potential Applications

Any application where you have **moving sensors** is a potential application for Monty. This could be physical movement of sensors on a robot. It could also be simulated movement such as our [simulations in Habitat](../how-monty-works/environment-agent.md) or the sensor patch cropped out of a larger 2D image in the [Monty meets world](benchmark-experiments#monty-meets-world) experiments. It could also be movement through conceptual space or another non-physical space such as navigating the internet.

Applications where we anticipate Monty to particularly shine are:
- Applications where **little data** to learn from is available
- Applications where **little compute to train** is available
- Applications where **little compute to do inference** is available (like on edge devices)
- Applications where **no supervised data** is available
- Applications where **continual learning and fast adaptation** is required
- Applications where the agent needs to **generalize/extrapolate to new tasks**
- Applications where **interpretability** is important
- Applications where **robustness** is important (to noise but also samples outside of the training distribution)
- Applications where **multimodal integration** is required or **multimodal transfer** (learning with one modality and inferring with another)
- Applications where the agent needs to **solve a wide range of tasks**
- Applications **where humans do well** but current AI does not

## Some Example Applications
To get an idea of how Monty could be used in the future, you can have a look at this video where we go over how key milestones on our research roadmap will unlock more and more applications. Of course, there is no way to anticipate the future and there will likely be many applications we are not thinking of today. But this video might give you a general idea of the types of applications a Thousand Brains System will excel in. 
[block:embed]
{
  "html": "<iframe class=\"embedly-embed\" src=\"//cdn.embedly.com/widgets/media.html?src=https%3A%2F%2Fwww.youtube.com%2Fembed%2FIap_sq1_BzE%3Ffeature%3Doembed&display_name=YouTube&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DIap_sq1_BzE&image=https%3A%2F%2Fi.ytimg.com%2Fvi%2FIap_sq1_BzE%2Fhqdefault.jpg&type=text%2Fhtml&schema=youtube\" width=\"854\" height=\"480\" scrolling=\"no\" title=\"YouTube embed\" frameborder=\"0\" allow=\"autoplay; fullscreen; encrypted-media; picture-in-picture;\" allowfullscreen=\"true\"></iframe>",
  "url": "https://www.youtube.com/watch?v=Iap_sq1_BzE",
  "title": "2025/04 - TBP Future Applications & Positive Impacts",
  "favicon": "https://www.youtube.com/favicon.ico",
  "image": "https://i.ytimg.com/vi/Iap_sq1_BzE/hqdefault.jpg",
  "provider": "https://www.youtube.com/",
  "href": "https://www.youtube.com/watch?v=Iap_sq1_BzE",
  "typeOfEmbed": "youtube"
}
[/block]

# Capabilities

For a list of current and future capabilities, see [Capabilities of the System](./vision-of-the-thousand-brains-project/capabilities-of-the-system.md). For experiments (and their results) measuring the current capabilities of the Monty implementation, see our [Benchmark Experiments](benchmark-experiments.md).
