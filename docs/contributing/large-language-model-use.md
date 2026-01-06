---
title: Large Language Model Use
---

We're really excited to receive contributions from the wider community - one of the highlights of our day is when we see a new PR or RFC opened by a non-Maintainer.

The reason for the guidance below is that we are increasingly encountering the limits of LLMs, and how these can negatively affect our time spent on reviews and discussions.

As such, the following outlines several concerning attributes that are associated with LLM-generated contributions, and why these can be problematic. While we do not institute any strict guidance, please note that if we feel that the use of AI/LLMs has resulted in a contribution that matches these concerning patterns, Maintainers reserve the right to not review your contribution. We do not want to discourage contributions, but this balancing act on our part is an unfortunate side-effect of the small size of the Maintainers team vs. the speed at which LLM content can be generated.


# Guidance on the Use of AI/LLMs

## Writing RFCs

We suggest that all RFCs are written by you, and that you try to avoid using Large Language Models (LLMs) like ChatGPT in the writing process. Some of our motivations for this are:
- We want to get a clear understanding of your solution, and we have found that LLMs give poor, if approximately correct, RFC proposals. The approximate correctness makes it more difficult to spot logical inconsistencies, and the additional verbosity makes them time-consuming to review. Logical inconsistencies are particularly common because the work of the TBP falls very much in the "out of training distribution" domain.
- When we review RFCs written by LLMs, it often ends up taking more time both for us, and for you, so it is much better if you write them yourself.
- Writing the RFC yourself will serve as a useful starting point if you intend to implement the idea afterwards. Formulating the proposal in your own words often surfaces issues and ideas you didn't think about previously, which is not something that usually happens when you have an LLM generate the proposal for you. Note that RFCs don't need to be long-winded arguments and detailed elaborations of every step you intend to take. They are meant to be a concise summary of your idea and form the basis for discussion and feedback from others. As such, it brings you no benefits, both in your time spent and understanding gained, if they are LLM-generated.


Note that it is perfectly acceptable to discuss ideas with an LLM when formulating your own thoughts. For example, there may be a concept in neuroscience or computer vision that you would like to understand better, and LLM conversations can be a useful starting point. However, we recommend that you always be wary of hallucinations in LLM outputs; due to their frequent occurrence, LLMs are better used as an initial - but not final - provider of information on a topic.

## Contributing Code to `tbp.monty`

In principle, you may use LLMs, such as code-assistants, when writing code that you contribute to the Thousand Brains Project. However, we suggest that you do so in a limited manner, being mindful of the below issues:

### Quality Concerns
- As with RFCs, we have found that our code-base is out-of-distribution, and the quality of code written by LLMs, while superficially correct, often is not.
- Similarly, Monty is fundamentally about developing sensorimotor AI that can succeed in embodied settings. These are precisely the settings that LLMs struggle to perform in, and so once again we have found they frequently provide incorrect solutions.
- LLM code is often verbose, or results in excessive, unnecessary changes to the code.

Due to the above reasons, LLM-generated code can take a great deal of time to review and debug. This can be avoided when PRs are written with intent by a person.

### Legal Concerns
- There are non-trivial legal concerns around contamination of code when using LLMs. See for example, [this recent study](https://arxiv.org/html/2408.02487v1), which demonstrates that LLM-generated code can violate copyright restrictions in a significant number of cases.
- A cornerstone of the Thousand Brains Project is its open-source nature, and this motivates our use of an MIT licence when we distribute our code. However, the inadvertent integration of copyright-protected code into `thousandbrainsproject/tbp.monty` could jeopardize the ability to make the code open-source, disrupting any distributions of the platform.

### Take-Aways

The high-level guidance based on the above is:
- Using an LLM to auto-complete variable names and other simple speed-ups is appropriate.
- Multi-line sections of algorithmic code written by LLMs should be checked for logical correctness and potential copyright violations before opening a PR into `thousandbrainsproject/tbp.monty`.

Below we provide further guidance on some edge cases.

#### Work on Research Prototypes
- [Research Prototypes](https://github.com/thousandbrainsproject/tbp.monty/blob/2ed5607ec45bb23bba449373b36628030cf4d1b4/rfcs/0000_code_guidance_for_researchers_and_community.md) are separate forks of `thousandbrainsproject/tbp.monty` intended to rapidly evaluate the merits of a particular research idea. As they are not part of the core Monty platform, the legal concerns described above are less relevant, however the code-quality issues remain.
- If you do end up integrating significant portions of LLM code into a Research Prototype PR, we ask that you clearly label this code as such. That way, if the RP is deemed significant enough to integrate into `thousandbrainsproject/tbp.monty`, any legal issues can be addressed at this time. However, this may delay the [Implementation Project process](https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0014_conducting_research_while_building_a_stable_platform.md#implementation-project), and so it is again suggested that you minimize using significant portions of code written by LLMs.
- To document that a portion of code was generated by an LLM, you can use the [multi-line commenting feature in GitHub](https://graphite.dev/guides/comment-multiple-lines-code-github), as shown in the example below:


<img src="../figures/contributing/llm_commenting_example.png" alt="Example of Highlighting LLM Code" width="800">



#### Agentic LLMs 
- Due to the above, we suggest that you do not use agentic workflows that write large amounts of code in an automated way, unless as a means of automating a simple task.
- An example of a good use of an agentic LLM setup would be widespread changes required to reflect an update in terminology. For example, in a [recent PR](https://github.com/thousandbrainsproject/tbp.tbs_sensorimotor_intelligence/pull/55/files), the order of two figures in our paper was swapped, requiring many small changes to the code and documentation. This was rapidly automated with LLM assistance. We then verified the correctness of the implementation after these changes.
- On the other hand, we suggest that you do not pass a [Future Work item](https://thousandbrainsproject.readme.io/docs/project-roadmap) description into an LLM, and then open a PR with all of the code it generated.

## Contributing on the Forums

- [The TBP Discourse forums](https://thousandbrains.discourse.group/) are an excellent setting to discuss nascent ideas.
- In these discussions, we would love to engage with you and your ideas. As such, we ask that you not post the outputs of LLMs as if these were your own ideas.
- If you would like to quote or refer to a conversation you have had with an LLM, please just indicate this clearly.
- See the advice under [Writing RFCs](#writing-rfcs) for suggestions about when LLMs can be useful "tutors" when exploring a topic.

## Productivity Tools
- Please note that the above guidance does not apply to productivity tools such as grammar checkers, which we understand you might find helpful in copy-editing text in RFCs and forum posts. If you need to use a more advanced AI tool for a specific reason (speech-to-text, etc.), we simply ask that you keep the concerns of this document in mind.