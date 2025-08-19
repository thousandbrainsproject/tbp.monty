- Start Date: 2025-08-19
- RFC PR: (leave this empty, it will be filled in after RFC is merged)

# Motivation

LLM code-assistants have the potential to improve the productivity of coders. More generally, there is optimism that AI tools might accelerate science. At the same time, their use can introduce non-intuitive drawbacks. The aim of this RFC is to describe these in more detail, and agree on what guidance we should ask of both ourselves and the community.


# Guidance on the Use of AI/LLMs

## Writing RFCs

We ask that all RFCs are written by you, and that you refrain from using Large Language Models (LLMs) like ChatGPT in this process. Some of our motivations for this are:
- We want to get a clear understanding of your solution, and we have found that LLMs give poor, if approximately correct, RFC proposals. This is particularly the case because the work of the TBP falls very much in the "out of training distribution" domain.
- When we review RFCs written by LLMs, it often ends up taking more time both for us, and for you, so it is much better if you write them yourself.

## Contributing Code to `tbp.monty`

In principle, you may use LLMs, such as code-assistants, when writing code that you contribute to the Thousand Brains Project. However, we ask that you do so in a limited manner, being mindful of the below issues:

### Quality Concerns
- As with RFCs, we have found that our code-base is out-of-distribution, and the quality of code written by LLMs, while superficially correct, often is not.
- Similarly, Monty is fundamentally about developing sensorimotor AI that can succeed in embodied settings. These are precisely the settings that LLMs struggle to perform in, and so once again we have found they frequently provide incorrect, logically inconsistent solutions.
- LLM code is often verbose, or results in excessive, unnecessary changes to the code.

Due to the above reasons, LLM-generated code can take a great deal of time to review and debug. This can be avoided when PRs are written with intent by a person.

### Legal Concerns
- There are non-trivial legal concerns around contamination of code when using LLMs. See for example, [this recent study](https://arxiv.org/html/2408.02487v1), which demonstrates that LLM-generated code can violate copyright restrictions in a significant number of cases.
- A cornerstone of the Thousand Brains Project is its open source nature, and this motivates our use of an MIT licence when we distribute our code. However, the inadvertent integration of copyright-protected code into `thousandbrainsproject/tbp.monty` could jeopardize the ability to make the code open-source, disrupting any distributions of the platform.

### Take-Aways

The high-level guidance based on the above is:
- Using an LLM to auto-complete variable names and other simple speed-ups can be appropriate.
- Multi-line sections of algorithmic code written by LLMs should be thoroughly checked for logical correctness and potential copyright violations before opening a PR into `thousandbrainsproject/tbp.monty`.

Below we provide further guidance on some edge cases.

#### Work on Research Prototypes
- [Research Prototypes](https://github.com/nielsleadholm/tbp.monty/blob/978b15653a4c08bb21e28752a2ea9e01a3da906b/rfcs/0000_code_guidance_for_researchers_and_community.md) are separate forks of `thousandbrainsproject/tbp.monty` intended to rapidly evaluate the merits of a particular research idea. As they are not part of the core Monty platform, the legal concerns described above are less relevant, however the code-quality issues remain.
- If you do end up integrating significant portions of LLM code into a Research Prototype PR, please ensure you clearly label this code as such. That way, if the RP is deemed significant enough to integrate into `thousandbrainsproject/tbp.monty`, any legal issues can be addressed at this time. However, this may delay the Implementation Project process, and so it is again advised that you minimize using significant portions of code written by LLMs.

#### Agentic LLMs 
- The issues highlighted mean that we ask that you do *not* use agentic workflows that write large amounts of code in an automated way, unless as a means of automating a simple task.
- An example of a reasonable use of an agentic LLM setup would be widespread changes required to reflect an update in terminology. For example, in a [recent PR](https://github.com/thousandbrainsproject/tbp.tbs_sensorimotor_intelligence/pull/55/files), the order of two figures in our paper was swapped, requiring many small changes to the code and documentation. This was rapidly automated with LLM assistance. We then verified the correctness of the implementation after these changes.
- On the other hand, please do not pass a Future Work item description into an LLM, and then open a PR with all of the code it generated. These kinds of contributions slow down, rather than accelerate, our shared mission at the Thousand Brains Project.

## Contributing on the Forums

- [The TBP Discourse forums](https://thousandbrains.discourse.group/) are an excellent setting to discuss nascent ideas.
- In these discussions, we would love to engage with you and your ideas. As such, we ask that you not post the outputs of LLMs as if these were your own ideas.
- If you would like to quote or refer to a conversation you have had with an LLM, please just indicate this clearly.

## Assistive Technology
- Please note that the above guidance does not apply to assistive technology such as grammar checkers, which we understand you might find helpful in copy-editing text in RFCs and forum posts. If you need to use a more advanced AI tool for a specific reason, we ask that you reach out to us directly to discuss this beforehand.