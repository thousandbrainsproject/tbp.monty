- Start Date: 2025-01-28
- RFC PR: https://github.com/thousandbrainsproject/tbp.monty/pull/159/

# Summary

The proposal is to establish a better way of organizing Thousand Brains Project repositories.

Currently there are two repositories:

1. `tbp.monty` - This is the main repository for the Monty framework.
2. `monty_lab` - This is what `nupic.monty/projects` used to be, and is currently a catch-all for code that is not part of the main Monty framework.

The high-level proposal is to break-up `monty_lab` where appropriate, and have a lower threshold for creating new, independent repositories.

More specifically, the proposed structure and description of each repository would be:

`tbp.monty`
- Core code-base for the Monty framework.
- All code has undergone thorough review, and Continuous Integration runs unit-tests and style checks on any new code.
- Should not become bloated with code that is unlikely to be re-used in future work, or is of sufficient complexity that it can be better understood as a separate package.
- Should not contain configs for various experiments; the only configs in tbp.monty are those used for our benchmark experiments.

`monty_lab`
- A repository to house discontinued or paused research projects like  `high_dim_coincidence_detection` and `grid_cells`.
- Local forks of `monty_lab` can be used by contributors to track in-progress work. However, this does not mean that such code, once mature enough to be shared, should necessarily be merged into the main `monty_lab` repository.
- In general, if the code is likely to be re-used in the future, or forms part of a paper, it should *not* go in `monty_lab` (more on this below).
- Continuous Integration does not include style or unit-test checks on new code. 
- When merging a discontinued project, PR review does not need to be as thorough as for code pushed to the main `tbp.monty` repository.
- Not intended as a code-base that community members can contribute to. As such, issues like bugs are less likely to be noticed and fixed. 

Additional repositories can then be created as needed. Two typical examples would be:

`tbp.name_of_package`
- E.g. `tbp.floppy`
- A package that is a collection of modules that are intended to be re-used in other projects. This does not imply that it is a package that will be maintained long-term for a broader community, but is rather something that we see ourselves re-using at the TBP.
- Should be well-documented, and have some unit-test coverage and style checks. PR reviews should be of a similar standard to `tbp.monty`.
- The code should be of high enough quality that it can be used with confidence with the current `tbp.monty` codebase at the time of creation. However, there is not an expectation that it will be usable as-is if `tbp.monty` undergoes MAJOR changes (per [semver](https://semver.org/)). This is to reduce the burden on contributors of such package repositories, and given the unknowability of how often a package will be used in the future.
- Open to contributions from the community.

`tbp.name_of_paper`
- E.g. `tbp.tbs_for_rapid_robust_learning_and_inference`
- A repository that contains the code required to replicate results and figures from a paper.
- A given paper can be broken up into multiple repositories if it makes use of highly distinct frameworks for different parts of the paper, such as Pytorch vs. Monty.
- PR reviews should be of a similar standard to `tbp.monty`. Code (typically configs but also analysis code) should be of a high standard given that it forms the basis of published work.

More concretely, the structure that would be created given our current codebase and work on the "Demonstrating Monty Capabilities" paper (actual title: Thousand Brains Systems for Rapid, Robust Learning and Inference) is as follows:

```
tbp.monty/
monty_lab/
tbp.floppy/
tbp.tbs_for_rapid_robust_learning_and_inference_monty/  # Most of the code for the paper, including for generating the figures
tbp.tbs_for_rapid_robust_learning_and_inference_pytorch/  # Code to generate the results for the Pytorch models used in the paper
```

### Other Guidance
- When opening a PR, if there is any doubt about the best destination for the code, it is best to discuss this with the team.
- We should have a low threshold for creating new repositories where appropriate.
- It is encouraged to add a description and tags to repositories such that the [overview page of the Thousand Brains Project GitHub organization](https://github.com/orgs/thousandbrainsproject/repositories) provides helpful context.

# Motivation

As the Thousand Brains Project grows, it will be important to have a better way of organizing the codebase. This RFC proposes we establish a sustainable structure that ensures code remains high quality and accessible, while minimizing overhead for contributors.

# Open Questions

- If a given paper is separated into multiple repositories, will this create unecessary challenges for someone who wants to replicate the results (e.g. generating a figure)? We can think about how to structure results folders to make this as easy as possible.

# Future Possibilities

This proposal also relates to a potential RFC on the creation of a "repository template" that can be reused for new repositories. We might structure these templates based on typical types of repositories (e.g. those that are standalone packages, vs. those that contain the configs and analysis code for a paper).

# Final Comments

This is an RFC, so I'm very open to alternative suggestions for how we can organize the codebase, or any concerns that this proposal raises.