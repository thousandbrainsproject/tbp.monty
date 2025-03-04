- Start Date: 2025-01-28
- RFC PR: https://github.com/thousandbrainsproject/tbp.monty/pull/159/

# Summary

The proposal is to establish a better way of organizing Thousand Brains Project repositories.

Currently there are two repositories:

1. `tbp.monty` - This is the main repository for the Monty framework.
2. `monty_lab` - This is what `nupic.monty/projects` used to be, and is currently a catch-all for code that is not part of the main Monty framework.

The high-level proposal is to discontinue the use of `monty_lab` as a location for new projects, and instead have a very low threshold for creating new, independent repositories for these.

More specifically, the proposed structure and description of each repository would be:

`tbp.monty`
- Core code-base for the Monty framework.
- All code has undergone thorough review, and Continuous Integration runs unit-tests and style checks on any new code.
- Should not become bloated with code that is unlikely to be re-used in future work, or is of sufficient complexity that it can be better understood as a separate package.
- Should not contain configs for various experiments; the only configs in tbp.monty are those used for our benchmark experiments.

`monty_lab`
- This will still be present to provide access to older projects like `monty_meets_world`. Further additions to this repository however are discouraged, unless they are e.g. to add fixes to code.

New repositories can then be created as needed for future projects that do not belong in `tbp.monty`. Two typical examples are below, although a repository need not be a package or contain code for a paper. If you are creating a Python package or library, please see the template repositories we have created for this (https://github.com/thousandbrainsproject/tbp.python_package and https://github.com/thousandbrainsproject/tbp.python_library).

`tbp.name_of_package`
- E.g. `tbp.floppy`
- A package that is a collection of modules that are intended to be re-used in other projects. This does not imply that it is a package that will be maintained long-term for a broader community, but is rather something that we see ourselves re-using at the TBP.
- Should be well-documented, and have some unit-test coverage and style checks. PR reviews should be of a similar standard to `tbp.monty`.
- The code should be of high enough quality that it can be used with confidence with the current `tbp.monty` codebase at the time of creation. However, there is not an expectation that it will be usable as-is if `tbp.monty` undergoes MAJOR changes (per [semver](https://semver.org/)). This is to reduce the burden on contributors of such package repositories, and given the unknowability of how often a package will be used in the future.
- Open to contributions from the community.

`tbp.name_of_paper`
- E.g. `tbp.tbs_for_rapid_robust_learning_and_inference`
- A repository that contains the code required to replicate results and figures from a paper. There should be a single repository for each paper.
- A given paper repository can be broken up using sub-folders if it makes use of highly distinct frameworks for different experiments, such as Pytorch vs. Monty.
    - For example, the `tbp.tbs_for_rapid_robust_learning_and_inference` repository would contain two sub-folders, `monty` and `pytorch`, corresponding to the different frameworks used for different parts of the paper. Note that these folders can be associated with different conda environments and .gitignore files as desired.
    - The version number associated with these frameworks should be specified, such that the requirements for replicating results are clear.
- PR reviews should be of a similar standard to `tbp.monty`. Code (typically configs but also analysis code) should be of a high standard given that it forms the basis of published work.

More concretely, the structure that would be created given our current codebase and work on the "Demonstrating Monty Capabilities" paper (actual title: Thousand Brains Systems for Rapid, Robust Learning and Inference) is as follows:

```
tbp.monty/
tbp.floppy/
tbp.tbs_for_rapid_robust_learning_and_inference/
```

### Other Guidance
- When opening a PR, if there is any doubt about the best destination for the code, it is best to discuss this with the team.
- We should have a low threshold for creating new repositories where appropriate.
- We should prepend all our repositories with `tbp.` at creation to indicate that they are part of the Thousand Brains Project (depending on where someone has cloned the repository to, this may not always be obvious on their local machine), and to begin carving out a namespace for the TBP in settings like PyPi.
- It is encouraged to add a description and tags to repositories such that the [overview page of the Thousand Brains Project GitHub organization](https://github.com/orgs/thousandbrainsproject/repositories) provides helpful context.
- If a repository relates to an old project that is no longer relevant, it can optionally be "archived" using GitHub's archive feature. This can help reduce the number of active repositories, and note that a repository can be unarchived at any time. It is no longer necessary to move this code to `monty_lab`. Finally, note that archived repositories keep the `tbp.` prefix.
- We will create an overview page using [GitHub's profile README feature](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/customizing-your-profile/managing-your-profile-readme) to provide helpful context on the key repositories that are part of the Thousand Brains Project. This should be updated from time to time to reflect the most important repositories for newcomers to the TBP.

# Motivation

As the Thousand Brains Project grows, it will be important to have a better way of organizing the codebase. This RFC proposes we establish a sustainable structure that ensures code remains high quality and accessible, while minimizing overhead for contributors.

# Open Questions
- None

# Future Possibilities

This proposal also relates to a potential RFC on the creation of a "repository template" that can be reused for new repositories. We might structure these templates based on typical types of repositories (e.g. those that are standalone packages, vs. those that contain the configs and analysis code for a paper).