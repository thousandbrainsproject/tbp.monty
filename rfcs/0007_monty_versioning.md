- Start Date: 2025-02-20
- RFC PR: https://github.com/thousandbrainsproject/tbp.monty/pull/186

# Summary

Begin versioning Monty using [semantic versioning](https://semver.org/).

# Motivation

We want papers published by the Thousand Brains Project to be reproducible. The likely approach the project will adopt is to create a repository for each paper and include the code to reproduce the paper in the repository. When the paper relies on `tbp.monty`, it should be clear which version of `tbp.monty` is needed to reproduce the paper. Ideally, the paper repository declares a specific `tbp.monty` version as a dependency that is installed like any other dependency.

# Explanation

Monty uses [semantic versioning](https://semver.org/) for versions.

The intent behind semantic versioning is to provide a way to rapidly and tersely communicate to people and machines the compatibility of a software change.

In summary, the version number is `MAJOR.MINOR.PATCH`.

* `MAJOR` is incremented when backwards-incompatible (breaking) changes are made.
* `MINOR` is incremented when new functionality is added in a backwards-compatible manner.
* `PATCH` is incremented when backwards-compatible bug fixes are made.
* CI, tools, infrastructure, documentation, or RFC changes do not increment the version number.

It is worth noting that the `MAJOR` version number `0` is special. When `MAJOR` is `0`, then incrementing the `MINOR` version number ***can include backwards-incompatible changes***. `MAJOR` version number `0` is the usual way of indicating that the software is still in development and not ready for production use.

We will use `MAJOR` version number `0` for the ongoing development of Monty. Incrementing the `MAJOR` version number beyond `0` should be a future RFC, as it will be a significant event.

The current version can be found in [`src/tbp/monty/__init__.py`](https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/__init__.py) as the `__version__` variable.

## Updating the version

### Proposed constraints

Here are some constraints related to updating the version:

* We want the current version to be in the source code.
* We want the current version to be programmatically accessible, e.g., by the `tools.print_version.cli` tool.
* The `main` branch is protected and changes can be pushed only via Pull Requests.

> [!NOTE]
> "Current version" can have a counterintuitive meaning due to different interpretations of what "current" means. For the purposes of this RFC, "current" is (somewhat circularly) defined as the version number in the source code.

### Process

1. Create a Pull Request that ***only*** updates the `__version__` variable in [`src/tbp/monty/__init__.py`](https://github.com/thousandbrainsproject/tbp.monty/blob/main/src/tbp/monty/__init__.py).

2. Get the Pull Request approved and merged in the usual way.

One of the benefits of using a Pull Request is that it offers a place to discuss what the new version number should be. Sometimes, determining whether something is a breaking change is non-obvious.

In the future, release workflows could be triggered by the version number change.

### Expected Workflow

By adopting this versioning scheme and process, the expected workflow is as follows:


- commit
- commit
- commit
- commit
- review all commits above and update the version accordingly (in the future this will also publish Monty)
- commit
- commit
- review all commits above and update the version accordingly (in the future this will also publish Monty)
- commit
- ...


### Rejected Alternatives

Automatic version updates based on the commit message are deferred to future work.

Using GitHub Action to increment the version number was prototyped and worked as expected. However, due to the constraints that Pull Requests are required, it turned out to be no more convenient than manually updating the version number via a normal Pull Request. The GitHub Action updated the version, created a Pull Request, and automatically merged it without approval. A normal Pull Request leaves the same commit and pull request audit trail, but requires Maintainer approval. Maintainer approval for updating the version number seems like a useful sanity check. In the future, when releases are triggered by version change, this approval will be helpful.

```yaml
name: Version

permissions:
  contents: write
  pull-requests: write

on:
  workflow_dispatch:
    inputs:
      email:
        description: "Email to use for the commit"
        required: true
        type: string
      name:
        description: "Name to use for the commit"
        required: true
        type: string
      version:
        description: "Version to set"
        required: true
        type: string

jobs:
  version:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          lfs: true
      - id: branch
        run: |
          echo "name=version-${{ inputs.version }}-$(date -u +"%Y%m%dT%H%M%SZ")" >> $GITHUB_OUTPUT
      - run: |
          git checkout -b ${{ steps.branch.outputs.name }}
      - run: |
          sed -i "s/__version__.*/__version__ = \"${{ inputs.version }}\"/" src/prototype/project/__init__.py
      - run: |
          git add src/prototype/project/__init__.py
      - run: |
          git config --global user.email "${{ inputs.email }}"
      - run: |
          git config --global user.name "${{ inputs.name }}"
      - run: |
          git commit -m "ci: version ${{ inputs.version }}"
      - run: |
          git push origin ${{ steps.branch.outputs.name }}
      - run: |
          gh pr create --base main --head ${{ steps.branch.outputs.name }} --title "ci: version ${{ inputs.version }}" --body "Version ${{ inputs.version }} set by @${{ github.event.sender.login }}"
        env:
          GH_TOKEN: ${{ github.token }}
      - run: |
          gh pr merge --admin --squash --delete-branch
        env:
          GH_TOKEN: ${{ github.token }}
      - if: ${{ failure() }}
        run: |
          git push origin --delete ${{ steps.branch.outputs.name }}
```

Release branches are deferred to possible future work. The way release branches would work is that we would never update `__version__` on the main branch. Instead, every time we want to make a new version, we would create a new branch, update `__version__`, and merge it into the release branch. This has some nice properties for automation, however, it interferes with some of the proposed constraints above.

## Accessing specific versions

The most detailed and specific way to access a specific version of `tbp.monty` is to clone the repository and checkout the specific commit.

If we tag the version commit with the corresponding version number, then the version can be accessed by checking out the tag. GitHub has [releases](https://github.com/thousandbrainsproject/tbp.monty/releases) and [tags](https://github.com/thousandbrainsproject/tbp.monty/tags) pages that list the tagged versions.

If we publish Monty to [PyPI](https://pypi.org/) or [Anaconda](https://anaconda.org/), then the specific published version could be installed via a package manager by specifying the published version.

# References

* [Semantic Versioning](https://semver.org/) is a standard for versioning software.
* [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) is a standard for commit messages.

# Future possibilities

If we were to adopt the [Conventional Commits specification](https://www.conventionalcommits.org/en/v1.0.0/#specification) then the version number updates could be automated based on the commit message.

The `tbp` CLI could be extended to include a `version` command that prepares a Pull Request to update the version number.

In the future, integrating into the `main` branch may become cumbersome if the pace of commits to `main` is high. In that case, we will update our versioning strategy. One way to do this is to create a `release` branch for each version that is used for releasing that version. This note is intended to highlight the possibility, but if we get there, we will get into the details then.