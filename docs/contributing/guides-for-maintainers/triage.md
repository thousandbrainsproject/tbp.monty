---
title: Triage
---
> [!NOTE]
> **For Maintainers**

The philosophy behind triage is to check issues for validity and accept them into the various **Maintainer** workflows. Triage is not intended to engage in lengthy discussions on Issues or review Pull Requests. These are separate activities.

The typical triage outcomes are:

- Label and accept the Issue or Pull Request with `triaged`.
- Label, request more information, and mark the Issue with `needs discussion` and `triaged`.
- Reject by closing the Issue or Pull Request with `invalid`.

# Issue Triage

> [!NOTE]
> Triage link (is issue, is open, is not triaged)
>
> <https://github.com/thousandbrainsproject/tbp.monty/issues?q=is:issue+is:open+-label:triaged>

The desired cadence for Issue Triage is at least once per business day.

A **Maintainer** will check the Issue for validity.

Do not reproduce or fix bugs during triage.

## Evaluation Criteria

### Title

A short descriptive title.

### Description

Ideally, the Issue creator followed the instructions in the Issue templates.

If not, and more information is needed, _do not close the Issue_. Instead, proceed with triage, request more information by commenting on the Issue, and add a `needs discussion` label to indicate that additional information is required. Remember to add the `triaged` label to indicate that the Issue was triaged after you applied any additional labels.

### Validity

A valid Issue is on-topic, well-formatted, contains expected information, and does not violate the code of conduct.

## Labels

Multiple labels can be assigned to an Issue.

- `bug`: Apply this label to bug reports.
- `documentation`: Apply this label if the Issue relates to documentation without affecting code.
- `enhancement`: Apply this label if the Issue relates to new functionality or changes in functional code.
- `infrastructure`: Apply this label if the Issue relates to infrastructure like GitHub, continuous integration, continuous deployment, publishing, etc.
- `invalid`: Apply this label if you are rejecting the Issue for validity.
- `needs discussion`: Apply this label if the Issue is missing information to determine what to do with it.
- `triaged`: At a minimum, apply this label if the Issue is valid and you have triaged it.

## Priority & Severity

Do not assign priority or severity to Issues (see: [RFC 2 PR and Issue Review](https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0002_pr_and_issue_review.md#issue)).

## Assignment & Mentions

Do not assign **Maintainers** to Issues. Issues remain unassigned so that anyone can work on them (see: [RFC 2 PR and Issue Review](https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0002_pr_and_issue_review.md#feature-requests-1)).

If you feel that someone should be notified of the Issue, make a comment and mention them in the comment.

# Pull Request Triage

The desired cadence for Pull Request Triage is at least once per business day.

## 1. Pending CLA

First, review any Pull Requests pending CLA.

> [!NOTE]
>Pending CLA link (is pull request, is open, is not a draft, is not triaged, is pending CLA)
>
> <https://github.com/thousandbrainsproject/tbp.monty/pulls?q=is%3Apr+is%3Aopen+-label%3Atriaged+draft%3Afalse+label%3Acla>

If the Pull Request CLA check is passing (you may need to rerun the CLA check), remove the `cla` label.

## 2. Triage

> [!NOTE]
> Triage link (is pull request, is open, is not a draft, is not triaged, is not pending CLA)
>
> <https://github.com/thousandbrainsproject/tbp.monty/pulls?q=is%3Apr+is%3Aopen+-label%3Atriaged+draft%3Afalse+-label%3Acla>

First, check if the Pull Request CLA check is passing. If the check is not passing, add the `cla` label and move on to the next Pull Request. The skipped Pull Request will be triaged again after the CLA check is passing.

A **Maintainer** will check the Pull Request for validity.

There is no priority or severity applied to Pull Requests.

A valid Pull Request is on-topic, well-formatted, contains expected information, does not depend on an unmerged Pull Request, and does not violate the code of conduct.

A Draft Pull Request is ignored and not triaged.

## Evaluation Criteria

### Title

The title should be short and descriptive. Because our commit messages use the pull request title, the title should be prefixed with a label based on the [Conventional Commits 1.0.0 Standard](https://www.conventionalcommits.org/en/v1.0.0/).

A breaking change is communicated by appending `!` after the label. This correlates with `MAJOR` in [RFC 7 - Monty versioning][monty-versioning].

Here are some examples of titles that meet these guidelines.

```
docs: remove obsolete MotorSystemConfig references
refactor!: Use TypedDict for SensorObservations
fix: OmniglotEnvironment includes view_finder ProprioceptiveState
chore: version 0.18.0
ci: tidy ci and prepare for turning on merge queue
fix: run_parallel no longer hands on mac
refactor: Address mypyp issues
test: Make hierarchy test less brittle
```

`tbp.monty` code adopts the following label `<type>`s:

- `fix`: Fix to a bug in the **src/tbp/monty** codebase. This correlates with `PATCH` in [RFC 7 - Monty versioning][monty-versioning].
- `feat`: Introduction of a new feature to the **scr/tbp/monty** codebase. This correlates with `MINOR` in [RFC 7 - Monty versioning][monty-versioning].
- `build`: Change to the build system or external dependencies.
- `ci`: Change to our GitHub Actions configuration files and scripts.
- `docs`: Documentation only update.
- `perf`: Performance improvement.
- `refactor`: A **src/tbp/monty** code change that neither fixes a bug nor adds a feature.
- `style`: Change that does not affect the meaning of the code (white-space, formatting, etc.).
- `test`: Adding or correcting tests.
- `chore`: The commit is a catch all for work outside of the types identified above. For example, the commit affects infrastructure, tooling, development, or other non-Monty framework code.
- `rfc`: RFC proposal.
- `revert`: Commit that reverts a previous commit.

Even with the above guidance, sometimes there might be doubt or disagreement on what type to use. If it seems like multiple types are appropriate, maybe there should be multiple pull requests. Otherwise, discuss in the pull request and select a best-fit type.

[monty-versioning]: https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0007_monty_versioning.md

> [!NOTE]
> `fix`, `feat`, and `refactor` types refer only to the **src/tbp/monty** codebase. Adding a new tool is not a feature. Fixing a tool is not a feature. Refactoring a tool is not a refactor of `src/tbp/monty`.
>
> By restricting these types to **src/tbp/monty**, it enables us to rapidly distinguish when we need to increment the `tbp.monty` version for publishing. Only `fix`, `feat`, `refactor` commits will be relevant to determine whether a `MINOR` or `MAJOR` version increment is required.
>
> By default, a version increment is `PATCH`. If there is a `feat` commit present, then the version increment is `MINOR`. If there is a breaking change commit present: `fix!`, `feat!`, `refactor!`, then the version increment is `MINOR` if and only if `MAJOR == 0`, and it is `MAJOR` otherwise.

> [!NOTE]
> In `tbp.monty`, we do not use the `BREAKING CHANGE` optional footer to indicate a breaking change.

See [RFC 10 - Conventional Commits](https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0010_conventional_commits.md) for more details.

### Description

If the Pull Request claims to resolve an Issue, that Issue is linked and valid.

If the Pull Request is standalone, it clearly and concisely describes what is being proposed and changed.

If the Pull Request is related to a previous RFC process, the RFC document is referenced.

### Commit History

Pull Request branches from a recent `main` commit.

Pull Request does not depend on another unmerged Pull Request.

> [!NOTE]
> Pull Requests that depend on unmerged Pull Requests add unnecessary complexity to the review process: **Maintainers** must track the status of multiple Pull Requests and re-review them if the dependent Pull Request is updated. Such dependency is much easier for the Pull Request author to track and to submit the Pull Request after all dependent code is already merged to `main`.

It is OK if the commit history is messy. It will be "squashed" when merged.

## Labels

Multiple labels can be assigned to a Pull Request. For example, an `enhancement` can come with `documentation` and continue along the Pull Request Flow after being `triaged`.

- `cla`: Apply this label if the Pull Request CLA check is failing.
- `documentation`: Apply this label if the Pull Request relates to documentation without affecting code.
- `enhancement`: Apply this label if the Pull Request implements new functionality or changes functional code.
- `infrastructure`: Apply this label if the Pull Request concerns infrastructure such as GitHub, continuous integration, continuous deployment, publishing, etc.
- `invalid`: Apply this label if you are rejecting the Pull Request for validity.
- `rfc:proposal`: Apply this label if the Pull Request is a [Request For Comments (RFC)](../request-for-comments-rfc.md).
- `triaged`: At a minimum, apply this label if the Pull Request is valid, you triaged it, and it should continue the [Pull Request Flow](../pull-requests/pull-request-flow.md).

## Priority & Severity

Do not assign priority or severity to Pull Requests.

## Assignment

Use your judgement to assign the Pull Request to one or more **Maintainers** for review (this is the [GitHub Assignees feature](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/assigning-issues-and-pull-requests-to-other-github-users)). If you are the author, you _may not_ assign the Pull Request to yourself. Note that for RFCs, the meaning of the Assignees list is different (see [Request For Comments (RFC)](../request-for-comments-rfc.md)).

Use your judgement to request a review for the Pull Request from one or more people (this is the [GitHub Reviewers feature](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review)).

If you feel that someone should be notified of the Issue, make a comment and mention them in the comment.
