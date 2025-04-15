- Start Date: 2025-04-11
- RFC PR: https://github.com/thousandbrainsproject/tbp.monty/pull/246

# Summary

[Conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) allow for rapid human and automated interpretation of commits.

# Motivation

As of this writing, we largely adopted the habit of labeling our commits as per https://www.conventionalcommits.org/en/v1.0.0/. 22 of the last 23 commits were done formatted in conventional commit style and currently active PRs authored by the TBP team adopt conventional commits as the pull request name.

Adopting conventional commits allows us to rapidly determine what version ([RFC 7 - Monty versioning](0007_monty_versioning.md)) to assign to new changes.

# Conventional Commit Style

`tbp.monty` commits use the [conventional commits 1.0.0 standard](https://www.conventionalcommits.org/en/v1.0.0/).

## Commit Messages

> [!NOTE]
> The following commit message guidance applies only to commits made to the `main` branch when a Pull Request is merged. This guidance does not apply to commit messages elsewhere, for example, on your working branch while working on a Pull Request.

The commit message to `main` branch should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types

`tbp.monty` code adopts the following `<type>`s:

- `fix`: Fix to a bug in the **src/tbp/monty** codebase. This correlates with `PATCH` in [RFC 7 - Monty versioning](0007_monty_versioning.md).
- `feat`: Introduction of a new feature to the **scr/tbp/monty** codebase. This correlates with `MINOR` in [RFC 7 - Monty versioning](0007_monty_versioning.md).
- `build`: Change to the build system or external dependencies.
- `ci`: Change to our GitHub Actions confguration files and scripts.
- `docs`: Documentation only update.
- `perf`: Performance improvement.
- `refactor`: A **src/tbp/monty** code change that neither fixes a bug nor adds a feature.
- `style`: Change that does not affect the meaning of the code (white-space, formatting, etc.).
- `test`: Adding or correcting tests.
- `chore`: The commit is a catch all for work outside of the types identified above. For example, the commit affects infrastructure, tooling, development, or other non-Monty framework code.
- `rfc`: RFC proposal.
- `revert`: Commit that reverts a previous commit.

Even with the above guidance, sometimes there might be doubt or disagreement on what type to use. If it seems like multiple types are appropriate, maybe there should be multiple pull requests. Otherwise, discuss in the pull request and select a best-fit type.

### Breaking Changes

A breaking change is communicated by appending `!` after the type/scope. This correlates with `MAJOR` in [RFC 7 - Monty versioning](0007_monty_versioning.md).

> [!NOTE]
> `fix`, `feat`, and `refactor` types refer only to the **src/tbp/monty** codebase. Adding a new tool is not a feature. Fixing a tool is not a feature. Refactoring a tool is not a refactor of `src/tbp/monty`.
>
> By restricting these types to **src/tbp/monty**, it enables us to rapidly distinguish when we need to increment the `tbp.monty` version for publishing. Only `fix`, `feat`, `refactor` commits will be relevant to determine whether a `MINOR` or `MAJOR` version increment is required.
>
> By default, a version increment is `PATCH`. If there is a `feat` commit present, then the version increment is `MINOR`. If there is a breaking change commit present: `fix!`, `feat!`, `refactor!`, then the version increment is `MINOR` if and only if `MAJOR == 0`, and it is `MAJOR` otherwise.

> [!NOTE]
> In `tbp.monty`, we do not use the `BREAKING CHANGE` optional footer to indicate a breaking change.

### Optional Scopes

`tbp.monty` does not use optional scopes.

### Description

This is the usual one-line message summarizing the commit.

### Optional Body

`tbp.monty`, usually, does not use optional body.

### Optional Footers

`tbp.monty` sometimes uses `Co-authored-by: ghusername <github_email@example.com>` footer(s) if the pull request has multiple authors.

## Examples

### Commit message with `!` to draw attention to breaking change

```
refactor!: motor policies are an attribute of MotorSystem
```

### Commit message introducing a new non-breaking feature

```
feat: logging GoalState objects
```

### Commit message incrementing Monty version

```
chore: version 0.3.0
```

### Commit message fixing a bug in a non-breaking manner

```
fix: Omniglot environment
```

### Commit message with scope

> [!NOTE]
> We do not use scopes.

```
feat(motor_sys): add MotorModule
```

### Commit message with scope and `!` to draw attention to breaking change

> [!NOTE]
> We do not use scopes.

```
feat(api)!: on the wire CMP
```

### Commit message with multi-paragraph body and multiple footers

> [!NOTE]
> We typically don't use these types of commits.

```
fix: prevent racing of requests

Introduce a request id and a reference to latest request. Dismiss
incoming responses other than from latest request.

Remove timeouts which were used to mitigate the racing issue but are
obsolete now.

Reviewed-by: Z
Refs: #123
```

## Pull Requests

By default, Pull Request titles are used as the suggested commit message when merging. **Maintainers** are responsible for ensuring that any merges to `main` contain a conventional commit message as defined in this RFC.

It is helpful to rename Pull Requests to match the intended commit message to communicate the scope of the Pull Request and to reduce the chance of mistakes in commit message when merging. However, we will not enforce a Pull Request naming standard so as not to introduce an additional barrier to contributors.
