---
title: Future Work Widget Metadata
---

The future work documents have special Frontmatter metadata that is used to power the future-work widget.

Here is an example of what the Frontmatter fields look like:

```example
---
title: Future Work Widget
description: Implement an interactive table view of the future work pages.
rfc: https://github.com/thousandbrainsproject/tbp.monty/blob/main/rfcs/0015_future_work.md
estimated-scope: medium
improved-metric: community-engagement
output-type: documentation
skills: github-actions, python, github-readme-sync-tool, S3, JS, HTML, CSS
contributor: codeallthethingz
status: in-progress
---
```
```template
---
title: 
description:
rfc: 
estimated-scope: 
improved-metric: 
output-type: 
skills: 
contributor: 
status: 
---
```

The following fields are validated against allow lists defined in snippet files to ensure consistency and quality.

# RFC

Does this work item require an RFC? (These values are processed in the `validator.py` code) and can be of the form:

`https://github\.com/thousandbrainsproject/tbp\.monty/.*` `required` `optional` `not-required`

# Estimated Scope

Very roughly, how big of a chunk of work is this? [Edit future-work-estimated-scope.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-estimated-scope.md). The estimated scope can be updated throughout working on a task as the scope becomes more clear.

!snippet[../../snippets/future-work-estimated-scope.md]

> [!NOTE] Notes on Some of the Fields
> **small** tasks in the future work table are usually still multi-day efforts. We only put reasonably large chunks of work into our future work documentation as even those marked as small need to be large enough to justify the effort of writing up a detailed future work page. For smaller items, check out our [GitHub issues](https://github.com/thousandbrainsproject/tbp.monty/issues) and other [ways to contribute](../../contributing/why-contribute.md). You can also have a look at the TODO comments throughout the Monty codebase or help up remove one of the many currently ignored ruff lint rules (listed in [pyproject.toml](../../../pyproject.toml)).
> 
> **medium** tasks are usually multi-week efforts that require several pieces of output (e.g. several PRs, data analysis, write ups, iterative testing, ...)
> 
> **large** tasks are usually multi-month efforts. For research items you could think of these tasks as the scope of a bachelor or master's thesis, and large enough to result in a potential publication. These items are often uncertain in the exact scope as it depends on the path someone chooses to take and intermediate results. They often require gaining a deeper understanding of Monty first and innovating upon what is currently there.
> 
> **unknown** is a label we often assign to open tasks that we have not scoped yet. We might realize the need for a specific feature but haven't thought through potential solutions deeply yet and hence can't give an estimate on the scope. Once the task is scoped, this can be updated.

# Improved Metric

What type of improvement does this work provide? [Edit future-work-improved-metric.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-improved-metric.md).

!snippet[../../snippets/future-work-improved-metric.md]

# Output Type

What type of output will this work produce? [Edit future-work-output-type.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-output-type.md).

!snippet[../../snippets/future-work-output-type.md]

> [!NOTE] Notes on Some of the Fields
> **publication** doesn't necessarily refer to publication in a peer-reviewed journal but could also be a general writeup or presentation of the results.

# Skills

Skills is a comma separated list of skills that will be needed to complete this. [Edit future-work-skills.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-skills.md).

!snippet[../../snippets/future-work-skills.md]

> [!NOTE] Notes on Some of the Fields
> **research** means the ability to run controlled experiments, interpret results, investigate unexpected behavior, come up with novel solutions, draw from existing literature, and communicate the results
> 
> **monty** refers to knowledge of how Monty works, how it is structured, and usually more detailed knowledge of the part of its algorithm that the specific future work item touches. This is usually required for tasks that involve innovative changes to the existing Monty algorithm or larger structural changes to the code base.

# Contributor

The contributor field should be GitHub usernames, as these are converted to their avatar inside the table.(These values are processed in the `validator.py` code) and can be of the form:

`[a-zA-Z0-9][a-zA-Z0-9-]{0,38}`

# Status

Is the work completed, or is it in progress? [Edit future-work-status.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-status.md).

!snippet[../../snippets/future-work-status.md]

> [!NOTE] Notes on Some of the Fields
> **open** are tasks that we recognize the need for but no one has started working on them.
> 
> **scoping** means that someone is currently in the process of writing or having an RFC reviewed or our team is actively discussing the approach in our research meetings.
> 
> **scoped** means an RFC or similar details document has been written for the planned approach.
> 
> **in-progress** are tasks that someone is currently working on. Note that this usually doesn't mean that no one else should work on them. Many tasks benefit from a second pair of eyes and external input so don't be deterred if an interesting task is already marked as in-progress.
> 
> **paused** means that someone has started work on this item but is currently not working on it. For example, an initial approach may have been tested but turned out not to work. New ideas and approaches are welcome.
> 
> **evergreen** refers to tasks that benefit from continual work and input. For example, updates to our documentation or designing new testbeds or demos for Monty. Most evergreen tasks are not listed in the future work table as they aren't large, self-contained chunks of work (see our [ways to contribute](../../contributing/why-contribute.md) page for more ways to contribute). However, some are significant projects of their own so we want to recognize people who are working on those and provide guidance in the task descriptions.
> 
> **completed** marks tasks that are done. We keep those in our docs for future reference and to acknowledge those that have contributed to it.

# Tags

Tags is a comma separated list of keywords, useful for filtering the future work items. [Edit future-work-tags.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-tags.md).

> [!NOTE] Tags are currently not used.

!snippet[../../snippets/future-work-tags.md]