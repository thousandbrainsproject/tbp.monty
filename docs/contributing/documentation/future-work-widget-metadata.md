---
title: Future Work Widget Metadata
---

The future work documents have special Frontmatter metadata that is used to power the future-work widget.  The following fields are validated against allow lists defined in snippet files to ensure consistency and quality.

# Tags

Tags is a comma separated list of keywords, useful for filtering the future work items. [Edit future-work-tags.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-tags.md).

!snippet[../../snippets/future-work-tags.md]

# Skills

Skills is a comma separated list of skills that will be needed to complete this. [Edit future-work-skills.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-skills.md).

!snippet[../../snippets/future-work-skills.md]

# Estimated Scope

Very roughly, how big of a chunk of work is this? [Edit future-work-estimated-scope.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-estimated-scope.md).

!snippet[../../snippets/future-work-estimated-scope.md]

# Status

Is the work completed, or is it in progress? [Edit future-work-status.md](https://github.com/thousandbrainsproject/tbp.monty/edit/main/docs/snippets/future-work-status.md).

!snippet[../../snippets/future-work-status.md]

# RFC

Does this work item required an RFC? (These values are processed in the `validator.py` code) and can be of the form:

`https://github\.com/thousandbrainsproject/tbp\.monty/.*` `required` `optional` `not-required` `unknown`

# Contributor

The contributor field should be GitHub usernames, as these are converted to their avatar inside the table.(These values are processed in the `validator.py` code) and can be of the form:

`[a-zA-Z0-9][a-zA-Z0-9-]{0,38}`