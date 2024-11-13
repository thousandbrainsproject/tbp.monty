---
title: Reviewing a Pull Request
---
Thank you for reviewing a Pull Request. Below are some guidelines to consider.

# Prerequisite: Automated Checks Pass

Before a Pull Request is reviewed, it should pass the automated checks. If the automated checks fail, you may want to wait until the author makes the required updates.

# Merge Criteria

For a Pull Request to be **Merged**, it must be **Approved** by at least one **Maintainer**, and the **Pre-merge checks** must pass. See [Pull Request Flow](pull-request-flow.md) for additional details.

If you are not a **Maintainer**, you may still review the Pull Request and provide insights. While unable to **Approve** the Pull Request, you still can influence what sort of code is included in Monty.

# Reviewing

Multiple aspects must be considered when conducting a Pull Request Review. **Generally, Maintainers should favor approving a Pull Request if it improves things** along one or several of the dimensions listed below.

## Design

Consider the overall design of the change. Does it make sense? Does it belong in Monty? Is now the time for this functionality? Does it fit with the design patterns within Monty?

## Functionality

Does the code do what the author intended? Are the changes good for both end-users and developers? Is there a demo available so that you can evaluate the functionality by experiencing it instead of reading the code? 

## Performance

Does the code perform as the author intended? Is it an improvement over current performance on our benchmarks (or at least no degradation)?

## Complexity

Complex code is more difficult to debug. The code should be as simple as possible, but not simpler.

## Tests

If applicable, ask for unit, integration, or end-to-end tests. The tests should fail when the code is broken. Complex tests are more difficult to debug. The tests should be as simple as possible, but not simpler.

## Benchmarks

If applicable, ask for benchmarks. Existing benchmarks should not worsen with the change.

## Naming

A good name is unambiguously telling the reader what the variable, class, or function is for.

## Style

Does the code follow the [Style Guide](../style-guide.md)? Prefer automated style formatting. 

## Comments

Are the comments necessary? Usually, comments should explain _why_, not the what or how.

Note that comments are distinct from _documentation_ in the code, such as class or method descriptions.

## Documentation

Is the change sufficiently documented? Can a user understand the new code or feature without any other background knowledge (like things discussed in Pull Request review comments or meetings)? Does every class and function have appropriate doc strings explaining their purpose, inputs and outputs? 

Note: code documentation can also be too verbose. If docstrings are getting too long, consider adding a new page to overall documentation instead. Comments don't need to explain every line of code.

## Every Line

Generally, look at every line of the Pull Request. If asked to review only a portion of the Pull Request, comment on what was reviewed.

If you have difficulty reading the code, others will also have difficulty; ask for clarification. Approach each pull request with fresh eyes and consider whether other uses will understand the changes. Just because you understand something (maybe because you talked about it in another conversation or you spent a lot of time thinking about the change) doesn't mean that others will. The code should be intuitive to understand and easy to read.

If you feel unqualified to review some of the code, ask someone else to review that portion of the code.

## Context

"Zoom out." Consider the change in the broader context of the file or the system as a whole. Should the changes now include refactoring or re-architecture?

## Scope

Does the pull request have an appropriate scope? Each pull request should address one specific problem or add one specific feature. If there are multiple additions in a pull request, consider asking the contributor to split them into separate pull requests. 

## Good Things

If you see things you like, let the author know. This keeps the review from being fully focused on mistakes and invites future contributions. Celebrate the awesome work of this individual.

# People

Remember the people involved in the Pull Request. Be welcoming. If there is a conflict or coming to an agreement is difficult, changing the medium from Pull Request comments to audio or video can help. Sometimes, complex topics can be more easily discussed with a real-time meeting, so don't hesitate to suggest a time when everyone can meet synchronously.  
Be cordial and polite in your review. A Pull Request is a gift and should be treated as such. Assume the best of intentions in submitted Pull Requests, even if that Pull Request is eventually rejected.

# Request Changes

Feel free to **Request Changes** on a Pull Request. Once changes are requested, it is up to the author to **Update the Pull Request**, at which point the Pull Request will again be **Reviewed**. See [Pull Request Flow](pull-request-flow.md) for additional details.

We encourage including the prefix `nit:` for suggestions or changes that are minor and wouldn’t prevent you from approving the Pull Request. This helps distinguish nitpicks from essential, blocking requests.

# Approve

A Pull Request only requires **Approval** from one **Maintainer**.

Once **Approved**, before the Pull Request is **Merged**, pre-merge checks must pass. See [Pull Request Flow](pull-request-flow.md) for additional details.