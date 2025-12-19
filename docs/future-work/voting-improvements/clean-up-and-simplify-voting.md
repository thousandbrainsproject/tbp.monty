---
title: Clean up and Simplify Voting
description: Simplify some of the voting code to make it more efficient and easier to understand.
rfc: optional
estimated-scope: medium
improved-metric: speed, learning-experience
output-type: PR, documentation
skills: python, monty, refactoring
contributor: 
status: open
---

The way we do voting is a bit nested and overly complex in the code. We could simplify it and add some clarifying documentation around it.

Additionally, we could use the hypothesized pose earlier on to filter out how votes are being applied to speed up the process.