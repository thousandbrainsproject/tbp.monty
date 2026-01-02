---
title: Automate Benchmark Experiments and Analysis
description: Automate the running and analysis of benchmark experiments.
rfc: https://github.com/thousandbrainsproject/tbp.cli/blob/main/rfcs/0002_monty_benchmarks.md
estimated-scope: large
improved-metric: infrastructure
output-type: PR, automation
skills: python, bash
contributor: codeallthethingz
status: paused
---

This item consists of two RFCs.

# 1. RFC to Convert to Python
The first is to convert the bash tool, to python.

RFC: https://github.com/thousandbrainsproject/tbp.cli/blob/main/rfcs/0001_python_reimplementation.md

## Summary

The tool would be better suited to Python. Transitioning from a Bash implementation to Python offers several advantages:
- **Improved Maintainability**: Programming languages make it easier to manage as the logic grows in complexity.
- **Alignment with Standards**: Using Python is our team’s tooling standard.
- **Enhanced Extensibility**: Python’s flexibility simplifies the process of adding new features and capabilities.

## Motivation

Running the Monty benchmarks is time-consuming for researchers. Automating this process with the cli tool would be beneficial. However, implementing this feature in Bash proved challenging due to the numerous moving parts, and would be better suited to a full-fledged programming language.

This RFC proposes rewriting the tool in Python to make maintenance and future enhancements easier. Using Python also ensures that both the research and engineering teams can more easily contribute to and maintain the tool moving forward.

This rewrite will also allow us to add an option for JSON output format.

## Implementation Notes

This work was partially completed in this branch which can be used as a reference.  The tool has since diverged from the original implementation, so those changes need to be incorporated.

### Branch Structure

The paused implementation has the current merged and open branches.

```
main
|- switch_to_python
|- efs_workstation
|- monty_star
|- automate-benchmarks (merged)
|- python-impl (merged - added easy_setup)
|- ls_instances (merged)
```

# 2. RFC to Automate the Running and Analysis of Benchmark Experiments
Once the tool is converted to python, the second part is to automate the running and analysis of benchmark experiments.

RFC: https://github.com/thousandbrainsproject/tbp.cli/blob/main/rfcs/0002_monty_benchmarks.md

## Summary

This RFC proposes a new feature that allows the research team to run benchmark experiments on our AWS infrastructure, and have the results from those benchmarks update the CSV files in your local tbp.monty repository in `/benchmarks/results`.

## Motivation

Currently, updating the benchmarks is a manual process that takes up valuable researcher time. Automating this process would be beneficial for the following reasons
- Manual updating is error prone.
- Manual updating is time consuming.
- Automation nudges towards more frequent updates without additional overhead.
