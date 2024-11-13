---
title: Style Guide
---
# Code Style Guide

We follow the [PEP8](https://peps.python.org/pep-0008/) Python style guide.

## Code Formatting

We use **ruff** to check proper code formatting with a **line length of 88**.

A convenient way to ensure your code is formatted correctly is using the ruff formatter. If you use VSCode, you can get the ruff formatter extension and set it to format on save (modified lines only) so your code always looks nice and matches our style requirements.

## Code Docstrings

Doc strings should begin with a one-line summary, which ends with a period. For more details, see [PEP 257](https://peps.python.org/pep-0257/) docstring guide.

## Libraries

### Numpy Preferred over PyTorch

After discovering that torch-to-numpy conversions (and the reverse) were a significant speed bottleneck in our algorithms, we decided to consistently use NumPy to represent the data in our system. 

We still require the PyTorch library since we use it for certain things, such as multiprocessing. However, please use NumPy operations for any vector and matrix operations whenever possible. If you think you cannot work with NumPy and need to use Torch, consider opening an RFC first to increase the chances of your PR being merged.

Another reason we discourage using PyTorch is to add a barrier for deep-learning to creep into Monty. Although we don't have a fundamental issue with contributors using deep learning, we worry that it will be the first thing someone's mind goes to when solving a problem (when you have a hammer...). We want contributors to think intentionally about whether deep-learning is the best solution for what they want to solve. Monty relies on very different principles than those most ML practitioners are used to, and so it is useful to think outside of the mental framework of deep-learning. More importantly, evidence that the brain can perform the long-range weight transport required by deep-learning's cornerstone algorithm - back-propagation - is extremely scarce. We are developing a system that, like the mammalian brain, should be able to use _local_ learning signals to rapidly update representations, while also remaining robust under conditions of continual learning. As a general rule therefore, please avoid Pytorch, and the algorithm that it is usually leveraged to support - back-propagation! 

## Source Code Copyright and License Header

All source code files must have a copyright and license header. The header must be placed at the top of the file, on the first line, before any other code. For example, in Python:

```python
# Copyright <YEARS> Numenta Inc.
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
```

The `<YEARS>` is the year of the file's creation, and an optional sequence or range of years if the file has been modified over time. For example, if a file was created in 2024 and not modified again, the first line of the header should be `# Copyright 2024 Numenta Inc.`. If the file has been modified in consecutive years between 2022 and 2024, the header should be `# Copyright 2022-2024 Numenta Inc.`. If the file has been modified in multiple non-consecutive years in 2022, then in 2024 and 2025, the header should be `# Copyright 2022,2024-2025 Numenta Inc.`.

In other words, if you are creating a new file, add the copyright and license header with the current year. If you are modifying an existing file and the header does not include the current year, then add the current year to the header. You should never need to modify anything aside from the year in the very first line of the header.

> [!NOTE]
> While we deeply value and appreciate every contribution, the source code file header is reserved for essential copyright and license information and will not be used for contributor acknowledgments.

# Documentation Style Guide

## Headings

In a document your first level of headings should be the `#` , then `##` and so on.   This is slightly confusing as usually `#` is reserved for the title, but on readme.com the `h1` tag is used for the actual title of the document.

Use headings to split up long text block into managable chunks.  

Headings can be referenced in other documents using a hash link `[Headings](doc:style-guide#headings)`. For example [Style Guide - Headings](style-guide.md#headings)

All headings should use capitalization following APA convention. For detailed guidelines see the [APA heading style guide](https://apastyle.apa.org/style-grammar-guidelines/capitalization/title-case).

## Footnotes

Footnotes should be referenced in the document with a `[1] `notation that is linked to a section at the bottom `# Footnotes`

For example

```
This needs a footnote[1](#footnote1)

# Footnotes 
<a name="footnote1">1</a>: Footnote text
```

## Images

Images should be placed in `/docs/figures` in the repo. 

Images use `snake_case.ext`

Images should generally be `png` or `svg` formats.  Use `jpg` if the file is actually a photograph.

Use the optimal image size.  If the file is larger than the display port on readme, consider shrinking it. 

For example, the following markdown creates the image below:

```markdown
![caption text](../figures/docs-only-example.png)
```

<div style="text-align:center"><img src="https://files.readme.io/5b9d5a186a651f0ddc17022c3a95e65400991aa56a6d8523abefabd4db1dc6c4-touch_vs_vision.png" /><caption>caption text</caption></div>

> [!WARNING] 
> Caption text is only visible on readme.com


## Callouts

Readme supports four color coded callouts

```
> 👍 Something good
```

> 👍 Something good

```
> 📘 Information
```

> 📘 Information

```
> ⚠️ Warning
```

> ⚠️ Warning

```
> ❗️ Alert
```

> ❗️ Alert

## Numbers

Billions of people use commas as a thousands separator, and billions use the period as the thousands separator.  As this documentation is expected to be widely used, we will **use space as the separator**, as this is the [internationally recommended convention](https://en.wikipedia.org/wiki/Decimal_separator).

For example, `1 million` is written numerically as `1 000 000`.