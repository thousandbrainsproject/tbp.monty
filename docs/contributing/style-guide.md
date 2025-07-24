---
title: Style Guide
---
# Code Style Guide

See the [Code Style Guide](./style-guide/code-style-guide.md).

# GitHub Actions

We use GitHub Actions to run our continuous integration workflows.

## GitHub Actions Naming Convention

### Workflow File Name

The workflow file name is the workflow name in snake_case, e.g., `potato_stuff.yml`.

### Workflow Name

The workflow name is a human-readable descriptive Capitalized Case name, e.g.,

```yml
name: Docs
```
```yml
name: Monty
```

```yml
name: Tools
```

```yml
name: Potato Stuff
```

### Job Name

The job name when in position of a key in a `jobs:` dictionary is a human-readable snake_case ending with `_<workflow_name>`.

When used as a value for the `name:` property, the job name is human-readable kebab-case ending with `-<workflow-name>`, e.g.,

```yml
jobs:
  check_docs:
    name: check-docs
```
```yml
jobs:
  install_monty:
    name: install-monty
```
```yml
jobs:
  test_tools:
    name: test-tools
```
```yml
jobs:
  check_style_potato_stuff:
    name: check-style-potato-stuff
```

# Documentation Style Guide

In general we try and stick to native markdown syntax, if you find yourself needing to use HTML, please chat with the team about your use case.  It might be something that we build into the sync tool.

## Headings

In a document your first level of headings should be the `#` , then `##` and so on.   This is slightly confusing as usually `#` is reserved for the title, but on readme.com the `h1` tag is used for the actual title of the document.

Use headings to split up long text blocks into manageable chunks.

Headings can be referenced in other documents using a hash link `[Headings](doc:style-guide#headings)`. For example [Style Guide - Headings](style-guide.md#headings)

All headings should use capitalization following APA convention. For detailed guidelines see the [APA heading style guide](https://apastyle.apa.org/style-grammar-guidelines/capitalization/title-case) and this can be tested with the [Vale](https://vale.sh/) tool and running `vale .` in the root of the repo.

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

Upload high quality images as people can click on the image to see the larger version.  You can add style attributes after the image path with `#width=300px` or similar.

For example, the following markdown creates the image below:

```markdown
![caption text](../figures/docs-only-example.png#width=300px)
```

<div style="display:flex; flex-direction:column; align-items:center">
  <img width="300px" src="https://files.readme.io/5b9d5a186a651f0ddc17022c3a95e65400991aa56a6d8523abefabd4db1dc6c4-touch_vs_vision.png" />
  <caption>caption text</caption>
</div>

> [!WARNING]
> Caption text is only visible on readme.com

## CSV Data Tables

You can inline CSV data tables in your markdown documents.  The following example shows how to create a table from a CSV file:

```
!table[../../benchmarks/example-table-for-docs.csv]
```

The CSV contains the following data:

```csv
Year,   Avg Global Temp. (°C),  Pirates | align right | hover Pirate Count
1800,   14.3,                   50 000
1850,   14.4,                   15 000
1900,   14.6,                    5 000
1950,   14.8,                    2 000
2000,   15.0,                      500
2020,   15.3,                      200
```

Which produces the following table:

!table[style-guide.csv]

Note that the CSV header row has bar separated syntax that allows you to specify the alignment of the columns `left` or `right` and the hover text.

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
