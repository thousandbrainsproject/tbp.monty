---
title: Documentation
---
> 📘 This page is about contributing Documentation
>
> For current Documentation see [Getting Started](../how-to-use-monty/getting-started.md)

# Overview

Our documentation is held in markdown files in the Monty repo under the [`/docs` folder](../../docs/). This documentation is synchronized to readme.com for viewing whenever a change is made. The order of sections, documents, and subdocuments is maintained by a hierarchy file called `/docs/hierarchy.md`. This is a fairly straightforward markdown document that is used to tell readme how to order the categories, documents and sub-documents.

> 📘 Edits to the documentation need to be submitted in the form of PRs to the Monty repository.


# Relative Links

Links to other documents should use the standard markdown link syntax, and should be relative to the documents location.

```markdown
relative link in the same directory
[Link Text](placeholder-example-doc.md)

a relative link, with a deep-link to a heading
[Link Text](../contributing/placeholder-example-doc.md#relative-links)
```

These links will work even if you're on a designated version of the documentation.

# Modifying a Document or Sub-Document

This is the simplest flow.  To modify a document simply edit the markdown file in your forked version of the monty repository and commit the changes by following the normal [Pull Requests](pull-requests.md) process.

# Creating a Document

To create a new document, create the new file in the category directory, then add a corresponding line in the `/docs/hierarchy.md` file.

```markdown Markdown
# my-category: My Category
- [my-new-doc](/my-category/new-placeholder-example-doc.md)
- [some-existing-doc](/my-category/placeholder-example-doc.md)
```

Then, create your markdown document `/docs/my-category/new-placeholder-example-doc.md` and add the appropriate Frontmatter.

```markdown
---
title: 'New Placeholder Example Doc'
---
# My first heading
```

> 🚧 Quotes
>
>Please put the title in single quotes and, if applicable, escape any single quotes using two single quotes in a row.
Exapmle: `title: 'My New Doc''s'`

> 🚧 Your title must match the url-safe slug
>
>So if your title is `My New Doc's` then your file name should be `my-new-docs.md`

Continue with the [Pull Requests](pull-requests.md) process.

## Creating Sub-Documents

Documents that are nested under other documents require that you create a folder with the same name as the parent document but without the `.md` extension.  Then, you place any sub-documents in that folder.  For example, if you were creating a document called `new-placeholder-example-doc.md` beneath the document `Category One/some-existing-doc.md` file you would create a folder called `category-one/some-existing-doc` and place the new document in that folder.


And then update the `hierarchy.md` file

```markdown markdown
# category-one: Category One
- [some-existing-doc](category-one/some-existing-doc.md)
  - [new-doc](category-one/some-existing-doc/new-placeholder-example-doc.md)

# category-two: Category Two
...
```

Continue with the [Pull Requests](pull-requests.md) process.

# Reordering a Document

If the move is within a category or sub-pages within a page, you can simply edit the `hierarchy.md` file and update the locations by moving the lines around.

If you are changing the parent path of a document (ie, sub-page -> page, or page -> sub-page, or page/sub-page -> new category, or sub-page -> new page, then along with updating the `hierarchy.md` file you also must update the folder structure to make sure the document is correctly located.  The sync tool will fail with a pre-check error saying there is a mismatch between the hierarchy file and the location on disk if they do not match up.

Continue with the [Pull Requests](pull-requests.md) process.

> 🚧 You cannot reorder categories as the readme.com API does not support this.
>
> Changes to the category order should be done in the readme.com UI and reflected in the `hierarchy.md` file.

# Deleting or Renaming a Document

If a document is well established (it has been around for more than 6 months), people may be using permalinks to it. Therefore, it is a good idea to create a redirect file rather than deleting or renaming it. To do this, set the document to hidden with a relocation link to a relevant area or new document location. Hidden files are reachable from the URL, just not shown in the navigation.

```markdown
---
title: 'Badly Named Doc'
hidden: true
---
>  ⚠️ this document has moved to <insert link>
```

Continue with the usual [Pull Requests](pull-requests.md) process.

# Creating a New Top-Level Category

To create a new category, simply create a new folder inside the `/docs` folder and add a reference to it in the `hierarchy.md` file.  Categories in the hierarchy file need a slug and title separated by a colon.

```markdown markdown
# category-one: Category One

# category-two: Category Two
```

# Checking Links

In our documentation sync tool there is a flag to check internal links, image references and hierarchy file references.  This is a good way to ensure that all links are working correctly before submitting a PR.

To check the links, [activate the conda environment](../how-to-use-monty/getting-started.md#miniconda), and then run the following command:

``` 
python -m tools.github_readme_sync.cli check docs
```

# Images

See the [Style Guide](style-guide.md#images) images section for details about creating and referencing images correctly.

# VS Code Snippets

> 👍 You have access to VS Code snippets

When you checkout the repository, you have access to markdown snippets for tables, code blocks, warnings and more.  While your cursor is in a markdown file, press `CMD + Shift + P` and select `Insert snippet` to select a desired documentation snippet.

# Style Guide

The documentation [Style Guide](style-guide.md#documentation-style-guide)

# Versioning

The Monty documentation uses the first two parts of semantic versioning (semver), as there is nothing to document for patch changes.  You can read about semver here <https://semver.org/>.