---
title: Better Documentation Platform
rfc: required
estimated-scope: large
improved-metric: community-engagement
output-type: documentation
skills: github-actions, python, github-readme-sync-tool, s3, javascript, html, css, markdown, yaml
---

The current platform for documentation is readme.com.  It has a number of drawbacks:
- Doesn't support preview versions unless you have an account with them.  We want preview versions to be available to everyone in the community, but hidden from the main documentation navigation.
- Doesn't support custom markdown plugins — notably we'd like to extend our documentation to support other display types:
  - LaTeX support
  - Mermaid
  - Jupyter notebooks
  - JavaScript integration
  - etc...
- The display format is limited in size.
- Doesn't allow complete control over theming.
- Doesn't provide UX control over the navigation, menus, etc.
- Local development support.

We are open to new platform suggestions, but our initial research suggests that a homegrown solution would be the best fit and more extensible for our future needs.

A proof of concept was created by modifying the `github_readme_sync` tool and can be viewed here:
https://github.com/codeallthethingz/tbp.monty/pull/14/files

> [!NOTE]
> This is a large task that involves multiple skills and technologies. If you're interested in contributing but don't have expertise in all areas, you can volunteer for part of this task. For example, if you're skilled in HTML and web design, you could design a new page without needing to know S3 or the current readme sync tool. You can partner with a TBP team member to accomplish this task together.

---

# Requirements

## Core Platform
- Versioned doc releases that match our code (major.minor).
- Atomic deploys from our GitHub repository.
- Preview versions that are hidden from the main documentation navigation.
- Permanent links to specific documents / versioned documents.
- Source documents in markdown.
- If its not homegrown, it should have API access to sync the content from a GitHub action.
- Local development support.

## Content and Presentation
- Code blocks.
- Images.
- Inline videos.
- Complete control over theming.
- Complete control over the navigation, menus, etc.
- Left hand navigation.
- Responsive layouts for mobile, tablet, and desktop.
- Header and footer branding.

## Navigation and Linking
- Deep linking.
- Redirects for deleted or renamed documents.
- Redirects for legacy URLs.
- SEO friendly slugs.
- Social media sharing using Open Graph tags.

## Search and Analytics
- Search.
- Integration with Google Analytics.

## Glossary Integration
- Automatic identification of glossary terms from `docs/overview/glossary.md` in all markdown files
- Subtle hover tooltips displaying glossary definitions when users hover over terms
- This feature may require the new platform as readme.com may not support the necessary markdown extensions or JavaScript integration