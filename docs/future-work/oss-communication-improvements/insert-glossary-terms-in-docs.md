---
title: Insert Glossary Terms in Docs
rfc: required
estimated-scope: small
improved-metric: learning-experience
skills: github-readme-sync-tool, css, javascript
output-type: documentation
---

Automatically identify glossary terms from `docs/overview/glossary.md` in all markdown files within the `/docs` directory and add subtle hover tooltips that display the glossary definition when users hover over those terms.

## Requirements

- Parse (or refactor into snippets) the glossary terms from `docs/overview/glossary.md`
- Scan all markdown files in `/docs` directory
- Identify instances of glossary terms in the documentation
- Add hover tooltips that display the glossary definition
- Tooltips should be subtle and non-intrusive to the reading experience

## Implementation Considerations

This feature may require waiting for the documentation platform rebuild (see [Better Documentation Platform](better-documentation-platform.md)) as readme.com does not support the necessary markdown extensions or JavaScript integration required for hover tooltips.

If implemented with the current github_readme_sync tool, it would need to:
- Parse glossary terms and their definitions
- Process markdown files to identify glossary term occurrences
- Insert appropriate markdown/html syntax for hover tooltips
- Ensure definitions link back to the full glossary page

## Expected Outcomes

- Enhanced documentation readability with contextual definitions
- Reduced need to navigate away from content to understand terms
- Consistent use of glossary terminology throughout documentation
- Better user experience for both new and experienced users

