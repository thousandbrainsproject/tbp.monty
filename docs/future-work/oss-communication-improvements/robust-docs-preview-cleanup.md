---
title: Robust Docs Preview Cleanup
rfc: not-required
estimated-scope: medium
improved-metric: infrastructure
skills: github-actions, bash
output-type: automation
---

Make the documentation preview GitHub Actions system more robust by tracking all preview versions created for a PR and cleaning them all up when the PR is merged or closed.

## Current Problem

When a PR creates a documentation preview, it uses a version name like `{monty_version}-{branch_name}`. If the version number changes during the Pull Request's lifetime (e.g., someone bumps the version), a new preview is created with the new version, but the old preview version is never cleaned up.

Currently, `cleanup_docs_preview.yml` only deletes the preview for the version that exists at the time the PR is closed/merged, leaving orphaned previews from earlier versions.

## Solution

Implement a tracking system that:

1. **Track preview versions**: When `docs_preview.yml` successfully creates a preview, store the preview version name in a hidden GitHub comment (using HTML comments similar to the existing `pin_comment` action pattern).

2. **Cleanup all tracked versions**: When `cleanup_docs_preview.yml` runs on PR close/merge, read the tracking comment, extract all preview versions that were created for that PR, and delete them all.

## Important Files

- `.github/workflows/docs_preview.yml` - Add step to track preview version after successful creation
- `.github/workflows/cleanup_docs_preview.yml` - Add step to read tracking comment and delete all tracked previews

