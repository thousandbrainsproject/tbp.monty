---
title: Better Website Platform
rfc: required
estimated-scope: large
improved-metric: community-engagement
output-type: website
skills: github-actions, s3, javascript, html, css, web-frameworks, css-frameworks
---

The current website (thousandbrains.org) is hosted on WordPress.  It has a number of drawbacks:

- A very basic two versioning system: live and staging.
- It costs money.
- Making updates requires a lot of manual, time consuming, error prone work.
- Global changes require manual updates to each page.
- There are security risks with the plugins on WordPress.
- The website is slow.
- There is a lot of cognitive load to figure out how to customize the website.
- The source code for the website is not available for modification by the community and goes against our open source philosophy.

# Implementation Considerations

As this will be mostly maintained by the TBP team, we have a preference for the css-framework to be TailwindCSS.