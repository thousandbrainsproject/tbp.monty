---
title: Request For Comments (RFC)
---
> [!NOTE]
> This page is about contributing RFCs
>
> For existing RFCs, see the [`rfcs` folder](https://github.com/thousandbrainsproject/tbp.monty/tree/main/rfcs).
>
> For RFCs in progress, see issues with `rfc:proposal` label.

**The Request for Comments (RFC) process is intended to provide a consistent and controlled path for substantial changes** to Monty so that all stakeholders can be confident about the project's direction. It should also help avoid situations where a contributor spends a lot of time implementing a feature or change that eventually does not get merged.

Many changes, including bug fixes, smaller changes, and documentation improvements, can be implemented and reviewed via [Pull Requests](pull-requests.md).

Substantial changes should undergo a design process and create consensus among the Monty community and Maintainers.

If you are unsure whether you intend to work on a substantial change, create a [new Feature Request issue](https://github.com/thousandbrainsproject/tbp.monty/issues/new?template=02_feature_request.yml). Maintainers will comment and let you know if it can benefit from an RFC.

# Not Another Process

The process here is intended to be as lightweight as reasonable for the present circumstances and not impose more structure than necessary. If you feel otherwise, please consider creating an RFC to update this process.

# When You Need to Follow This Process

You need to follow this process if you intend to make substantial changes to Monty or its associated open-source framework and workflows. What constitutes a substantial change is evolving based on community norms and varies depending on what part of the ecosystem you are proposing to change, but may include the following:

- Any fundamental iteration of the Monty architecture
- Removing Monty features
- Any changes to the Cortical Message Protocol (CMP)
- Breaking changes to the API
- Diverging from Monty's brain-inspired philosophy and principles

Some changes do not require an RFC:

- Rephrasing, reorganizing, refactoring, or otherwise "changing shape does not change meaning."
- Additions that strictly improve objective, numerical quality criteria (warning removal, speedup, better platform coverage, more parallelism, etc.)

**If you submit a pull request for a substantial change without going through the RFC process, it may be closed with a request to submit an RFC first.**

# Before Creating an RFC

A hastily proposed RFC can hurt its chances of acceptance. Low-quality proposals, proposals for previously rejected features, or those that don't fit into the near-term roadmap may be quickly rejected, which can demotivate the unprepared contributor. Laying some groundwork ahead of the RFC can make the process smoother.  Please have a look at some of our past RFCs to see their scope. [`rfcs` folder](https://github.com/thousandbrainsproject/tbp.monty/tree/main/rfcs).

Although there is no single way to prepare for submitting an RFC, it is generally a good idea to pursue feedback from other project developers beforehand to ascertain that the RFC may be desirable; having a consistent impact on the project requires concerted effort toward consensus-building.

The most common preparation for writing and submitting an RFC is **discussing the idea on the** [Monty Researcher/Developer Forum](https://thousandbrains.discourse.group/)  **or submitting an issue or feature request**.

# Overall Workflow

To contribute a substantial change to Monty, you must first merge the RFC into the Monty repository as a markdown file. At that point, the RFC is `active` and may be implemented with the goal of eventual inclusion into Monty.

1. **Fork the Monty repository** (see the development [Getting Started](../how-to-use-monty/getting-started.md) guide)
2. **Copy** `rfcs/0000_template.md` to `rfcs/0000_my_proposal.md`. (where "my-proposal" is short but descriptive). Don't assign an RFC number yet. The file will be renamed accordingly if the RFC is accepted.
3. **Fill in the RFC**. Carefully consider the details. RFCs that do not present convincing motivation, demonstrate a lack of understanding of the design's impact or do not fully explore the drawbacks or alternatives tend to be poorly received.
4. **Submit an RFC [Pull Request]\(doc:contributing-pull-requests**). As a pull request, the RFC will receive design feedback from the broader community, and you should be prepared to revise it in response.
5. Each RFC Pull Request will be triaged, given an `rfc:proposal` label, and assigned to a Maintainer who will serve as your primary point of contact for the RFC.
6. **Build consensus and integrate feedback**. RFCs with broad support are much more likely to make progress than those that don't receive any comments. Contact the RFC Pull Request assignee for help identifying stakeholders and obstacles.
7. In due course, one of the Maintainers will propose a "**motion for final comment period (FCP)**" along with the _disposition_ for the RFC (merge or close).
   - This step is taken when enough of the tradeoffs have been discussed so that the Maintainers can decide. This does not require consensus amongst all participants in the RFC thread (which is usually impossible). However, the argument supporting the disposition of the RFC needs to have already been clearly articulated, and there should not be a strong consensus _against_ that position. Maintainers use their best judgment in taking this step, and the FCP itself ensures there is ample time and notification for stakeholders to push back if it is made prematurely.
   - For RFCs with lengthy discussions, the motion for FCP is usually preceded by a _summary comment_ that attempts to outline the current state of the discussion and major tradeoffs/points of disagreement.
8. The FCP lasts **until all Maintainers approve or abstain from the disposition**. This way, all stakeholders can lodge any final objections before reaching a decision.
9. Once the FCP elapses, the **RFC is either merged or closed.** If substantial new arguments or ideas are raised, the FCP is canceled, and the RFC goes back into development mode.
   > [!NOTE]
   > Maintainers
   >
   > 1. Prior to merging the Pull Request, make one last commit:
   >    - Assign the next available sequential number to the RFC
   >    - Rename Pull Request to include the RFC number, e.g., RFC 4 Action Object
   >    - Rename the `rfcs/0000_my_proposal.md` accordingly.
   >    - Update asset folder and any links to assets in that folder if present, like `rfcs/0000_my_proposal/`
   >    - Provide the link to the RFC Pull Request in the `RFC PR` field at the top of the RFC text
   > 2. Merge the Pull Request. The commit message should consist of the `rfc:` prefix, RFC number, title, and pull request number, e.g., `rfc: RFC 3 No Three Day Wait (#366)`
   > 3. Create an Issue in the project that tracks implementation of the merged RFC to fulfill the "Every accepted RFC has an associated issue, tracking its implementation in the Monty repository" requirement. Issue title should be RFC number, title, and the word "Implementation", e.g., `RFC 4 Action Object Implementation`.

# The RFC Life-Cycle

Once an RFC becomes `active` then contributors may implement it and submit the implementation as a pull request to the Monty repository. Being `active` does not guarantee the feature will ultimately be merged. It does mean that in principle, all the major stakeholders have agreed to the feature and are amenable to merging it.

Furthermore, the fact that a given RFC has been accepted and is `active` implies nothing about what priority is assigned to its implementation, nor does it imply anything about whether a Maintainer has been assigned the task of implementing it. While it is not _necessary_ that the author of the RFC also write the implementation, it is by far the most effective way to see an RFC through to completion: **Authors should not expect that others will take on responsibility for implementing their accepted RFC.**

Modifications to `active` RFCs can be done in follow-up Pull Requests. We strive to write each RFC in a manner that will reflect the final design of the feature. Still, the nature of the process means that we cannot expect every merged RFC to reflect what the end result will be at the time of the next major release; therefore, we try to **keep each RFC document somewhat in sync with how the feature is actually being implemented**, tracking such changes via follow-up pull requests to the document.

In general,** once accepted, RFCs should not be substantially changed**. Only very minor changes should be submitted as amendments. More substantial changes should be new RFCs, with a note added to the original RFC. Exactly what counts as a very minor change is up to the Maintainers to decide.

# Reviewing RFCs

While the RFC pull request is open, **Maintainers may schedule meetings with the author and relevant stakeholders** to discuss the issues in greater detail. A summary from each meeting will be posted back to the RFC pull request.

**Maintainers make final decisions about RFCs after the benefits and drawbacks are well understood.** These decisions can be made at anytime and Maintainers will regularly make them. When a decision is made, the RFC pull request will be merged or closed. In either case, if the reasoning from the thread discussion is unclear, Maintainers will add a comment describing the rationale for the decision.

# Implementing an RFC

Some accepted RFCs represent vital features that need to be implemented right away. Other accepted RFCs can represent features that can wait until someone feels like doing the work. **Every accepted RFC has an associated issue, tracking its implementation in the Monty repository.**

The author of an RFC is not obligated to implement it. Anyone, including the author, is welcome to submit an implementation for review after the RFC has been accepted.

If you are interested in working on the implementation of an `active` RFC, but cannot determine if someone else is already working on it; feel free to ask (e.g., by leaving a comment on the associated issue).