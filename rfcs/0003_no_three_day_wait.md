- Start Date: 2024-09-04
- RFC PR: Unavailable

NOTE: While this RFC process is document-based, you are encouraged to also include visual media to help convey your ideas.

# Summary
[summary]: #summary

Removes the three day wait period at the end of the RFC process. Instead, asks for Maintainers to approve, abstain, or object to an RFC Pull Request as part of the Final Comment Period.

# Motivation
[motivation]: #motivation

So far, with the three RFCs that have entered the Final Comment Period, it seems that the three day wait built into the Final Comment Period is extraneous. Maintainers either actively participate in the RFC and approve it, or when asked to sign off on the Final Comment Period, they review and approve the RFC to be merged. Extra three days of waiting seems unnecessary with our current setup.

# Guide-level explanation
[guide-level-explanation]: #guide-level-explanation

## Current RFC Overall Workflow

The current flow on a PR is that a Maintainer motions for FCP. Other Maintainers review the PR and approve it or ask for changes. Once all Maintainers approve, then FCP _begins_. FCP then lasts for three days. Then the PR is merged.

### Overall Workflow

To contribute a substantial change to Monty, you must first merge the RFC into the Monty repository as a markdown file. At that point, the RFC is `active` and may be implemented with the goal of eventual inclusion into Monty.

1. **Fork the Monty repository** (see the development [Getting Started](doc:development-getting-started) guide)
2. **Copy** `rfcs/0000_template.md` to `rfcs/text/0000_my_proposal.md`. (where "my-proposal" is short but descriptive). Don't assign an RFC number yet. The file will be renamed accordingly if the RFC is accepted.
3. **Fill in the RFC**. Carefully consider the details. RFCs that do not present convincing motivation, demonstrate a lack of understanding of the design's impact or do not fully explore the drawbacks or alternatives tend to be poorly received.
4. **Submit an RFC [Pull Request]\(doc:contributing-pull-requests**). As a pull request, the RFC will receive design feedback from the broader community, and you should be prepared to revise it in response.
5. Each RFC Pull Request will be triaged, given an `rfc:proposal` label, and assigned to a Maintainer who will serve as your primary point of contact for the RFC.
6. **Build consensus and integrate feedback**. RFCs with broad support are much more likely to make progress than those that don't receive any comments. Contact the RFC Pull Request assignee for help identifying stakeholders and obstacles.
7. In due course, one of the Maintainers will propose a "**motion for final comment period (FCP)**" along with the _disposition_ for the RFC (merge or close).
   - This step is taken when enough of the tradeoffs have been discussed so that the Maintainers can decide. This does not require consensus amongst all participants in the RFC thread (which is usually impossible). However, the argument supporting the disposition of the RFC needs to have already been clearly articulated, and there should not be a strong consensus _against_ that position. Maintainers use their best judgment in taking this step, and the FCP itself ensures there is ample time and notification for stakeholders to push back if it is made prematurely.
   - For RFCs with lengthy discussions, the motion for FCP is usually preceded by a _summary comment_ that attempts to outline the current state of the discussion and major tradeoffs/points of disagreement.
   - Before actually entering the FCP, _all_ Maintainers must sign off. This is often when many Maintainers first review the RFC in full depth.
8. The FCP lasts **three working days**. This way, all stakeholders can lodge any final objections before reaching a decision.
9. Once the FCP elapses, the **RFC is either merged or closed.** If substantial new arguments or ideas are raised, the FCP is canceled, and the RFC goes back into development mode.

## Suggested RFC Overall Workflow

### Changes

- Remove last substep from step 7. "Before actually entering the FCP, ...".
- Updated step 8 so that the FCP now lasts until all Maintainers approve or abstain from the disposition.

The combined effect of the changes is that now, in an RFC PR, a Maintainer will motion for FCP. Maintainers will approve or abstain. PR is merged. No waiting period that doesn't seem to be used anyway.

In effect, this redefines the meaning of the Final Comment Period to start when a Maintainer calls for one, and to end when all Maintainers acted, with no built in three days of waiting.

### Overall workflow

To contribute a substantial change to Monty, you must first merge the RFC into the Monty repository as a markdown file. At that point, the RFC is `active` and may be implemented with the goal of eventual inclusion into Monty.

1. **Fork the Monty repository** (see the development [Getting Started](doc:development-getting-started) guide)
2. **Copy** `rfcs/0000_template.md` to `rfcs/text/0000_my_proposal.md`. (where "my-proposal" is short but descriptive). Don't assign an RFC number yet. The file will be renamed accordingly if the RFC is accepted.
3. **Fill in the RFC**. Carefully consider the details. RFCs that do not present convincing motivation, demonstrate a lack of understanding of the design's impact or do not fully explore the drawbacks or alternatives tend to be poorly received.
4. **Submit an RFC [Pull Request]\(doc:contributing-pull-requests**). As a pull request, the RFC will receive design feedback from the broader community, and you should be prepared to revise it in response.
5. Each RFC Pull Request will be triaged, given an `rfc:proposal` label, and assigned to a Maintainer who will serve as your primary point of contact for the RFC.
6. **Build consensus and integrate feedback**. RFCs with broad support are much more likely to make progress than those that don't receive any comments. Contact the RFC Pull Request assignee for help identifying stakeholders and obstacles.
7. In due course, one of the Maintainers will propose a "**motion for final comment period (FCP)**" along with the _disposition_ for the RFC (merge or close).
   - This step is taken when enough of the tradeoffs have been discussed so that the Maintainers can decide. This does not require consensus amongst all participants in the RFC thread (which is usually impossible). However, the argument supporting the disposition of the RFC needs to have already been clearly articulated, and there should not be a strong consensus _against_ that position. Maintainers use their best judgment in taking this step, and the FCP itself ensures there is ample time and notification for stakeholders to push back if it is made prematurely.
   - For RFCs with lengthy discussions, the motion for FCP is usually preceded by a _summary comment_ that attempts to outline the current state of the discussion and major tradeoffs/points of disagreement.
8. The FCP lasts **until all Maintainers approve or abstain from the disposition.** This way, all stakeholders can lodge any final objections before reaching a decision.
9. Once the FCP elapses, the **RFC is either merged or closed.** If substantial new arguments or ideas are raised, the FCP is canceled, and the RFC goes back into development mode.

# Reference-level explanation
[reference-level-explanation]: #reference-level-explanation

No additional detail from Guide-level explanation.

# Drawbacks
[drawbacks]: #drawbacks

The Final Comment Period exists for a reason in other RFC processes, but it doesn't seem we have a use case for it yet. So, we lose out on the use case for now.

# Rationale and alternatives
[rationale-and-alternatives]: #rationale-and-alternatives

Alternative is current state.

# Prior art and references
[prior-art-and-references]: #prior-art-and-references

Omitted.

# Unresolved questions
[unresolved-questions]: #unresolved-questions

None at this time.

# Future possibilities
[future-possibilities]: #future-possibilities

If we rediscover a compelling use case for the Final Comment Period once we have external Maintainers, we can reintroduce it.