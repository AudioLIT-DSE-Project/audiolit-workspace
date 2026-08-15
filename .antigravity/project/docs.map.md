# Documentation Map — AudioLIT

> Which document owns which subject. **Read the real document.** Nothing in
> `project/` restates what these cover.

## Authority order

`docs/SAD.md` and `docs/SRS.md` win over Linear issue bodies, which win over
assumptions. `docs/README.md` errata supersede stale text in either.

| Document | Owns | Read when |
|---|---|---|
| `docs/README.md` | conventions, authority order, **errata**, branch model, scope discipline, FR quick index | **first, always** |
| `docs/SAD.md` | architecture of record (§5.1/§5.2 layers, §6.1, §8.1/§8.2, §11.1–§11.3) | Phase 3, any design question |
| `docs/SRS.md` | committed requirements, FR/NFR text, scope | Phase 1, any scope question |
| `.antigravity/project/ISSUE_PLAN.md` | dependency-ordered issue index, tiers, status | Phase 1, picking work |
| `.antigravity/project/TEAM_AGENT_WORKFLOW.md` | how a session should run here, when to stop and ask | onboarding a session |
| `.antigravity/project/PR_REVIEW_CHECKLIST.md` | what a reviewer checks before merging | Phase 5, Phase 6 |
| `docs/rq_fanout_pattern.md` | the RQ fan-out orchestration pattern | orchestration work |
| Linear (team `LIT`) | **scope of record per issue** — the Tier-C stamp header | Phase 1, every ticket |
| `README.md` | monorepo layout, local run instructions | onboarding |

## Requirements

`docs/README.md` already carries a committed-FR pointer index (FR1–FR4, FR6–FR12,
FR15–FR17). This system does not duplicate it. **There is no FR5, FR13 or FR14**
in the reconciled SRS — FR5 was demoted to non-committed stretch.

## Forbidden for generated content

Setup may not restate: architecture · requirement text · repository layout ·
branch model · scope rules · the FR index · PR review steps · errata. All are
owned above. Anything generated here covers only what none of them do.
