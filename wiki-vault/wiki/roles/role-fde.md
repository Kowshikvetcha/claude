---
title: Forward Deployed Engineer
type: role
domain: meta
roles: [fde]
updated: 2026-09-13
---

# Forward Deployed Engineer

## What this role actually does
Forward Deployed Engineer (FDE) is a distinct hybrid, not a rebranded ML Engineer or AI Engineer role: you sit embedded with a client, own rapid prototyping and implementation of an AI/data product against that client's specific systems and constraints, and are judged as much on communication and delivery speed as on technical depth. The role is popularized by companies like Palantir and is now being adopted by AI-first startups selling into enterprises — the pitch is "we build the last mile of your AI product with you, on your data, in your environment."

In India, this role is genuinely less standardized than the others in this vault — expect real variation in what "FDE" means between companies. At some companies it is close to a solutions architect who also codes; at others it's closer to a generalist AI engineer who happens to travel to (or call into) client sites. The common thread is **breadth over depth**: you'll touch RAG, agents, data integration, and basic ML across many client verticals rather than mastering one system deeply, and you'll be expected to scope a client's problem, prototype fast (often in days, not sprints), and communicate tradeoffs to non-technical client stakeholders — sometimes technical ones — under real deadline pressure.

Because the client's data, infra, and constraints change every engagement, a large part of the job is **rapid, defensible technical judgment**: picking a workable architecture quickly with incomplete information, being honest about what won't scale, and iterating live based on client feedback rather than working from a fixed spec. This is closer to consulting-plus-engineering than to a typical product engineering role, and it's a poor fit for someone who wants deep ownership of one system over a long horizon.

## Typical interview loop
This is the **most variable** loop of any role in this vault — company-to-company variation is large, and India-specific patterns are still forming. What shows up most often:
1. **Recruiter screen** — often explicitly probes for client-facing comfort and past ambiguous-scope work, not just technical background.
2. **Live build or take-home** — build a working prototype (often a small RAG or data-integration demo) in a short window, sometimes live with the interviewer watching and changing requirements mid-exercise. This is far more common here than in the other five roles.
3. **Technical breadth round** — rapid-fire across RAG, agents, data pipelines, and basic system design; testing whether you can reason quickly across domains rather than deeply in one.
4. **Client scenario / case round** — given an ambiguous, realistic client problem, asked to scope it, propose an approach, and defend tradeoffs under pushback — evaluated on communication as much as correctness.
5. **Behavioral / culture round** — heavy emphasis on handling ambiguity, client pushback, and working independently with limited support; often the deciding round.

Some companies fold rounds 2–3 into a single half-day or full-day onsite build session. Expect less standardization here than anywhere else — confirm the format directly with the recruiter rather than assuming.

## Domain weighting
| Domain | Weight | Why |
|---|---|---|
| [[moc-behavioral]] | Very high | Client-facing communication, handling pushback, and comfort with ambiguity are as heavily weighted as technical skill — unusual for an engineering role. |
| [[moc-nlp-llm]] | High | Most FDE engagements in the current market involve LLM-based products; baseline fluency is assumed. |
| [[moc-rag]] | High | The most common building block for client prototypes — document Q&A, search, and assistants. |
| [[moc-agents]] | High | Increasingly the ask for client automation workflows; breadth matters more than deep multi-agent theory. |
| [[moc-data-engineering]] | High | Client data is always messy and in an unfamiliar system; fast, pragmatic integration skill is core. |
| [[moc-system-design]] | Medium | Enough to propose a workable architecture quickly and be honest about its limits, not a deep distributed-systems bar. |
| [[moc-programming]] | Medium | Strong general coding and rapid-prototyping ability; DSA depth matters less than build speed and correctness. |
| [[moc-classical-ml]] | Low-medium | Occasionally needed for a client's specific problem; not the centre of the role. |
| [[moc-mlops]] | Low | Prototypes usually hand off to a client or internal team for hardening; deep production-ops ownership is less common here. |
| [[moc-sql]] | Low-medium | Frequently needed for quick client data exploration. |

## JD vocabulary
- **"Forward deployed" / "embedded engineer"** → you work at or with the client, often on-site or in frequent live sessions, not purely remote in your own team's backlog.
- **"Rapid prototyping" / "0 to 1 in weeks"** → speed is a first-class requirement; expect to be evaluated on how fast you produce something demoable, not just on eventual polish.
- **"Client-facing"** → you will present technical work directly to client stakeholders, including ones who will push back or change scope mid-conversation; this is tested directly in interviews.
- **"Generalist" / "full-stack AI"** → breadth across RAG, agents, data integration, and basic backend work is explicitly wanted; a narrow specialist profile is a weaker fit for this specific title.
- **"Ambiguity" / "0 to 1 problems"** → a near-universal phrase in FDE JDs; expect scenario questions with deliberately incomplete information.
- **"Own the outcome, not just the build"** → signals that you're expected to care about whether the client's actual problem got solved, not just whether the code shipped — probed in behavioral rounds.

## Readiness checklist
- [ ] Can scope an ambiguous client problem out loud in under 5 minutes, naming the 2–3 questions you'd ask before building anything.
- [ ] Can build a working RAG prototype quickly (ingestion → chunking → retrieval → generation) using whatever tools are on hand.
- [ ] Can design and articulate a basic agentic workflow for a client automation problem, explaining tradeoffs simply to a non-technical audience.
- [ ] Can integrate with messy, unfamiliar client data (inconsistent schemas, missing docs) and describe your triage approach.
- [ ] Can propose an architecture under real time pressure and explicitly state what it will NOT handle at scale.
- [ ] Can defend a technical tradeoff against pushback from a client stakeholder without becoming defensive or over-promising.
- [ ] Can explain, in plain language with no jargon, what a RAG system or an agent actually does and why it might fail.
- [ ] Can describe how you'd hand off a prototype to a client's or company's own engineering team for hardening.
- [ ] Can tell a clear story (STAR format) about a time scope changed on you mid-project and how you adapted.
- [ ] Can reason across at least two very different problem domains in the same conversation (e.g. a data pipeline question, then an agent design question) without losing coherence.
- [ ] Comfortable being watched/evaluated while building live, including narrating your reasoning as you go.
- [ ] Can explain when you'd say no to a client ask because it's the wrong solution, and how you'd say it.

## The night-before-list
- [[moc-behavioral]]
- [[star-method]]
- [[stakeholder-and-conflict-questions]]
- [[project-narrative-construction]]
- [[handling-failure-questions]]
- [[rag-overview]]
- [[chunking-strategies]]
- [[hybrid-search-bm25-vector]]
- [[rag-failure-modes]]
- [[agent-fundamentals]]
- [[tool-calling-and-function-schemas]]
- [[agent-guardrails-and-safety]]
- [[data-pipeline-fundamentals]]
- [[data-quality-and-validation]]
- [[ml-system-design-framework]]
- [[llm-system-design-framework]]
- [[questions-to-ask-interviewers]]
- [[interview-day-playbook]]

## Related
- [[moc-behavioral]]
- [[moc-rag]]
- [[moc-agents]]
- [[moc-data-engineering]]
- [[plan-12-week]]
- [[plan-7-day-sprint]]
