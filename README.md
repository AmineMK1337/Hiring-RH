# Hirewall

Hirewall provides secure AI agents for HR: the agent executes, the human decides. AI agents screen resumes, summarize the recruiter inbox, draft replies, propose interview slots and answer employee HR questions, while a security layer defends against prompt injection, data poisoning and malicious uploads, and every outgoing action waits in an approval queue until a human approves it.

## Folder overview

| Folder | Contents |
|---|---|
| `docs/` | Architecture, API contract, data model, security, demo script, credits |
| `data/` | Fictional demo data (resumes, job offers, emails, calendar, HR documents) and attack samples |
| `backend/gateway/` | HTTP API: routers, schemas, auth, middleware |
| `backend/orchestrator/` | Agent workflow graph, state, routing, approval pauses, checkpoints |
| `backend/services/` | Agent services: screener, summarizer, drafter, scheduler, employee_qa |
| `backend/security/` | Input/output defenses and the red-team attack suite |
| `backend/core/` | Shared building blocks: LLM client, DB, approval queue, executor, audit, integrations, logging |
| `backend/tests/` | Integration and end-to-end tests |
| `frontend/` | React (Vite) web app |
| `scripts/` | Data generation, seeding, attack suite runner, demo reset |
| `deploy/` | Deployment assets (Neo4j init) |
| `.github/` | PR template, CODEOWNERS, CI workflow |

## Team rules

1. `docs/api-contract.md` is the source of truth, written before code.
2. Branches are named `feat/<name>-<thing>` and merged into `main` by PR only.
3. No secrets are committed; only `.env.example`.
4. Fictional data only, no real personal data.
5. Credit third-party code in [docs/CREDITS.md](docs/CREDITS.md).
6. Services never import from other services, only from `core/` and `security/`.
7. Only `core/executor` may send or record anything, and only after human approval.

## Ownership

| Folder | Owner |
|---|---|
| `docs/` | TBD |
| `data/` | TBD |
| `backend/gateway/` | TBD |
| `backend/orchestrator/` | TBD |
| `backend/services/screener/` | TBD |
| `backend/services/summarizer/` | TBD |
| `backend/services/drafter/` | TBD |
| `backend/services/scheduler/` | TBD |
| `backend/services/employee_qa/` | TBD |
| `backend/security/` | TBD |
| `backend/core/` | TBD |
| `frontend/` | TBD |
| `scripts/` | TBD |
| `deploy/` | TBD |

## Credits

See [docs/CREDITS.md](docs/CREDITS.md).
