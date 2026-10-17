# Enterprise Stateful Agent Engine

> A resilient, stateful operations agent built on **LangGraph**, **Multi-Provider Fallback Routing**, **Agentic Hybrid RAG**, and **Selective Human-in-the-Loop (HITL) Interrupts**.

Designed for mission-critical automation where autonomous agents must execute multi-step reasoning, query live infrastructure telemetry, reference domain manuals from PostgreSQL, and enforce human authorization gates before dispatching high-impact physical actions.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-stateful-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-336791)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [Architecture](#architecture)
- [Core Engineering Capabilities](#core-engineering-capabilities)
- [Repository Structure](#repository-structure)
- [Setup & Execution](#setup--execution)
- [Production Execution Trace](#production-execution-trace)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Roadmap](#roadmap)
- [License](#license)

---

## Why This Exists

Most agent demos are stateless chains that either (a) run fully autonomous — unsafe for anything that touches production — or (b) interrupt on every step — causing operator fatigue. This engine demonstrates the production pattern:

- **Stateful graph execution** with reducer-based memory
- **Selective interrupts** only on mutating, high-impact tools
- **Multi-provider fallback** so a single `429` never kills the run
- **Hybrid retrieval** (dense + BM25 + reranker) exposed as an autonomous tool, not a forced upfront lookup

---

## Architecture

Unlike stateless chains, this engine models agent execution as a **Directed Cyclic Graph (DCG)** where nodes execute discrete tasks, reducers preserve conversation history, and conditional edges route execution dynamically based on tool payload semantics.

```
                          [ User Query ]
                                │
                                ▼
                         ┌──────────────┐
                         │    START     │
                         └──────┬───────┘
                                │
                                ▼
                  ┌────────────────────────────┐
         ┌───────►│         agent_node         │
         │        │  - Groq (gpt-oss-120b)     │
         │        │  - Fallback: Gemini 3.8-F  │
         │        └─────────────┬──────────────┘
         │                      │
         │       [ route_with_selective_interrupt ]
         │                      │
         │      ┌───────────────┼───────────────┐
         │      │               │               │
         │ [No Tools]     [Safe Tools]   [Sensitive Tool]
         │      │               │               │
         │      ▼               │               ▼
         │   ┌─────┐            │    ┌─────────────────────┐
         │   │ END │            │    │ 🚨 human_approval   │
         │   └─────┘            │    │ (Breakpoint Pause)  │
         │                      │    └──────────┬──────────┘
         │                      │               │
         │                      │      [Operator Approval]
         │                      │         ├── REJECT: Inject Feedback ──┐
         │                      │         └── APPROVE: Continue         │
         │                      │                     │                 │
         │                      ▼                     ▼                 │
         │               ┌──────────────────────────────┐               │
         └───────────────┤          tools_node          │◄──────────────┘
                         │ - calculate_feed_consumption │
                         │ - check_equipment_telemetry  │
                         │ - query_farm_knowledge_base  │
                         │ - dispatch_maintenance_ticket│
                         └──────────────────────────────┘
```

---

## Core Engineering Capabilities

### 1. Multi-Provider Fallback Routing

Enterprise agents cannot fail when a single provider suffers API outages, capacity spikes, or rate-limit saturation (`HTTP 429 / 503`).

| Layer | Provider | Model | Role |
|---|---|---|---|
| Primary | Groq | `openai/gpt-oss-120b` | Low-latency reasoning + tool calling |
| Fallback | Google | `gemini-3.8-flash` | Continuity on primary failure |

**Behavior:** If the primary exhausts quota or drops sockets, the runtime intercepts the error without crashing the graph, re-serializes the state payload, and executes the fallback via `.with_fallbacks()` within milliseconds.

---

### 2. Selective Human-in-the-Loop (HITL) Interrupts

Standard agent implementations either run fully autonomous (unsafe for production actions) or interrupt on every single step (severe operator fatigue). This engine interrupts **only** on mutating tools.

**Safe, read-only tools** (execute instantly):

- `calculate_feed_consumption`
- `check_equipment_telemetry`
- `query_farm_knowledge_base`

**Mutating, transactional tools** (require authorization):

- `dispatch_maintenance_ticket`

**Flow:**

1. `route_with_selective_interrupt` detects a sensitive tool call
2. Compiler pauses at `interrupt_before=["human_approval"]`
3. State snapshot saved to checkpointer
4. Control yielded to operator
5. Operator approves (`yes`) → dispatch proceeds
6. Operator rejects (`no`) → rejection feedback injected into context for graceful re-planning

---

### 3. Agentic RAG Integration (PostgreSQL + pgvector)

Instead of forcing every user interaction through an upfront search query, the agent uses **`query_farm_knowledge_base`** as an autonomous, invokable tool.

**Stage 1 — Hybrid Retrieval**

- PostgreSQL with `pgvector`
- 768-dimension HNSW dense vectors
- GIN full-text search tokens
- Fused via **Reciprocal Rank Fusion (RRF)**

**Stage 2 — Reranking**

- `FlashRank` (`ms-marco-TinyBERT-L-2-v2`) running on CPU
- Strips passages below the `0.20` relevance floor

**Tool Chaining Example**

The agent autonomously chains telemetry reads (`PUMP-TX8: 248 hours runtime, low oil`) with documentation lookups (`MAN-PUMP-TX8: 15W-40 industrial lubricant specification`) to construct grounded operational tickets.

---

### 4. Stateful Checkpointing & Session Memory

- State schemas defined via `TypedDict`
- `add_messages` reducer prevents state clobbering across turns
- Backed by `MemorySaver` (swap-in ready for `PostgresSaver`)
- Conversations isolated by `thread_id`
- Agent retains contextual memory across arbitrary follow-up turns **without re-running previous tool calls**

---

## Repository Structure

```
.
├── agent.py            # Baseline ReAct agent loop and node definitions
├── agent_memory.py     # Multi-turn stateful agent with selective HITL interrupts
├── rag_tool.py         # PostgreSQL hybrid RRF + FlashRank tool
├── requirements.txt    # Frozen Python runtime dependencies
├── .env.example        # Environment variable template
└── README.md           # This document
```

| File | Purpose |
|---|---|
| `agent.py` | Baseline ReAct loop — useful as a reference implementation |
| `agent_memory.py` | **Main entry point** — stateful graph with HITL gates |
| `rag_tool.py` | Encapsulated hybrid retrieval tool |
| `requirements.txt` | Pinned dependencies |

---

## Setup & Execution

### 1. Prerequisites

- Python 3.11+
- PostgreSQL with `pgvector` extension (port `5432`)
- Active API keys for **Groq** and **Google Gemini**

### 2. Environment Setup

```bash
git clone https://github.com/0xAnthonyRx/stateful-agent-engine.git
cd stateful-agent-engine
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Populate `.env`:

```ini
DATABASE_URL="postgresql://postgres:postgres@localhost:5432/vectordb"
GEMINI_API_KEY="AIzaSy..."
GROQ_API_KEY="gsk_..."
EMBEDDING_MODEL="gemini/gemini-embedding-001"
```

### 4. Start the Vector Database

```bash
docker start pgvector-db
```

### 5. Launch the Interactive HITL Agent

```bash
python3 agent_memory.py
```

---

## Production Execution Trace

The trace below illustrates the agent handling a compound query: inspecting equipment telemetry, querying PostgreSQL for technical specifications in the background, formulating a work order, and halting at the HITL gate for operator authorization.

```
=== Production HITL Agent Activated (Type 'exit' to quit) ===

You: Check the status of pump PUMP-TX8 and see what lubricant it needs in the manual.

Agent:
Pump Status:
- Device: PUMP-TX8
- Current Status: OPERATIONAL
- Operating Hours: 248 h (approaching 250-hour service interval)
- Oil Level: WARNING - Low

Lubricant Required (per equipment manual):
- Type: 15W-40 industrial lubricant
- Part Number: PUMP-TX8-LUB

Recommended Action:
Refill or replace the oil with specified 15W-40 lubricant before the 250h threshold.
Let me know if you'd like me to create a maintenance work order.

------------------------------------------------------------

You: Yes, please create the emergency maintenance ticket with priority URGENT.

============================================================
🚨 HUMAN-IN-THE-LOOP INTERRUPT GATE (High-Stakes Action)
The agent proposed the following action(s):
  -> Tool: dispatch_maintenance_ticket
     Args: {'equipment_id': 'PUMP-TX8', 'parts_required': '15W-40 industrial lubricant (Part serial: PUMP-TX8-LUB)', 'priority': 'URGENT'}
============================================================
Authorize this execution? (yes/no): yes

[Authorization Granted] Resuming execution...

Agent:
✅ The emergency maintenance ticket has been created and dispatched:

- Ticket ID: TICKET-9921
- Asset: PUMP-TX8
- Priority: URGENT
- Required Part: 15W-40 industrial lubricant (Part serial: PUMP-TX8-LUB)

The central depot crew has been notified and dispatched.
```

---

## Design Decisions & Tradeoffs

| Decision | Rationale | Tradeoff |
|---|---|---|
| **Selective HITL** over full autonomy | Safety-critical physical actions require authorization | Slight latency on mutating tools only |
| **Multi-provider fallback** | Single point of failure is unacceptable in production | Small complexity in error interception |
| **Hybrid RAG as a tool** (not upfront) | Agent decides when retrieval is needed | Requires better tool descriptions |
| **FlashRank on CPU** | No GPU dependency, low cost | Slightly slower than GPU rerankers |
| **`MemorySaver` default** | Fast local dev | Must swap to `PostgresSaver` for true persistence |
| **RRF fusion** | Robust across retrieval modalities | Extra compute vs. single-method |

---

## Roadmap

- [ ] Migrate `MemorySaver` → `PostgresSaver` for durable cross-session state
- [ ] Add OpenTelemetry tracing for node-level latency
- [ ] Expand tool registry (inventory checks, crew availability)
- [ ] Expose as a FastAPI service with streaming responses
- [ ] Add evaluation harness for retrieval quality (nDCG@k)

---

## License

MIT — see [LICENSE](LICENSE) for details.
