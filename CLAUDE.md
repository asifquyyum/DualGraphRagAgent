# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic Graph RAG is a two-phase system for quantitative finance research that combines knowledge graph retrieval with epistemic reasoning. It tracks beliefs and uncertainties during the query loop, deciding when to explore more, simulate scenarios, or synthesize a final answer.

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_world_state.py -v

# Run a single test
python -m pytest tests/test_schema.py::TestBelief::test_add_supporting_evidence -v

# Lint with ruff
ruff check src/

# Type check
mypy src/

# Ingest documents into the knowledge graph
python -m src.main ingest ./data/

# Query the system
python -m src.main query "What happens to my SPX straddle when VIX spikes?"

# Query with verbose epistemic trace
python -m src.main query -v "What is variance risk premium?"

# Show system status
python -m src.main status

# Display ontology schema
python -m src.main schema

# Use Neo4j instead of NetworkX (requires running Neo4j)
python -m src.main ingest ./data/ --use-neo4j
python -m src.main query "..." --use-neo4j
```

## Architecture

### LangGraph Workflow State Machine

The query loop is orchestrated by `src/graph/langgraph_workflow.py` as a state machine:

```
[semantic_alignment] → [retrieval] → [update_world_model] → [check_sufficiency]
                                              ↑                      │
                                              │         ┌────────────┼────────────┐
                                              │         ↓            ↓            ↓
                                              └── [graph_traversal] [simulate] [synthesize] → END
```

**Routing Logic** (`WorldState.get_routing_decision()`):
- `has_sufficient_evidence()` or max iterations → `synthesize`
- `has_counterfactual_uncertainty()` → `simulate_scenario`
- Unresolved uncertainties or remaining questions → `graph_traversal`

### Epistemic State Tracking

`src/schema/world_state.py` maintains the system's beliefs and uncertainties:

- **Belief**: A proposition with status (HYPOTHETICAL, INFERRED, CONFIRMED, CONTRADICTED) and confidence (0-1)
- **Uncertainty**: A gap in knowledge with type (MISSING_DATA, AMBIGUOUS, COUNTERFACTUAL, CONFLICTING) and priority
- **WorldState**: Aggregates beliefs, uncertainties, explored nodes, and answer completeness

The `WorldModelAgent` updates the world state after each retrieval/traversal, and `SufficiencyChecker` determines when to stop.

### Agent Responsibilities

| Agent | File | Role |
|-------|------|------|
| SemanticAligner | `src/agents/semantic_aligner.py` | Maps user query to ontology terms, generates sub-questions |
| CypherAgent | `src/agents/cypher_agent.py` | Generates and executes Cypher queries (self-correcting, max 5 retries) |
| TraversalAgent | `src/agents/traversal_agent.py` | Explores graph neighbors to resolve uncertainties |
| WorldModelAgent | `src/agents/world_model_agent.py` | Updates beliefs/uncertainties based on new evidence |
| SufficiencyChecker | `src/agents/sufficiency_checker.py` | Decides if enough info exists to answer |
| SimulationAgent | `src/agents/simulation_agent.py` | Runs counterfactual "what if" scenarios |
| SynthesisAgent | `src/agents/synthesis_agent.py` | Generates final answer with citations and confidence |

### Graph Storage Abstraction

`src/graph/graph_interface.py` defines `GraphStore` ABC with two implementations:
- `NetworkXStore`: In-memory for prototyping (default)
- `Neo4jStore`: Production store (requires Neo4j connection)

Both are swappable via `--use-neo4j` flag.

### Document Ingestion Pipeline

1. `DocumentChunker` (`src/ingestion/chunker.py`): Splits PDFs into chunks using LlamaIndex
2. `ExtractionAgent` (`src/ingestion/extraction_agent.py`): Extracts entities/relationships using LLM with ontology guidance
3. Validation: Filters extracted items to match ontology types
4. Bulk load into graph store

### Domain Ontology

`src/schema/ontology.py` defines the quant finance knowledge schema:

**Node Types**: Instrument, Strategy, MarketCondition, RiskFactor, Event, Concept

**Edge Types**: TRADES, HEDGES, PERFORMS_IN, AFFECTED_BY, RELATES_TO, DEFINED_AS, CAUSES, INDICATES

The ontology also maps domain slang (e.g., "vol" → "volatility", "straddle" → "long_straddle").

### LLM Provider

`src/llm/gemini_provider.py` wraps the `google.genai` SDK with:
- `generate()`: Free-form text generation
- `generate_structured()`: JSON output validated against Pydantic models (falls back to text+parsing if schema unsupported)
- `generate_cypher()`: Cypher query generation with schema context

## Configuration

Environment variables (in `.env`):
- `GEMINI_API_KEY`: Required for LLM calls
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`: Optional for Neo4j backend
- `MAX_ITERATIONS`: Query loop limit (default 10)
- `SUFFICIENCY_THRESHOLD`: Completeness threshold (default 0.8)

## Key Pydantic Models

- `Entity` / `Relationship`: Graph nodes and edges with extraction confidence
- `Belief` / `Uncertainty`: Epistemic state items
- `WorldState`: Full epistemic context for a query
- `AlignedQuery`: Semantic alignment result with mapped terms
- `SynthesizedAnswer`: Final answer with confidence, citations, and follow-ups
