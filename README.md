# Agentic Graph RAG

A two-phase agentic Graph RAG system with dynamic epistemic reasoning for quantitative finance research.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Ingest documents
python -m src.main ingest ./data/

# Query the system
python -m src.main query "What happens to my SPX straddle when VIX spikes?"
```

## Configuration

Copy `.env.example` to `.env` and set:
- `GEMINI_API_KEY`: Your Gemini API key
- `NEO4J_URI`: Neo4j connection URI (optional)
