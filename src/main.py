"""CLI entry point for the Agentic Graph RAG system.

Provides commands for document ingestion, querying, and system management.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# Load environment variables
load_dotenv()

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Agentic Graph RAG - Epistemic reasoning for quantitative finance."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--use-neo4j", is_flag=True, help="Use Neo4j instead of NetworkX")
@click.option("--generate-schema", is_flag=True, help="Auto-generate schema from documents")
def ingest(path: str, use_neo4j: bool, generate_schema: bool):
    """Ingest documents from PATH into the knowledge graph."""
    from src.graph.networkx_store import NetworkXStore
    from src.ingestion.chunker import DocumentChunker
    from src.ingestion.extraction_agent import ExtractionAgent
    from src.ingestion.meta_agent import MetaAgent
    from src.llm.gemini_provider import GeminiProvider
    from src.schema.ontology import create_quant_finance_ontology

    console.print(Panel("Starting document ingestion", style="bold blue"))

    # Initialize components
    try:
        llm = GeminiProvider()
        console.print("[green]✓[/green] LLM provider initialized")
    except ValueError as e:
        console.print(f"[red]✗[/red] LLM initialization failed: {e}")
        console.print("Set GEMINI_API_KEY in .env file")
        return

    # Initialize graph store
    if use_neo4j:
        try:
            from src.graph.neo4j_store import Neo4jStore
            store = Neo4jStore()
            console.print("[green]✓[/green] Neo4j store initialized")
        except Exception as e:
            console.print(f"[red]✗[/red] Neo4j connection failed: {e}")
            console.print("Falling back to NetworkX")
            store = NetworkXStore()
    else:
        store = NetworkXStore()
        console.print("[green]✓[/green] NetworkX store initialized")

    # Initialize ontology
    ontology = create_quant_finance_ontology()

    if generate_schema:
        console.print("Analyzing documents to generate schema...")
        chunker = DocumentChunker()
        meta_agent = MetaAgent(llm)

        # Get sample chunks
        path_obj = Path(path)
        if path_obj.is_dir():
            chunks = chunker.chunk_directory(path_obj)
        else:
            chunks = chunker.chunk_file(path_obj)

        if chunks:
            sample_texts = [c.text for c in chunks[:10]]
            proposal = meta_agent.analyze_documents(sample_texts, ontology)
            ontology = meta_agent.merge_ontologies(
                ontology, meta_agent.proposal_to_ontology(proposal)
            )
            console.print(f"[green]✓[/green] Schema extended with {len(proposal.node_types)} new types")

    store.initialize(ontology)

    # Chunk documents
    console.print("Chunking documents...")
    chunker = DocumentChunker()
    path_obj = Path(path)

    if path_obj.is_dir():
        chunks = chunker.chunk_directory(path_obj)
    else:
        chunks = chunker.chunk_file(path_obj)

    console.print(f"[green]✓[/green] Created {len(chunks)} chunks")

    # Extract entities
    console.print("Extracting entities and relationships...")
    extraction_agent = ExtractionAgent(llm, ontology)

    with Progress() as progress:
        task = progress.add_task("Extracting...", total=len(chunks))
        entities, relationships = extraction_agent.extract_from_chunks(chunks)
        progress.update(task, completed=len(chunks))

    console.print(f"[green]✓[/green] Extracted {len(entities)} entities, {len(relationships)} relationships")

    # Load into graph
    console.print("Loading into graph...")
    store.bulk_add_entities(entities)
    store.bulk_add_relationships(relationships)

    # Show statistics
    stats = store.get_statistics()
    table = Table(title="Graph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total Entities", str(stats["num_entities"]))
    table.add_row("Total Relationships", str(stats["num_relationships"]))
    for node_type, count in stats.get("node_types", {}).items():
        table.add_row(f"  {node_type}", str(count))

    console.print(table)
    console.print(Panel("[green]Ingestion complete![/green]", style="bold green"))


@cli.command()
@click.argument("question")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed epistemic trace")
@click.option("--use-neo4j", is_flag=True, help="Use Neo4j instead of NetworkX")
@click.option("--max-iterations", default=10, help="Maximum query loop iterations")
def query(question: str, verbose: bool, use_neo4j: bool, max_iterations: int):
    """Query the knowledge graph with QUESTION."""
    from src.graph.langgraph_workflow import GraphRAGWorkflow
    from src.graph.networkx_store import NetworkXStore
    from src.llm.gemini_provider import GeminiProvider
    from src.schema.ontology import create_quant_finance_ontology

    console.print(Panel(f"Query: {question}", style="bold blue"))

    # Initialize components
    try:
        llm = GeminiProvider()
    except ValueError as e:
        console.print(f"[red]✗[/red] LLM initialization failed: {e}")
        return

    # Initialize graph store
    if use_neo4j:
        try:
            from src.graph.neo4j_store import Neo4jStore
            store = Neo4jStore()
        except Exception:
            store = NetworkXStore()
    else:
        store = NetworkXStore()

    ontology = create_quant_finance_ontology()
    store.initialize(ontology)

    # Create workflow
    workflow = GraphRAGWorkflow(llm, ontology, store, max_iterations=max_iterations)

    # Run query
    with console.status("Processing query..."):
        if verbose:
            result = workflow.run_verbose(question)
            world_state = result.get("world_state")
            answer = result.get("final_answer")

            if world_state:
                # Show epistemic trace
                console.print("\n[bold]Epistemic Trace:[/bold]")
                console.print(f"Iterations: {world_state.iteration_count}/{world_state.max_iterations}")
                console.print(f"Completeness: {world_state.answer_completeness:.0%}")
                console.print(f"Explored Nodes: {len(world_state.explored_nodes)}")

                if world_state.beliefs:
                    console.print("\n[bold]Beliefs:[/bold]")
                    for belief in list(world_state.beliefs.values())[:5]:
                        status_color = {
                            "confirmed": "green",
                            "contradicted": "red",
                            "inferred": "yellow",
                            "hypothetical": "dim",
                        }.get(belief.status.value, "white")
                        console.print(
                            f"  [{status_color}]•[/{status_color}] {belief.content} "
                            f"({belief.confidence:.0%})"
                        )

                if world_state.uncertainties:
                    console.print("\n[bold]Remaining Uncertainties:[/bold]")
                    for u in list(world_state.get_unresolved_uncertainties())[:3]:
                        console.print(f"  [yellow]?[/yellow] {u.description}")

        else:
            answer = workflow.run(question)

    # Display answer
    if answer:
        console.print("\n")
        console.print(Panel(answer.answer, title="Answer", style="green"))
        console.print(f"\nConfidence: {answer.confidence:.0%}")

        if answer.citations:
            console.print("\n[bold]Citations:[/bold]")
            for citation in answer.citations[:5]:
                console.print(f"  • {citation.content[:100]}...")

        if answer.uncertainties_noted:
            console.print("\n[bold]Uncertainties:[/bold]")
            for u in answer.uncertainties_noted:
                console.print(f"  ? {u}")

        if answer.follow_up_questions:
            console.print("\n[bold]Follow-up Questions:[/bold]")
            for q in answer.follow_up_questions:
                console.print(f"  → {q}")
    else:
        console.print("[red]Failed to generate answer[/red]")


@cli.command()
@click.option("--use-neo4j", is_flag=True, help="Check Neo4j connection")
def status(use_neo4j: bool):
    """Show system status."""
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    # Check environment
    gemini_key = os.getenv("GEMINI_API_KEY")
    table.add_row("GEMINI_API_KEY", "[green]Set[/green]" if gemini_key else "[red]Missing[/red]")

    # Check Neo4j if requested
    if use_neo4j:
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        table.add_row("NEO4J_URI", neo4j_uri)

        try:
            from src.tools.neo4j_tools import Neo4jTools
            tools = Neo4jTools()
            if tools.verify_connection():
                table.add_row("Neo4j Connection", "[green]Connected[/green]")
            else:
                table.add_row("Neo4j Connection", "[red]Failed[/red]")
            tools.close()
        except Exception as e:
            table.add_row("Neo4j Connection", f"[red]Error: {e}[/red]")

    console.print(table)


@cli.command()
def schema():
    """Display the current ontology schema."""
    from src.schema.ontology import create_quant_finance_ontology

    ontology = create_quant_finance_ontology()

    console.print(Panel(f"Ontology: {ontology.name} v{ontology.version}", style="bold blue"))

    # Node types
    console.print("\n[bold]Node Types:[/bold]")
    for name, node_schema in ontology.node_schemas.items():
        props = ", ".join(p.name for p in node_schema.properties)
        console.print(f"  [cyan]{name}[/cyan]: {node_schema.description}")
        console.print(f"    Properties: {props}")

    # Edge types
    console.print("\n[bold]Relationship Types:[/bold]")
    for name, edge_schema in ontology.edge_schemas.items():
        console.print(f"  [yellow]{name}[/yellow]: {edge_schema.description}")
        console.print(f"    ({edge_schema.source_types}) → ({edge_schema.target_types})")

    # Domain terms
    console.print("\n[bold]Domain Terms:[/bold]")
    for slang, canonical in list(ontology.domain_terms.items())[:10]:
        console.print(f"  {slang} → {canonical}")


if __name__ == "__main__":
    cli()
