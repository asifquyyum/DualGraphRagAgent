"""Gemini LLM provider with structured output support.

Wraps the google.genai SDK for use in the Graph RAG system,
with support for structured JSON outputs via Pydantic models.
"""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel, ValidationError
from pydantic_settings import BaseSettings


class GeminiSettings(BaseSettings):
    """Configuration for Gemini API."""

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    temperature: float = 0.7
    max_output_tokens: int = 8192
    top_p: float = 0.95
    top_k: int = 40

    class Config:
        """Pydantic settings config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields from .env (e.g., NEO4J settings)


T = TypeVar("T", bound=BaseModel)


class GeminiProvider:
    """Provider for Gemini LLM with structured output support.

    Supports both free-form text generation and structured JSON outputs
    validated against Pydantic models.
    """

    def __init__(self, settings: GeminiSettings | None = None) -> None:
        """Initialize the Gemini provider.

        Args:
            settings: Optional settings. If not provided, will load from environment.
        """
        self.settings = settings or GeminiSettings()

        if not self.settings.gemini_api_key:
            self.settings.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

        if not self.settings.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or environment variables."
            )

        # Initialize client with API key
        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model_name = self.settings.gemini_model

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            temperature: Optional temperature override

        Returns:
            Generated text response
        """
        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.settings.temperature,
            max_output_tokens=self.settings.max_output_tokens,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
        )

        if system_instruction:
            config.system_instruction = system_instruction

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )
        return response.text

    def generate_structured(
        self,
        prompt: str,
        output_schema: type[T],
        system_instruction: str | None = None,
        max_retries: int = 3,
    ) -> T:
        """Generate structured output validated against a Pydantic model.

        Args:
            prompt: The user prompt
            output_schema: Pydantic model class for output validation
            system_instruction: Optional system instruction
            max_retries: Number of retries if validation fails

        Returns:
            Instance of output_schema populated with generated data

        Raises:
            ValueError: If unable to generate valid output after retries
        """
        last_error: Exception | None = None
        use_native_schema = True

        for attempt in range(max_retries):
            try:
                if use_native_schema:
                    # Try native structured output first
                    config = types.GenerateContentConfig(
                        temperature=max(0.1, self.settings.temperature - 0.1 * attempt),
                        max_output_tokens=self.settings.max_output_tokens,
                        top_p=self.settings.top_p,
                        top_k=self.settings.top_k,
                        response_mime_type="application/json",
                        response_schema=output_schema,
                    )

                    if system_instruction:
                        config.system_instruction = system_instruction

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )

                    # Try to use parsed response first (auto-parsed by SDK)
                    if hasattr(response, 'parsed') and response.parsed is not None:
                        return response.parsed

                    # Fallback to manual parsing
                    cleaned = response.text.strip()
                else:
                    # Fallback: text generation with JSON schema in prompt
                    schema_json = output_schema.model_json_schema()
                    schema_instruction = f"""
You must respond with valid JSON that conforms to this schema:
{json.dumps(schema_json, indent=2)}

Respond ONLY with the JSON object, no additional text or markdown code blocks.
"""
                    full_instruction = schema_instruction
                    if system_instruction:
                        full_instruction = f"{system_instruction}\n\n{schema_instruction}"

                    config = types.GenerateContentConfig(
                        temperature=max(0.1, self.settings.temperature - 0.1 * attempt),
                        max_output_tokens=self.settings.max_output_tokens,
                        top_p=self.settings.top_p,
                        top_k=self.settings.top_k,
                        system_instruction=full_instruction,
                    )

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    cleaned = response.text.strip()

                # Clean markdown if present
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                cleaned = cleaned.strip()

                data = json.loads(cleaned)
                return output_schema.model_validate(data)

            except json.JSONDecodeError as e:
                last_error = e
                continue
            except ValidationError as e:
                last_error = e
                continue
            except Exception as e:
                # Check if it's a schema limitation error - switch to fallback
                error_str = str(e).lower()
                if "additionalproperties" in error_str or "not supported" in error_str:
                    use_native_schema = False
                    # Don't count this as an attempt, retry with fallback
                    continue
                last_error = e
                continue

        raise ValueError(
            f"Failed to generate valid structured output after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def generate_with_context(
        self,
        query: str,
        context: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate response with context (for RAG-style queries).

        Args:
            query: The user's question
            context: Retrieved context to inform the answer
            system_instruction: Optional system instruction

        Returns:
            Generated response
        """
        prompt = f"""Context:
{context}

Question: {query}

Based on the context provided, answer the question. If the context doesn't contain
enough information to fully answer, indicate what's missing."""

        return self.generate(prompt, system_instruction=system_instruction)

    def generate_cypher(
        self,
        query: str,
        schema: str,
        examples: list[tuple[str, str]] | None = None,
    ) -> str:
        """Generate Cypher query from natural language.

        Args:
            query: Natural language query
            schema: Graph schema description
            examples: Optional list of (question, cypher) examples

        Returns:
            Generated Cypher query
        """
        examples_str = ""
        if examples:
            examples_str = "\n\nExamples:\n"
            for question, cypher in examples:
                examples_str += f"Q: {question}\nCypher: {cypher}\n\n"

        system_instruction = f"""You are a Cypher query expert. Generate Cypher queries for Neo4j
based on the user's natural language question and the provided graph schema.

{schema}
{examples_str}

Respond ONLY with the Cypher query, no explanations. Use MATCH, WHERE, RETURN clauses appropriately.
Handle node labels, relationship types, and property names exactly as shown in the schema."""

        return self.generate(query, system_instruction=system_instruction, temperature=0.1)

    def extract_entities(
        self,
        text: str,
        entity_types: list[str],
        relationship_types: list[str],
    ) -> dict[str, Any]:
        """Extract entities and relationships from text.

        Args:
            text: Text to extract from
            entity_types: List of valid entity types
            relationship_types: List of valid relationship types

        Returns:
            Dictionary with 'entities' and 'relationships' lists
        """
        system_instruction = f"""You are an expert at extracting structured information from text.
Extract entities and relationships based on these schemas:

Entity Types: {', '.join(entity_types)}
Relationship Types: {', '.join(relationship_types)}

For each entity, provide: type, name, and relevant properties.
For each relationship, provide: type, source_name, target_name, and properties.

Respond with valid JSON containing 'entities' and 'relationships' arrays."""

        prompt = f"Extract all entities and relationships from this text:\n\n{text}"

        response = self.generate(prompt, system_instruction=system_instruction, temperature=0.2)

        # Parse JSON response
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        try:
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            return {"entities": [], "relationships": []}

    def critique_and_refine(
        self,
        original: str,
        critique_prompt: str,
        max_iterations: int = 3,
    ) -> str:
        """Iteratively critique and refine a response.

        Args:
            original: Original response to refine
            critique_prompt: Instructions for critique
            max_iterations: Maximum refinement iterations

        Returns:
            Refined response
        """
        current = original

        for _ in range(max_iterations):
            # Generate critique
            critique = self.generate(
                f"Critique this response:\n\n{current}\n\nCritique instructions: {critique_prompt}",
                temperature=0.3,
            )

            # Check if critique suggests improvements needed
            if "no improvements" in critique.lower() or "looks good" in critique.lower():
                break

            # Refine based on critique
            current = self.generate(
                f"Original response:\n{current}\n\nCritique:\n{critique}\n\n"
                "Provide an improved response addressing the critique.",
                temperature=0.5,
            )

        return current
