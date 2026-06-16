#!/usr/bin/env python3
"""
Ollama Model Benchmark Tool

A lightweight tool for measuring LLM performance metrics via Ollama:
- Token processing speed (t/s)
- Model load time
- Prompt evaluation time
- Response generation time

Usage:
    python benchmark.py [-v] [-m MODEL_NAMES...] [-p PROMPTS...]

Example:
    python benchmark.py --verbose --models llama2:13b codellama:34b
"""

import argparse
from typing import List, Dict, Optional
from datetime import datetime

import ollama
from pydantic import BaseModel, Field

from tabulate import tabulate


class Message(BaseModel):
    """Represents a single message in the chat interaction."""
    role: str
    content: str


def nanosec_to_sec(nanosec: int) -> float:
    """Converts nanoseconds to seconds."""
    return nanosec / 1_000_000_000


class OllamaResponse(BaseModel):
    """
    Represents a structured response from the Ollama API.
    Contains performance metrics and message content.
    """
    model: str
    created_at: datetime | None = None
    message: Message
    done: bool
    total_duration: float = Field(default=0.0)
    load_duration: float = Field(default=0.0)
    prompt_eval_count: int = Field(default=0)
    prompt_eval_duration: float = Field(default=0.0)
    eval_count: int = Field(default=0)
    eval_duration: float = Field(default=0.0)

    @classmethod
    def from_chat_response(cls, response) -> 'OllamaResponse':
        """
        Converts an Ollama API response into an OllamaResponse instance.

        Args:
            response: Raw response from Ollama API

        Returns:
            OllamaResponse: Structured response object
        """
        return cls(
            model=response.model,
            message=Message(
                role=response.message.role,
                content=response.message.content
            ),
            done=response.done,
            total_duration=nanosec_to_sec(getattr(response, 'total_duration', 0)),
            load_duration=nanosec_to_sec(getattr(response, 'load_duration', 0)),
            prompt_eval_count=getattr(response, 'prompt_eval_count', 0),
            prompt_eval_duration=nanosec_to_sec(getattr(response, 'prompt_eval_duration', 0)),
            eval_count=getattr(response, 'eval_count', 0),
            eval_duration=nanosec_to_sec(getattr(response, 'eval_duration', 0))
        )

    @property
    def prompt_eval_rate(self) -> float:
        return self.prompt_eval_count / self.prompt_eval_duration if self.prompt_eval_duration > 0 else 0.0

    @property
    def eval_rate(self) -> float:
        return self.eval_count / self.eval_duration if self.eval_duration > 0 else 0.0

    @property
    def total_rate(self) -> float:
        total_secs = self.prompt_eval_duration + self.eval_duration
        return (self.prompt_eval_count + self.eval_count) / total_secs if total_secs > 0 else 0.0

    @classmethod
    def aggregate(cls, responses: List['OllamaResponse']) -> 'OllamaResponse':
        """Aggregates multiple responses into a single summary response."""
        return cls(
            model=responses[0].model,
            created_at=datetime.now(),
            message=Message(
                role="system",
                content=f"Aggregate stats across {len(responses)} runs",
            ),
            done=True,
            total_duration=sum(r.total_duration for r in responses),
            load_duration=sum(r.load_duration for r in responses),
            prompt_eval_count=sum(r.prompt_eval_count for r in responses),
            prompt_eval_duration=sum(r.prompt_eval_duration for r in responses),
            eval_count=sum(r.eval_count for r in responses),
            eval_duration=sum(r.eval_duration for r in responses),
        )


def run_benchmark(
        model_name: str,
        prompt: str,
        verbose: bool
) -> Optional[OllamaResponse]:
    """
    Executes a benchmark run for a specific model and prompt.

    Args:
        model_name: Name of the Ollama model to benchmark
        prompt: Input text to send to the model
        verbose: If True, prints streaming output

    Returns:
        OllamaResponse object containing benchmark results, or None if failed
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        if verbose:
            # For verbose mode, we'll collect the content while streaming
            content = ""
            stream = ollama.chat(
                model=model_name,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if hasattr(chunk.message, 'content'):
                    content += chunk.message.content
                    print(chunk.message.content, end="", flush=True)

            if not content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response. Please check if:")
                print("1. The model is properly loaded")
                print("2. The Ollama server is functioning correctly")
                print("3. Try running 'ollama run {model_name}' in terminal to verify model output")
                return None

            # Make a non-streaming call to get the metrics
            response = ollama.chat(
                model=model_name,
                messages=messages,
            )

            # Check if response has content
            if not hasattr(response.message, 'content') or not response.message.content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response in non-streaming mode")
                return None

            # Create response with collected content and metrics
            return OllamaResponse.from_chat_response(response)
        else:
            # For non-verbose mode, just make a single non-streaming call
            response = ollama.chat(
                model=model_name,
                messages=messages,
            )

            # Check if response has content
            if not hasattr(response.message, 'content') or not response.message.content.strip():
                print(f"\nError: Ollama model {model_name} returned empty response. Please check if:")
                print("1. The model is properly loaded")
                print("2. The Ollama server is functioning correctly")
                print("3. Try running 'ollama run {model_name}' in terminal to verify model output")
                return None

            return OllamaResponse.from_chat_response(response)

    except Exception as e:
        print(f"Error benchmarking {model_name}: {str(e)}")
        return None


def inference_stats(model_response: OllamaResponse) -> None:
    """
    Calculates and prints detailed inference statistics for a model response.

    Args:
        model_response: OllamaResponse containing benchmark metrics
    """
    print(
        f"""
----------------------------------------------------
        Model: {model_response.model}
        Performance Metrics:
            Prompt Processing:  {model_response.prompt_eval_rate:.2f} tokens/sec
            Generation Speed:   {model_response.eval_rate:.2f} tokens/sec
            Combined Speed:     {model_response.total_rate:.2f} tokens/sec

        Workload Stats:
            Input Tokens:       {model_response.prompt_eval_count}
            Generated Tokens:   {model_response.eval_count}
            Model Load Time:    {model_response.load_duration:.2f}s
            Processing Time:    {model_response.prompt_eval_duration:.2f}s
            Generation Time:    {model_response.eval_duration:.2f}s
            Total Time:         {model_response.total_duration:.2f}s
----------------------------------------------------
        """
    )


def average_stats(responses: List[OllamaResponse]) -> None:
    """
    Calculates and prints average statistics across multiple benchmark runs.

    Args:
        responses: List of OllamaResponse objects from multiple runs
    """
    if not responses:
        print("No stats to average")
        return

    # Calculate aggregate metrics
    res = OllamaResponse.aggregate(responses)
    print("Average stats:")
    inference_stats(res)


def table_stats(benchmarks: Dict[str, List[OllamaResponse]]) -> None:
    """
    Calculates and prints average statistics across multiple benchmark runs and models, output as table

    Args:
        benchmarks: Dict of modelNames and List of OllamaResponse objects from multiple runs
    """
    if not benchmarks:
        print("No results to output")
        return

    print("Table stats:")
    table: List[List] = []
    for model_name, responses in benchmarks.items():
        # Calculate aggregate metrics
        res = OllamaResponse.aggregate(responses)

        table.append([
            model_name,
            res.prompt_eval_rate,
            res.eval_rate,
            res.total_rate,
            res.load_duration,
            res.prompt_eval_count,
            res.prompt_eval_duration,
            res.eval_count,
            res.eval_duration,
            res.total_duration
        ])

    print(tabulate(table, headers=["Model\nName", "Prompt\nEvaluation Rate\n(T/s)", "Evaluation\nRate\n(T/s)",
                                   "Total\nRate\n(T/s)", "Load Time\n(s)",
                                   "Prompt\nEvaluation Count", "Prompt\nEvaluation Time\n(s)",
                                   "Evaluation\nCount", "Evaluation\nTime\n(s)", "Total Time\n(s)"], tablefmt="orgtbl",
                   floatfmt=".2f"))


def get_benchmark_models(test_models: List[str] = []) -> List[str]:
    """
    Retrieves and validates the list of models to benchmark.

    Args:
        test_models: List of specific models to test

    Returns:
        List of validated model names available for benchmarking
    """
    response = ollama.list()
    available_models = [model.get("model") for model in response.get("models", [])]

    if not test_models:
        # Use a default subset of models if none specified
        default_models = ["llama3", "mistral", "codellama", "deepseek", "gpt-oss", "gemma"]  # Common default models
        model_names = [m for m in available_models if any(d in m for d in default_models)]
        if not model_names:
            model_names = available_models[:3]  # Take first 3 available models if no defaults found
        # sort default subset alphabetically
        model_names.sort()
    else:
        # Filter requested models against available ones
        model_names = [model for model in test_models if model in available_models]
        if len(model_names) < len(test_models):
            missing_models = set(test_models) - set(available_models)
            print(f"Warning: Some requested models are not available: {missing_models}")

    if not model_names:
        raise RuntimeError("No valid models found for benchmarking")

    print(f"Evaluating models: {model_names}\n")
    return model_names


def main() -> None:
    """
    Main execution function for the benchmark tool.
    Handles argument parsing and orchestrates the benchmark process.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark performance metrics for Ollama models."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output including streaming responses",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="Specific models to benchmark. Tests all available models if not specified.",
    )
    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=[
            # Short analytical question to test basic reasoning
            "Explain the process of photosynthesis in plants, including the key chemical reactions and energy transformations involved.",

            # Medium-length creative task
            "Write a detailed story about a time traveler who visits three different historical periods. Include specific details about each era and the protagonist's interactions.",
            #
            # # Long complex analysis
            # "Analyze the potential impact of artificial intelligence on global employment over the next decade. Consider various industries, economic factors, and potential mitigation strategies. Provide specific examples and data-driven reasoning.",
            #
            # # Technical task with specific requirements
            # "Write a Python function that implements a binary search tree with methods for insertion, deletion, and traversal. Include comments explaining the time complexity of each operation.",
            #
            # # Structured output task
            # "Create a detailed business plan for a renewable energy startup. Include sections on market analysis, financial projections, competitive advantages, and risk assessment. Format the response with clear headings and bullet points.",
        ],
        help="Prompts to use for benchmarking. Multiple prompts can be specified. Default prompts test various capabilities including analysis, creativity, technical knowledge, and structured output.",
    )
    parser.add_argument(
        "-t",
        "--table_output",
        action="store_true",
        help="Output as table instead of separate results per model",
        default=False,
    )

    args = parser.parse_args()
    print(
        f"\nVerbose: {args.verbose}\nTest models: {args.models}\nPrompts: {args.prompts}\nTable Output: {args.table_output}"
    )

    model_names = get_benchmark_models(args.models)
    benchmarks: Dict[str, List[OllamaResponse]] = {}

    # Execute benchmarks for each model and prompt
    for model_name in model_names:
        responses: List[OllamaResponse] = []
        for prompt in args.prompts:
            if args.verbose:
                print(f"\n\nBenchmarking: {model_name}\nPrompt: {prompt}")

            if response := run_benchmark(model_name, prompt, verbose=args.verbose):
                responses.append(response)
                if args.verbose:
                    print(f"Response: {response.message.content}")
                    inference_stats(response)

        benchmarks[model_name] = responses

    if args.table_output:
        table_stats(benchmarks)
    else:
        # Calculate and display average statistics
        for model_name, responses in benchmarks.items():
            average_stats(responses)


if __name__ == "__main__":
    main()
