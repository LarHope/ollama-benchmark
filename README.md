# Ollama Benchmark

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-supported-blueviolet.svg)](https://ollama.com/)

A lightweight benchmarking tool for measuring LLM inference performance through [Ollama](https://ollama.com/). Get detailed tokens-per-second metrics, load times, and generation speed for any model running on your local hardware.

## Features

- **Prompt processing speed** — measures tokens/sec for input evaluation
- **Generation speed** — measures tokens/sec for output generation
- **Model load time** — tracks cold-start overhead
- **Multi-model comparison** — benchmark several models in a single run
- **Table output** — side-by-side comparison across models with `-t`
- **Custom prompts** — supply your own prompts or use the built-in test suite
- **Zero dependencies on cloud** — everything runs locally through Ollama

## Example Output

**Dual 3090 Ti GPU, Epyc 7763 CPU — Ubuntu 22.04:**

```
----------------------------------------------------
        Model: deepseek-r1:70b
        Performance Metrics:
            Prompt Processing:  336.73 tokens/sec
            Generation Speed:   17.65 tokens/sec
            Combined Speed:     18.01 tokens/sec

        Workload Stats:
            Input Tokens:       165
            Generated Tokens:   7673
            Model Load Time:    6.11s
            Processing Time:    0.49s
            Generation Time:    434.70s
            Total Time:         441.31s
----------------------------------------------------
```

**Single 3090 GPU, 13900KS CPU — WSL2 (Ubuntu 22.04) on Windows 11:**

```
----------------------------------------------------
        Model: deepseek-r1:32b
        Performance Metrics:
            Prompt Processing:  399.05 tokens/sec
            Generation Speed:   27.18 tokens/sec
            Combined Speed:     27.58 tokens/sec

        Workload Stats:
            Input Tokens:       168
            Generated Tokens:   10601
            Model Load Time:    15.44s
            Processing Time:    0.42s
            Generation Time:    390.00s
            Total Time:         405.87s
----------------------------------------------------
```

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.com/) installed and running

### Installation

**Using pip** (recommended):

```bash
pip install git+https://github.com/LarHope/ollama-benchmark.git
```

**From source:**

```bash
git clone https://github.com/LarHope/ollama-benchmark.git
cd ollama-benchmark
python3 -m venv venv && source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -e .
```

## Usage

Make sure the Ollama server is running:

```bash
ollama serve
```

Run benchmarks:

```bash
# Benchmark all available models with default prompts
ollama-benchmark

# Benchmark specific models
ollama-benchmark --models deepseek-r1:70b llama3:8b

# Custom prompts
ollama-benchmark --models mistral --prompts "Write a hello world in Rust" "Explain quantum computing"

# Table comparison output
ollama-benchmark --table_output --models deepseek-r1:70b deepseek-r1:32b llama3:8b

# Verbose mode (shows streaming responses)
ollama-benchmark --verbose --models deepseek-r1:70b
```

### CLI Reference

| Flag | Description |
|------|-------------|
| `-v, --verbose` | Show streaming responses and per-prompt stats |
| `-m, --models` | Space-separated list of models to benchmark (default: all available) |
| `-p, --prompts` | Space-separated list of custom prompts |
| `-t, --table_output` | Display results as a comparison table |

### Default Benchmark Suite

When no custom prompts are provided, the tool runs a suite covering:

- Analytical reasoning
- Creative writing
- Complex analysis
- Technical knowledge
- Structured output generation

## How It Works

1. Connects to your local Ollama instance
2. Sends each prompt to each model
3. Captures timing metrics from the Ollama API response (total duration, prompt eval, generation)
4. Calculates tokens/sec for prompt processing, generation, and combined throughput
5. Outputs per-model averages or a comparison table

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
