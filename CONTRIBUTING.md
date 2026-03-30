# Contributing to Ollama Benchmark

Thanks for your interest in contributing! Here's how to get started.

## Getting Started

1. Fork the repo and clone your fork
2. Create a virtual environment: `python3 -m venv venv && source venv/bin/activate`
3. Install in editable mode: `pip install -e .`
4. Create a branch: `git checkout -b my-feature`

## Development

### Running the tool locally

```bash
ollama serve  # make sure Ollama is running
python benchmark.py --verbose --models <your-model>
```

### Code Style

- Follow PEP 8
- Use type hints where practical
- Keep functions focused and well-documented

## Submitting Changes

1. Make sure your changes work locally
2. Write a clear commit message (e.g., `fix: guard against zero division`, `docs: add usage example`)
3. Open a Pull Request with a brief description of what and why

## Reporting Bugs

Open an issue using the **Bug Report** template. Include:
- What you expected to happen
- What actually happened
- Your environment (OS, Python version, Ollama version, GPU)

## Feature Requests

Open an issue using the **Feature Request** template. Describe the use case and, if possible, sketch a proposed approach.

## Questions?

Open a Discussion or reach out in an issue. We're happy to help.
