# Ollama based LLM Benchmark

This tool allows you to get the t/s (tokens per second) of Large Language Models (LLMs) running on your local machine. Currently we only support testing Ollama llms

## Example output

Output on a Nvidia 4090 windows desktop

```bash
Average stats:
(Running on dual 3090 Ti GPU, Epyc 7763 CPU in Ubuntu 22.04)

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

Average stats: 
(Running on single 3090 GPU, 13900KS CPU in WSL2(Ubuntu 22.04) in Windows 11)

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

Follow these instructions to set up and run benchmarks on your system.

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.com/) installed and configured

### Installation

#### Using pip

Install the package directly using pip using a tool of your choice.

```bash
pip install git+https://github.com/larryhopecode/ollama-benchmark.git
```

#### Local Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/larryhopecode/ollama-benchmark.git
   cd ollama-benchmark
   ```

2. **Set up Python environment**

   ```bash
   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**

Install the package in editable mode, in case you want to modify the code.

   ```bash
   pip install -e .
   ```

## Usage

1. **Start Ollama Server**

   Ensure the Ollama service is running:

   ```bash
   ollama serve
   ```

2. **Run Benchmarks**

   Basic usage:

   ```bash
   ollama-benchmark
   ```

   With options:

   ```bash
   ollama-benchmark --verbose --models deepseek-r1:70b --prompts "Write a hello world program" "Explain quantum computing"
   ```

### Command Line Options

- `-v, --verbose`: Enable detailed output including streaming responses
- `-m, --models`: Space-separated list of models to benchmark (defaults to all available models)
- `-p, --prompts`: Space-separated list of custom prompts (defaults to a predefined set testing various capabilities)

The default benchmark suite includes prompts testing:

- Analytical reasoning
- Creative writing
- Complex analysis
- Technical knowledge
- Structured output generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
