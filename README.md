<div align="center">

![Logo](./.github/media/logo.png)

<br><br>

# Open-o3

[Blog](https://yuka.living/2025/02/26/open-o3-a-framework-for-exponentially-improving-llm-accuracy-through-systematic-resampling/) | [Website](https://lumenlab.cc) | [Feedback](https://github.com/ChihayaYuka/Open-o3/issues)

</div>

> [!TIP]
> This is a framework of reasoning, not a model.    You can use the framework on any LLM.

Open-o3 is the open-source version of OpenAI's o3.    This project aims to provide a powerful, open, and accessible language model for researchers and developers, advancing the field of artificial intelligence.    Open-o3 is committed to offering flexible and customizable interfaces to support various reasoning tasks such as numerical calculations, programming, logic reasoning, etc.    Through this open-source project, we hope to foster innovation and application of AI technology.

## Project Description

Open-o3 is an efficient and scalable language model, which is an open-source reproduction of OpenAI's o3 model.    Our goal is to provide developers and researchers with a reliable and easy-to-use tool to help them implement and explore smarter and more complex natural language processing applications.

## Features

- **Systematic Resampling**: Implements OpenAI's o3 approach to exponentially improve accuracy through iterative reasoning.
- **Model-Agnostic**: Compatible with any LLM, allowing you to leverage your preferred model.
- **Result Tracking**: Save and analyze reasoning paths to improve performance.
- **Customizable Prompts**: Tailor the system prompt to your specific use cases and domains.
- **Extensible Architecture**: Easily build upon the core framework for specialized applications.

## Quick Start

### Basic Usage

```python
from open_o3 import o3

system_prompt = "You are a large reasoning model Open-o3 developed by Lumen Lab."
reasoner = o3(system_prompt=system_prompt, enable_tda=True, save_results=True)
reasoner.run_example()
```

### Advanced Configuration

```python
from open_o3 import o3

# Custom configuration
reasoner = o3(
system_prompt="You are specialized in solving mathematical problems.",
model="deepseek-r1",
max_iterations=5,
temperature=0.7,
enable_tda=True,
save_results=True,
result_path="./reasoning_logs/"
)

# Solve a specific problem
result = reasoner.solve("What is the integral of x²?")
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | str | Required | Initial instructions for the model |
| `model` | str | "gpt-3.5-turbo" | The LLM to use |
| `enable_tda` | bool | False | Whether to use TDA |
| `max_iterations` | int | 3 | Maximum reasoning iterations |
| `temperature` | float | 0.8 | Sampling temperature for the model |
| `save_results` | bool | False | Whether to save reasoning paths |
| `result_path` | str | ".   /results/" | Path to save reasoning logs |

## Examples

### Mathematical Reasoning

```python
result = reasoner.solve("Solve for x: 2x² + 5x - 3 = 0")
```

### Code Generation

```python
code = reasoner.solve("Write a function to check if a string is a palindrome in Python")
```

### Logical Reasoning

```python
analysis = reasoner.solve("If all A are B, and some B are C, what can we conclude about A and C?")
```

## Usage Example

```python
system_prompt = "You are a large reasoning model Open-o3 developed by Lumen Lab."
reasoner = o3(system_prompt=system_prompt, enable_tda=True, save_results=True)
reasoner.run_example()
```

## Development and Contribution

We welcome contributions in any form, whether it's reporting bugs or submitting new features.    To contribute, you can:

1. Create a new issue to report a problem or suggest a new feature.
2. Fork the repository and develop new features or fix bugs on your own branch.
3. Submit a pull request.

## Roadmap

- [ ] Multi-model ensemble reasoning
- [ ] Fine-tuning support for specific domains
- [ ] Performance benchmarks against other reasoning frameworks
- [ ] Community templates for common reasoning tasks

## Citation

If you use Open-o3 in your research, please cite:

```bibtex
@software{Open-o3,
author = {Lumen Lab},
title = {Open-o3: A Framework for Exponentially Improving LLM Accuracy},
url = {https://github.com/ChihayaYuka/Open-o3},
year = {2025},
}
```

## License

This project is licensed under the MIT License, allowing you to freely use, modify, and distribute the code.    For more details, please refer to the [LICENSE](LICENSE) file.

## Contact Information

If you have any questions, suggestions, or feedback, feel free to contact us via the following m