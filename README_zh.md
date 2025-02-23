# Open-o3

> [!TIPS]
> 这是一个推理框架，而非模型。您可以将该框架用于任何 LLM 上。

简体中文 | [English](README.md)

![License](https://img.shields.io/badge/License-MIT-green.svg) ![Version](https://img.shields.io/badge/Version-0.1.0-blue) ![Last Updated](https://img.shields.io/badge/Last%20Updated-2025/2/21-orange)

Open-o3 是 OpenAI 的 o3 模型的开源版本。本项目旨在为研究人员和开发者提供一个开放、可访问的强大语言模型，促进人工智能领域的进步。Open-o3 致力于提供灵活、可定制的接口，支持各种 Reasoning 任务，如数值计算、编写程序、逻辑推理等。我们希望通过这一开源项目，促进 AI 技术的创新与应用。

## 项目描述

Open-o3 是一个高效、可扩展的语言模型，基于 OpenAI 的 o3 模型进行了开源复现。我们的目标是为开发者和研究人员提供一个可靠且易于使用的工具，帮助他们实现和探索更智能、更复杂的自然语言处理应用。

## 使用示例

```python
    system_prompt = "您是由流明智能开发的大型推理模型 Open-o3."
    reasoner = o3(system_prompt=system_prompt, enable_tda=True, save_results=True)   
    reasoner.run_example()
```

## 开发与贡献

我们欢迎任何形式的贡献，无论是报告 bug 还是提交新特性。要参与贡献，您可以：

1. 创建一个新 issue，报告问题或建议新功能。
2. 从仓库中 fork 出自己的分支，进行功能开发或 bug 修复。
3. 提交 pull request。

## 许可证

本项目采用 MIT 许可证，允许您自由使用、修改和分发代码。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 联系方式

如果你有任何问题、建议或反馈，请随时通过以下方式联系我们：

- 邮箱：[yuka@lumenlab.cc](mailto:yuka@lumenlab.cc)
- GitHub Issues: [https://github.com/ChihayaYuka/Open-o3](https://github.com/ChihayaYuka/Open-o3)

## 致谢

特别感谢所有为 Open-o3 项目做出贡献的开发者和研究人员。

> [!TIP]
> 我们也感谢 OpenAI 团队提供的启发，使得这一开源项目成为可能。

感谢以下开源项目的支持：

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch](https://pytorch.org/)

也感谢所有参与测试、报告问题和贡献代码的开发者们。

