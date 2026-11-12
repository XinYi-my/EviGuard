# EviGuard

论文：[EviGuard: Evidence-Guided Connector Intervention for Cross-Modal Safety Unlearning in Multimodal LLMs](https://doi.org/10.1145/3767308.3834945)，ACM Multimedia 2026。

EviGuard 通过证据引导的多模态 connector 干预实现安全遗忘，包含风险结构融合（RSF）、connector 局部选择性遗忘（CLSU）和迭代反馈。代码提供 LLaVA-1.5-7B、InternVL2.5-8B 两种骨干模型的集成，支持 SafeEraser、VLGuard 的训练与评测，以及 SIUO 评测。

## 安装与使用

完整环境安装、模型路径、数据准备和命令示例见 [英文 README](README.md)。主要步骤如下：

1. 获取 LLaVA-1.5-7B 或 InternVL2.5-8B、BLIP-2 和相应数据集，修改 `configs/` 中的路径。示例配置分别对应骨干模型与数据集组合。
2. 创建独立 Python 环境并安装依赖。训练入口面向 CUDA，训练期间同时加载冻结参考模型。
3. 执行 `scripts/precompute_evidence.py --config ...` 预计算证据，再用 `scripts/diagnose_data.py --config ...` 检查数据与证据缓存。
4. 使用 `scripts/train_safeeraser.py` 或 `scripts/train_vlguard.py` 训练。阶段步数和 warmup 均以优化器更新次数计，梯度累积独立配置。
5. 使用 `scripts/eval_safeeraser.py` 或 `scripts/eval_vlguard.py` 评测，显式指定完整的 `--checkpoint`；基座模型评测使用 `--vanilla`。

## 实现与评测

RSF 将 OCR、图像标签与 connector 激活融合，按样本计算风险与调制量；CLSU 使用加权中心化 SVD 构建风险子空间，结合方向抑制、正交保留和冻结参考模型约束。checkpoint 包含可训练参数、connector、RSF、风险子空间及解析后的配置，加载后在生成过程中应用 RSF。

SafeEraser 按源图像分组确定性留出，训练和评测应保持源 JSON、图像引用及随机种子一致。SIUO 只用于评测。GPT 评测使用明确的提示词和严格解析，运行错误会显式报出；不满足计算前提的指标以 `null` 表示。

- [实现说明](docs/IMPLEMENTATION.md)
- [数据格式与评测协议](docs/DATA_AND_EVALUATION.md)
- [测试说明](docs/VALIDATION.md)

## 许可证

代码使用 [Apache-2.0](LICENSE) 许可证。第三方模型权重和数据集遵循各自许可证。
