# EviGuard 开源说明

论文：[EviGuard: Evidence-Guided Connector Intervention for Cross-Modal Safety Unlearning in Multimodal LLMs](https://doi.org/10.1145/3767308.3834945)，ACM Multimedia 2026。

本仓库是根据保留下来的提交版源码整理、并对照另一份实验快照和最终论文修正的维护版源码。原服务器已过期，最终 checkpoint、完整训练日志、最终解析配置及原始预测/评测记录未能保留下来。本仓库不提供替代这些历史产物的文件，也不声称已经重新复现论文数值。

`configs/` 中四份 YAML 是新实验的示例起点，不是恢复出来的论文最终配置。修复涉及梯度路径、损失尺度、数据划分、步数定义及评测方式，因此新运行结果必须重新验证。代码来源和输入文件 SHA256 见 [SOURCE_MANIFEST.json](docs/SOURCE_MANIFEST.json)。

完整安装、模型路径、证据预计算、训练、恢复、评测命令见 [英文 README](README.md)。主要步骤如下：

1. 获取原始 LLaVA-1.5-7B 或 InternVL2.5-8B、BLIP-2 和相应数据集，修改 YAML 中的路径。
2. 创建独立 Python 环境，安装依赖。实际训练入口面向 CUDA；同时需要常驻冻结参考模型。
3. 执行 `scripts/precompute_evidence.py --config ...`，再用 `scripts/diagnose_data.py --config ...` 检查数据和证据。
4. 使用 `scripts/train_safeeraser.py` 或 `scripts/train_vlguard.py` 训练。阶段步数和 warmup 均以优化器更新次数计，梯度累积独立配置。
5. 使用 `scripts/eval_safeeraser.py` 或 `scripts/eval_vlguard.py`，显式指定完整 `--checkpoint`，或显式选择 `--vanilla`。SIUO 仅用于评测。

本次维护修正了 RSF 第二次前向对 connector 的梯度、按样本对应 InternVL patch、中心化风险子空间及几何损失、Stage 1 最终损失、视觉 token 的训练标签遮罩，以及完整干预状态的保存/恢复等问题。API 评测失败不会退化为关键词或文本重叠分数，数据/图片缺失和生成失败不会算作安全成功。

SafeEraser 使用按图像分组的确定性留出协议，不能声称与历史实验划分一致。历史脚本中不符合其方法定义或存在评测/恢复问题的基线、消融自动运行入口没有纳入本次公开运行范围；详细原因见 [版本说明](docs/RELEASE_NOTES.md)。单元测试中的合成输入只验证程序行为，不能用作论文实验结果。

保留原仓库 Apache-2.0 许可证；不分发第三方模型权重和数据集。
