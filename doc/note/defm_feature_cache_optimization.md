# DeFM Feature Cache Optimization

## Background

Kuavo S54 Rough 使用冻结的 DeFM ViT-S/14 编码 `42x42` 米制深度图。优化前，每张深度图产生
`3x3` 个 Patch Tokens，每个 Token 为 `384` 维，展平后得到 `3456` 维 Feature。

实际训练时间对比如下：

| Task | Collection time | Learning time |
| --- | ---: | ---: |
| Kuavo S54 Rough + DeFM | 2.249 s | 6.064 s |
| Head-CNN | 1.080 s | 0.203 s |

DeFM Learning time 约为 Head-CNN 的 29.9 倍。主要原因并非 Encoder 参数被更新，而是 PPO 每轮执行
`5` 个 learning epochs 和 `4` 个 mini-batches，Actor 与 Critic 在每个 mini-batch 中都会重新执行一次完整
DeFM ViT。共享 Encoder 仅共享参数，不共享前向计算结果。

## Implemented Optimization

### Patch Token Feature Reduction

每个 `384` 维 Patch Token 按连续通道分为 `32` 组，每组包含 `12` 个通道，并使用固定平均进行降维：

```text
9 x 384
-> reshape to 9 x 32 x 12
-> mean over each 12-channel group
-> 9 x 32
-> flatten to 288
```

`defm_cfg.token_feature_dim` 默认设置为 `32`，并要求它是 `384` 的正因数。

该修改将 DeFM Feature 从 `3456` 维降低至 `288` 维，减少策略 MLP 参数量、反向传播计算量和 rollout
缓存占用。

### General Rollout Extra Storage

`RolloutStorage.Transition`、`RolloutStorage` 和 `RolloutStorage.Batch` 增加可选的
`extra: TensorDict | None`，用于保存与 rollout 样本严格对齐的额外张量。

DeFM 使用以下嵌套结构缓存 Feature：

```python
{
    "actor_features": {
        "actor_depth": Tensor[B, 288],
    },
    "critic_features": {
        "critic_depth": Tensor[B, 288],
    },
}
```

该结构支持后续扩展多个相机、多个视角和多个 Encoder。Storage 在首个 transition 延迟分配缓存，并在
后续 transition 中校验 key、shape 和 dtype 不变。

`extra` 使用与 observations 相同的索引进入 feedforward mini-batch，并使用相同的 dones 进行 recurrent
trajectory split 和 padding。

### PPO Frozen Feature Cache

当 Actor 和 Critic 均声明支持 Feature Cache，且所有 DeFM Encoder 均被冻结时，PPO 自动启用缓存：

1. Rollout collection 时，Actor 和 Critic 分别执行一次 DeFM Encoder。
2. 编码结果被 detach 后保存到 `transition.extra`。
3. PPO learning 阶段通过 `batch.extra` 直接执行策略 MLP，不再调用 DeFM Encoder。
4. `compute_returns()` 仍对 rollout 末尾 observation 执行一次 Critic DeFM 前向。

缓存模式不支持 symmetry augmentation，因为镜像后的 observation 无法与原始缓存 Feature 保持一致。

冻结参数已从优化器中排除，共享的可训练参数也会按对象去重。

## Compatibility Impact

- DeFM 策略输入从 `3456` 维 Feature 改为 `288` 维 Feature。
- 修改后的策略 MLP 与旧 DeFM checkpoint 不兼容，需要重新训练。
- `trainable=True` 的 DeFM Encoder 不启用缓存，并保留原有梯度路径。
- 非 DeFM 模型和未提供 `extra` 的现有 PPO、Storage、Distillation 路径保持原有行为。
- JIT/ONNX 推理仍从原始深度执行 DeFM，并使用相同的固定分组平均。

## Completed Verification

已完成以下轻量 CPU/mock 验证，未加载真实 TorchHub DeFM，也未使用 GPU：

- `9x384` Patch Tokens 正确转换为 `288` 维 Feature。
- 从缓存 Feature 执行策略头与正常前向输出一致。
- 嵌套 `extra` 在 feedforward shuffle 后与 rollout 样本保持对齐。
- 嵌套 `extra` 在 recurrent trajectory padding 后与 observations 保持对齐。
- 模拟 runner 的 `torch.inference_mode()` collection 后，Storage 缓存仍为普通 Tensor，可参与 learning。
- PPO update 前后 mock DeFM 调用次数保持 `9 -> 9`，确认 learning 阶段不再执行 Encoder。
- 冻结 DeFM 参数不在优化器中。
- 非 DeFM PPO 路径保持 `extra=None` 并正常完成 update。
- Python 语法编译和 `git diff --check` 通过。

完整 pytest 未运行：当前虚拟环境未安装可用的 `pytest`。真实 TorchHub、GPU 和训练验证暂未执行，以免影响
当前训练任务。

## Follow-up Plan

资源允许后执行以下验证：

1. 使用真实 DeFM 运行短周期 Kuavo S54 Rough 训练，记录 Collection time、Learning time 和 GPU 显存。
2. 确认 Learning time 相比优化前的 `6.064 s` 显著下降，并验证 learning 阶段无 DeFM Encoder 调用。
3. 对比优化前后的 reward、KL、value loss 和策略收敛速度，评估固定分组平均的信息损失。
4. 补齐完整 rsl_rl pytest、JIT 和 ONNX 导出测试。
5. 根据训练结果评估是否需要支持可配置缓存 dtype，或进一步合并 Actor/Critic 相同深度输入的编码计算。

## Expected Result

- 每次 PPO update 中 DeFM Encoder 调用次数从约 `40` 次降低为 `0` 次。
- DeFM Feature 缓存和策略输入维度降低约 `12` 倍。
- Collection time 仍包含 Actor/Critic DeFM 前向，不会出现同等幅度下降。
- Learning time 应显著下降并接近普通 MLP/CNN 策略的数量级。
