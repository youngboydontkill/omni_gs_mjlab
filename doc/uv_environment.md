# uv 环境要求

## 必须固定的版本

本 workspace 使用本地改造版 `rsl-rl-lib`，其中包含 DeFM 支持。以下版本必须保持固定：

- Python：`>=3.11,<3.12`
- `mjlab==1.4.0`
- `rsl-rl-lib==5.2.0`
- `huggingface-hub>=0.36.0`
- `omegaconf>=2.3.0`

`rsl-rl-lib` 必须解析到 workspace 成员 `packages/rsl_rl`。不要替换为 PyPI 包，也不要升级到 `5.2.0` 以上，因为 `mjlab==1.4.0` 依赖 5.2.0 版本的接口。

当前 `uv.lock` 中主要运行时包的解析结果如下：

| 包 | 锁定版本 / 来源 |
| --- | --- |
| `mjlab` | `1.4.0` |
| `rsl-rl-lib` | `5.2.0`，editable，来自 `packages/rsl_rl` |
| `torch` | `2.12.0` |
| `torchvision` | `0.27.0` |
| `tensordict` | `0.13.0` |
| `numpy` | `2.4.6` |
| `huggingface-hub` | `1.19.0` |
| `omegaconf` | `2.3.1` |

请把 `uv.lock` 视为传递依赖快照的权威来源。上表中的精确传递依赖版本只有在有意执行 `uv lock` 更新时才应变化。

## DeFM 安装模型

不要安装 `defm` 源码包。`DefmModel` 使用以下方式加载 `defm_vit_s14`：

```python
torch.hub.load("leggedrobotics/defm:main", "defm_vit_s14", pretrained=True)
```

第一次构造模型时需要网络访问，并会把 DeFM 仓库源码写入 TorchHub 缓存。启用 pretrained 模式时还会下载模型 checkpoint。如果缓存需要放到指定位置，请在运行前设置 `TORCH_HOME`：

```bash
export TORCH_HOME="$HOME/.cache/torch"
```

如果需要可复现或离线部署，请将 `repo_or_dir` 设置为固定的 DeFM Git ref，并通过 `pretrained_path` 提供本地 checkpoint。

安装 DeFM 源码包会把 DeFM 自身的依赖约束带入 uv 解析过程，其中包括不同的 NumPy 约束。因此，本 workspace 中不要安装 DeFM 源码包。

## 创建和同步环境

所有 uv 命令都应从仓库根目录运行：

```bash
uv sync --locked
```

当依赖声明被有意修改后，先重新生成 lock 文件，再同步环境：

```bash
uv lock
uv sync --locked
```

不要在缺少 workspace source 配置的情况下运行 `uv add rsl-rl-lib`。根目录 `pyproject.toml` 必须保留以下配置：

```toml
[tool.uv.sources]
rsl-rl-lib = { workspace = true }

[tool.uv.workspace]
members = ["packages/rsl_rl"]
```

## 验证

以下命令可以在不加载 DeFM、不使用 GPU 的情况下验证依赖解析：

```bash
uv lock --check
```

构造 `DefmModel` 之前，请确认系统内存和 GPU 显存足够。默认情况下，模型会把每个 ViT patch token 压缩为 32 个分组通道均值，然后再展平 token。更高的深度图分辨率仍会增加策略输入维度和 rollout feature cache 的大小。
