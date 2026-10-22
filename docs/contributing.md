# 贡献指南

## 开始前

1. 先在 issue 中说明目标 provider、模型或端点、认证方式，以及期望的统一输入输出行为。
2. 不要把 API key、真实用户消息、厂商账户标识或响应日志提交进仓库。
3. 新功能优先复用 `ModelResponse`、`Choices`、`Usage` 与现有错误形状；需要扩展时，说明兼容性影响。

## 本地测试

```bash
python -m pip install -e . -r requirements-dev.txt
python -m pytest --ignore=tests/my
```

`tests/my/` 是被忽略的本地测试目录，会调用真实的外部服务，pytest 默认也会收集它，所以它不能作为稳定的回归门槛。上面的命令只运行受版本控制的测试，这些测试使用模拟的上游响应，不需要任何密钥。

仓库目前没有 lint、格式化、类型检查或 CI 配置。请至少保证新增和修改的测试在本地通过，并在 PR 中写明 Python 版本、openai 与 anthropic 的版本和执行的命令。

### 在多个版本下测试

UnionLLM 不锁定 openai 与 anthropic 的大版本，改动 SDK 相关代码时，请至少在新旧两组版本下各跑一次。用 [uv](https://docs.astral.sh/uv/) 可以很快建出隔离环境：

```bash
uv venv --python 3.10 .venv/old
uv pip install --python .venv/old/bin/python -e . -r requirements-dev.txt "openai==2.54.0" "anthropic==0.120.0"
.venv/old/bin/python -m pytest --ignore=tests/my

uv venv --python 3.13 .venv/new
uv pip install --python .venv/new/bin/python -e . -r requirements-dev.txt -U openai anthropic
.venv/new/bin/python -m pytest --ignore=tests/my
```

openai 3 与 anthropic 1 起，SDK 的 HTTP 层是 `httpx2`。测试里要拦截 SDK 的请求时，请模拟 `httpx2`，不要模拟 `httpx`。

## 新增或修改 provider 的最低要求

- 在 `unionllm/providers/` 实现认证、请求、非流式与（若上游支持）流式转换。
- 在 `unionllm/main.py` 显式注册路由，并说明是否、何时回退到 LiteLLM。新代码不要新增对 LiteLLM 的依赖。
- 为认证缺失、参数映射、非流式、流式和上游错误映射编写不依赖真实密钥的测试。
- 更新 [Provider 矩阵](providers.md)、[架构说明](architecture.md)、README 与 [CHANGELOG](../CHANGELOG.md)；不兼容的变化还要写进迁移说明。
- 新增依赖时，同步 `pyproject.toml` 与 `requirements.txt`，并说明最低兼容版本。

## 文档原则

- 以仓库实际的路由和测试为依据。价格、可售模型和开通规则链接到厂商官方文档，不在本仓库复制容易过期的清单。
- 示例必须能直接运行，不能包含失效的参数名或错误的 provider。
- provider 文档要写明特殊分支、环境变量和已知边界，不要承诺“所有模型、所有功能”。
- README 也是 PyPI 上的项目说明，PyPI 不解析相对链接，所以 README 中的链接使用指向 GitHub 的绝对地址；`docs/` 内部的链接使用相对地址。

## 发布

1. 更新 `pyproject.toml` 中的版本号。
2. 把 CHANGELOG 中“未发布”一节改为版本号和发布日期，并补上底部的链接。
3. 构建并检查 PyPI 上的项目说明能否正常渲染：

   ```bash
   python -m pip install build twine
   python -m build
   python -m twine check dist/*
   ```

4. 在干净环境里安装构建出的 wheel，跑一遍测试。
5. 上传 PyPI，并给发布提交打上 `vX.Y.Z` 标签。

## 提交与审查

一个 PR 只做一件事；代码、测试、依赖与文档一起提交。需要真实上游账户验证的改动，请在 PR 中写明哪些测试是模拟的、哪些做过真实调用，以及还有哪些风险没有覆盖。
