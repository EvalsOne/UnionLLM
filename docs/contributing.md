# 贡献指南

## 开始前

1. 先在 issue 中说明目标 provider、模型/端点、认证方式和期望的统一输入输出行为。
2. 不要把 API key、真实用户消息、厂商账户标识或响应日志提交进仓库。
3. 新功能应优先复用 `ModelResponse`、`Choices`、`Usage` 与现有错误形状；如需扩展，必须说明兼容性影响。

## 本地校验

```bash
python -m pip install -r requirements.txt
python -m pytest --ignore=tests/my
```

`tests/my/` 是被忽略的本地/真实服务测试目录，pytest 的默认递归会收集它；它不能作为稳定回归门槛。上面的命令只运行受版本控制的测试。项目目前没有 lint、格式化、类型检查或 CI 配置被跟踪。提交者应至少保证新增/更新测试在本地通过，并在 PR 中说明 Python 版本与执行命令。

## 新增或修改 provider 的最低交付面

- 在 `unionllm/providers/` 实现认证、请求、非流式与（若上游支持）流式转换。
- 在 `unionllm/main.py` 显式注册路由，并明确是否/何时回退至 LiteLLM。
- 为认证缺失、参数映射、非流式、流式和上游错误映射编写不依赖真实密钥的测试。
- 更新[Provider 矩阵](providers.md)、[架构说明](architecture.md)与 README；不要仅更新一个示例页面。
- 若新增依赖，同步 `pyproject.toml` 与 `requirements.txt`，并说明最小兼容版本。

## 文档原则

- 以仓库实际路由和测试为依据；价格、可售模型和账号开通规则链接到官方文档，不在本仓库复制易过期清单。
- 示例必须可直接解析，不能包含失效的参数名或错误 provider。
- provider 页面应指出实现的特殊分支、环境变量和已知边界，不能承诺“所有模型/所有功能”。

## 提交与审查

保持 PR 单一目的；代码、测试、依赖与文档一起提交。若变更需要真实上游账户验证，在 PR 中明确哪些测试是 mock、哪些是已实际验证，以及未覆盖的风险。
