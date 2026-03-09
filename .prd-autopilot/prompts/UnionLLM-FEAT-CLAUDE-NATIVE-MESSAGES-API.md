Card ID: FEAT-CLAUDE-NATIVE-MESSAGES-API
Project: UnionLLM
Repo: /var/www/UnionLLM
Worktree: /var/www/UnionLLM/.worktrees/UnionLLM/FEAT-CLAUDE-NATIVE-MESSAGES-API
Codex invoke: prompt
Result schema: /var/www/prd/skills/prd-worker/assets/result.schema.json
Result JSON path: /var/www/UnionLLM/.prd-autopilot/results/UnionLLM-FEAT-CLAUDE-NATIVE-MESSAGES-API.json
Worker log path: /var/www/UnionLLM/.prd-autopilot/results/UnionLLM-FEAT-CLAUDE-NATIVE-MESSAGES-API.log
Date: 2026-03-09
Started at: 2026-03-09T04:30:06.829Z
----
You are a coding agent working on ONE PRD card.

Required skill:
- You MUST use the prd-worker skill for this run.
- If you cannot access prd-worker, finish with outcome="blocked" and include a blocker: "prd-worker skill unavailable".

Hard constraints:
- Do NOT edit the PRD hub at /var/www/prd. Treat it as read-only.
- Make code changes ONLY inside the repo worktree at: /var/www/UnionLLM/.worktrees/UnionLLM/FEAT-CLAUDE-NATIVE-MESSAGES-API
- Writing to the supervisor-provided artifact paths (Result JSON path / Worker log path) is allowed (and required in prompt/TUI mode).
- Immediately before the FINAL JSON, output a short natural-language summary message (not JSON).
- You MUST finish by emitting a FINAL JSON response matching the required output schema.
- Your FINAL message must be ONLY the JSON object (no prose before/after).
- Include the same human-readable summary inside the FINAL JSON "notes" field so it is captured in logs/artifacts.
  - outcome: "in-review" if you implemented + validated the change.
  - outcome: "blocked" if you cannot proceed (missing info, cannot run validation, unclear AC, etc.).

IMPORTANT (prompt/TUI mode): Codex cannot auto-save your last message.
- You MUST ALSO write the same FINAL JSON object to the file path in "Result JSON path".
- Write ONLY the JSON object to that file (do not include the human-readable summary).
- The "Result JSON path" may be outside the worktree; writing to it is permitted for this run.
- You may use PRD_AUTOPILOT_RESULT_PATH / PRD_AUTOPILOT_SCHEMA_PATH env vars if available.

PRD card content:
---
---
id: FEAT-CLAUDE-NATIVE-MESSAGES-API
title: "UnionLLM: Claude models 原生接入 Anthropic Messages API（替代 LiteLLM），支持 stream/多模态/tool calling/prompt caching"
type: feature
status: pending
priority: P1
severity: S2
component: provider
owner: ""
reporter: "everfly"
created_at: 2026-03-09
updated_at: 2026-03-09
due_at: null
spec: "self"
related_files: []
related_cards: []
labels: ["UnionLLM","provider","anthropic","claude","native","prompt-caching"]
estimate: "M"
---

## Background / Problem Statement
目前 UnionLLM 对 Claude models 的支持依赖 LiteLLM（而类似 moonshot 等 provider 已经有原生实现）。

依赖中间层带来的问题：
- 能力映射与一致性受限：tool calling、多模态、streaming 事件/分片、usage 字段等容易出现不一致
- 引入额外维护成本：多一层依赖与排障路径
- Anthropic 新能力落地滞后：例如 prompt caching

参考文档（Anthropic Messages API）：
- https://platform.claude.com/docs/en/get-started

## Impact
- User impact:
  - UnionLLM 用户可直接使用 Claude models 的原生能力（stream、text+image、tool calling）
  - 减少 LiteLLM 依赖导致的兼容性问题
  - 支持 prompt caching 以降低成本/延迟（取决于调用场景与命中率）
- Components affected:
  - provider/adapter 层（新增 anthropic-native 实现）
  - streaming 聚合与统一 response 映射
  - usage/metrics 结构（增加 caching 相关字段映射，若适用）

## Current Behavior
- Claude models 通过 LiteLLM 路径接入（非 Anthropic 原生 Messages API）
- prompt caching 无法按 Anthropic 官方方式启用或行为不可控

## Expected Behavior
在 UnionLLM 中新增 Claude models 的原生实现：
- 仅支持 Anthropic **Messages API**（不做 Completions API）
- 支持 **non-stream + stream**
- 支持 **text + image** 输入
- 支持 **tool calling**（并与 UnionLLM 现有 request/response 规范对齐）
- 支持 Anthropic **prompt caching**（automatic cache）
- 支持鉴权：环境变量 `ANTHROPIC_API_KEY` + 调用时传入（调用参数优先）
- endpoint 默认 `https://api.anthropic.com`，并支持 `custom base_url`

## Solution / Design (Optional)
### Provider：Anthropic Native（Messages API）
- 请求：`POST {base_url}/v1/messages`
- Headers（以官方为准）：
  - `x-api-key: <key>`
  - `anthropic-version: <version>`
  - `content-type: application/json`
- base_url：默认 `https://api.anthropic.com`，允许通过配置覆盖（代理/网关）

### Request 映射（对齐 UnionLLM 统一规范）
- `model`：UnionLLM 侧保留 `claude-*` 的选择
- `messages` → Anthropic `messages`
- system prompt：映射到 Anthropic `system`（如 UnionLLM 已有该抽象）
- 多模态：将 UnionLLM 的 image 输入映射为 Anthropic content blocks（`type: image` + source）
- tools：UnionLLM tool schema → Anthropic `tools`

### Stream
- Anthropic SSE → UnionLLM streaming 分片规范
- 保证：delta 拼接正确、结束信号正确、工具调用事件正确、异常可恢复/可追踪

### Prompt Caching（automatic cache）
- 目标：支持 Anthropic prompt caching（如 `cache_control`）
- “automatic”策略建议（可调整）：
  - 对 system prompt、tools schema、以及长上下文固定前缀默认加 cache_control
  - 支持 per-request 覆盖开关（例如 `enable_prompt_cache` / `cache_policy`，命名以 UnionLLM 现有风格为准）
- 输出：尽可能把 Anthropic 返回的 caching/usage 指标映射进 UnionLLM usage（若官方返回）

## Acceptance Criteria
- [ ] Claude non-stream：纯文本对话可用，并符合 UnionLLM 现有 response 结构
- [ ] Claude stream：SSE 流式输出可用，分片/结束事件符合 UnionLLM 现有 streaming 规范
- [ ] text + image：输入图片可用（按 UnionLLM 既有 image 输入结构），映射到 Anthropic content blocks 正确
- [ ] tool calling：UnionLLM tools 入参可用，返回的 tool call 结构可被 UnionLLM 上层统一消费
- [ ] 鉴权：仅设置 `ANTHROPIC_API_KEY` 可跑通；调用时传入 key 可覆盖环境变量
- [ ] base_url：默认 endpoint 可用；custom base_url 可用
- [ ] prompt caching：开启后请求体正确携带 cache_control；响应中的 caching/usage（如有）能在 UnionLLM 中体现

## Test Plan
- Build/test commands:
  - TBD（按 UnionLLM repo 现有测试方式补充）
- Manual validation:
  - 非流式：发送简单对话，验证 response 结构
  - 流式：验证 token/delta 连续性、终止事件、错误处理
  - 多模态：发送带图片输入的请求
  - 工具调用：构造一个可触发 tool call 的样例
  - caching：同一前缀多次请求验证 cache 相关指标（若可观察）
- Regression areas:
  - 现有 LiteLLM Claude 路径（需可回滚或并存）
  - streaming 聚合器对其他 provider 的行为

## Risks & Rollback
- Risks:
  - Anthropic stream 事件与 UnionLLM streaming 规范不一致，需额外适配
  - tool calling block 映射差异导致上层解析问题
  - prompt caching 的“automatic策略”涉及隐私/驻留策略与命中收益，需要明确默认行为
- Rollback plan:
  - 提供 feature flag / 配置开关切回 LiteLLM 路径

## Clarifications / Open Questions
- Open questions:
  - UnionLLM 对 caching/usage 字段的标准结构是什么？是否已有字段承载 cache hit/cache tokens？
  - image 输入在 UnionLLM 的标准格式（URL/base64/bytes）具体是哪一种/哪些？
  - tool calling 在 UnionLLM 的统一 schema（tool definition & tool call response）细节需要对齐到哪个版本？

## Progress Log
### 2026-03-09
- Status: pending
- Completed:
  - 创建需求卡：Claude models 原生接入 Anthropic Messages API，支持 stream/多模态/tool calling/prompt caching
- Next:
  - Autopilot 分发实现任务到对应 repo
- Blockers:
  - 需要确认 UnionLLM repo 的 request/response & tools schema 细节（用于精确映射）
- Notes:
  - 参考：https://platform.claude.com/docs/en/get-started
---

Now begin.

Reminder: Immediately before the FINAL JSON, output a short natural-language summary message.
Reminder: Your FINAL message must be a single JSON object matching the schema.