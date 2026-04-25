Card ID: FEAT-OPENAI-NATIVE-CHAT-COMPLETIONS
Project: UnionLLM
Repo: /var/www/UnionLLM
Worktree: /var/www/UnionLLM/.worktrees/UnionLLM/FEAT-OPENAI-NATIVE-CHAT-COMPLETIONS
Codex invoke: prompt
Result schema: /private/var/www/prd/skills/prd-worker/assets/result.schema.json
Result JSON path: /var/www/UnionLLM/.prd-autopilot/results/UnionLLM-FEAT-OPENAI-NATIVE-CHAT-COMPLETIONS.json
Worker log path: /var/www/UnionLLM/.prd-autopilot/results/UnionLLM-FEAT-OPENAI-NATIVE-CHAT-COMPLETIONS.log
Date: 2026-03-09
Started at: 2026-03-09T16:30:01.666Z
----
You are a coding agent working on ONE PRD card.

Required skill:
- You MUST use the prd-worker skill for this run.
- If you cannot access prd-worker, finish with outcome="blocked" and include a blocker: "prd-worker skill unavailable".

Hard constraints:
- Do NOT edit the PRD hub at /private/var/www/prd. Treat it as read-only.
- Make code changes ONLY inside the repo worktree at: /var/www/UnionLLM/.worktrees/UnionLLM/FEAT-OPENAI-NATIVE-CHAT-COMPLETIONS
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
id: FEAT-OPENAI-NATIVE-CHAT-COMPLETIONS
title: "UnionLLM: OpenAI models 原生接入 Chat/Completions（替代 LiteLLM），对齐统一协议"
type: feature
status: "pending"
priority: P1
severity: S2
component: provider
owner: ""
reporter: "everfly"
created_at: 2026-03-09
updated_at: "2026-03-09"
due_at: null
spec: "self"
related_files: []
related_cards: []
labels: ["UnionLLM","provider","openai","native","chat","completions","streaming"]
estimate: "M"
---

## Background / Problem Statement
当前 UnionLLM 对 OpenAI models 的支持依赖 LiteLLM 作为中间层。

依赖中间层带来的问题：
- 能力映射与一致性受限：streaming 事件/分片、tool calling/结构化输出、多模态输入、usage 字段等易出现偏差
- 维护与排障复杂度增加：多一层依赖与兼容层
- 版本演进滞后：OpenAI API 新能力、参数、返回结构变化需要等待中间层支持

参考文档（OpenAI API Reference）：
- https://developers.openai.com/api/reference/overview/

## Impact
- User impact:
  - OpenAI models 可直接走 UnionLLM 原生 provider，减少不确定性
  - streaming、usage 等行为与 UnionLLM 其他原生 provider 更一致
  - 降低对 LiteLLM 的耦合，减少部署复杂度
- Components affected:
  - provider/adapter 层（新增 openai-native 实现）
  - streaming 聚合与统一 response 映射
  - 鉴权与 endpoint 配置（base_url / proxy）

## Current Behavior
- OpenAI models 通过 LiteLLM 路径接入（非 OpenAI 原生 Chat/Completions）

## Expected Behavior
在 UnionLLM 中新增 OpenAI models 的原生实现：
- 支持原生 **Chat Completions**（以及需要时的 **Completions**）
- 支持 **non-stream + stream**
- 对齐 UnionLLM 现有 request/response 规范（统一消息结构、统一 usage、统一错误结构）
- 支持鉴权：环境变量 `OPENAI_API_KEY` + 调用时传入（调用参数优先）
- endpoint 默认 `https://api.openai.com`，并支持 `custom base_url`

> 说明：若 UnionLLM 已将“对话”统一收敛到单一接口（例如 chat），可以优先做 chat；completions 可作为兼容层实现。

## Solution / Design (Optional)
### Provider：OpenAI Native
- Chat：`POST {base_url}/v1/chat/completions`
- Completions：`POST {base_url}/v1/completions`
- Headers：`Authorization: Bearer <key>`，`Content-Type: application/json`
- base_url：默认 `https://api.openai.com`，允许通过配置覆盖（代理/网关）

### Request/Response 映射（对齐 UnionLLM 统一规范）
- `model`：保留 `gpt-*` 等模型名
- `messages`：UnionLLM messages → OpenAI chat messages
- `stream`：映射到 OpenAI stream（SSE）
- `tools/tool_choice`（如 UnionLLM 已支持）：映射到 OpenAI tool calling
- `response_format`/JSON schema（如 UnionLLM 已支持结构化输出）：映射到 OpenAI 相关参数
- `usage`：把 OpenAI usage 映射到 UnionLLM 统一 usage 字段

### Streaming
- OpenAI SSE chunk → UnionLLM streaming 分片规范
- 保证：delta 拼接正确、finish_reason 正确、错误可追踪

## Acceptance Criteria
- [ ] Chat non-stream：纯文本对话可用，并符合 UnionLLM 现有 response 结构
- [ ] Chat stream：SSE 流式输出可用，分片/结束事件符合 UnionLLM 现有 streaming 规范
- [ ] Completions non-stream：在启用 completions 时可用（若决定实现）
- [ ] Completions stream：在启用 completions 时可用（若决定实现）
- [ ] 鉴权：仅设置 `OPENAI_API_KEY` 可跑通；调用时传入 key 可覆盖环境变量
- [ ] base_url：默认 endpoint 可用；custom base_url 可用
- [ ] 错误结构：OpenAI 错误能映射为 UnionLLM 统一错误结构（含 status code、message、request_id 若可得）

## Test Plan
- Build/test commands:
  - TBD（按 UnionLLM repo 现有测试方式补充）
- Manual validation:
  - chat：非流式与流式
  - completions：非流式与流式（若实现）
  - base_url：指向代理验证
- Regression areas:
  - 现有 LiteLLM OpenAI 路径（需可回滚或并存）
  - streaming 聚合器对其他 provider 的行为

## Risks & Rollback
- Risks:
  - OpenAI 参数/接口演进较快，需要明确 UnionLLM 对齐的“统一协议”版本
  - streaming chunk 与 UnionLLM streaming 规范差异需要适配
- Rollback plan:
  - 提供 feature flag / 配置开关切回 LiteLLM 路径

## Clarifications / Open Questions
- 是否需要同时支持新的 Responses API（若 UnionLLM roadmap 已转向该接口）？本需求目前只要求 chat/completions。
- 是否需要覆盖多模态输入（image）与音频/实时等能力（可拆分后续卡）？

## Progress Log
### 2026-03-09
- Status: pending
- Completed:
  - 创建需求卡：OpenAI models 原生接入 Chat/Completions，替代 LiteLLM
- Next:
  - 明确 UnionLLM 统一协议中与 OpenAI 对齐的字段集（tools、response_format、usage、error）
  - 实现 openai-native provider 并补齐回归测试
- Blockers:
  - 需确认 UnionLLM 现有 request/response & tools/structured-output schema 细节
- Notes:
  - 参考：https://developers.openai.com/api/reference/overview/

## Autopilot

### 2026-03-09

- Status: blocked
- Reason: Missing repo mapping in hub AGENT.md
- Blockers:
  - project=UnionLLM
---

Now begin.

Reminder: Immediately before the FINAL JSON, output a short natural-language summary message.
Reminder: Your FINAL message must be a single JSON object matching the schema.