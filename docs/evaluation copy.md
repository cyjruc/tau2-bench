# 任务模式与评估
本文档讲解 τ‑bench 中任务的打分机制，最重要的是说明 **`evaluation_criteria.actions` 的实际作用**。如果你看过 `data/tau2/domains/airline/tasks.json`，并且以为文件里列出的动作是智能体必须执行的，那么本文档正是为你准备的。

## 一句话速览（TL;DR）
- 任务最终奖励是 `evaluation_criteria.reward_basis` 中各个分量的**乘积**。
- 航空、零售、电信领域默认的 `reward_basis` 为 `["DB", "COMMUNICATE"]`，与原始 τ‑bench 论文保持一致。
- `evaluation_criteria.actions` 只是**一条可以完成任务的参考轨迹**，不是唯一正确路径。系统会在全新的“标准环境”中复现该参考动作，得到目标数据库最终状态，之后通过哈希值与智能体运行后的数据库状态做比对。**并不强制智能体必须走这条路径**；只要工具调用序列最终产出等价的数据库状态，即可通过数据库校验。
- 只有当 `reward_basis` 中包含 `RewardType.ACTION` 时，才会要求智能体必须复现 `actions` 中的动作。该模式仅在少量 `banking_knowledge` 任务中使用；**航空/零售/电信完全没有使用该模式**。开启该标记后，列表动作被视作唯一正确轨迹，这个假设条件很强，因此绝大多数 τ‑bench 任务都刻意不启用。
- 这是刻意的设计决策，详见 [issue #224][issue‑224] 和 [RFC #129][issue‑129] 的相关讨论。本文把这点明确写出，避免使用者踩坑。

[issue‑224]: https://github.com/sierra‑research/tau2‑bench/issues/224
[issue‑129]: https://github.com/sierra‑research/tau2‑bench/issues/129

## 通俗解读任务模式
任务的 `evaluation_criteria` 字段（参考 [`src/tau2/data_model/tasks.py`](../src/tau2/data_model/tasks.py)）包含5个子字段：

| 字段 | 含义 | 何时会影响最终奖励 |
|---|---|---|
| `actions` | 一组动作列表，记录**一条**能够解决任务的工具调用参考轨迹。系统总会在全新的标准环境复现该动作，以此得到目标数据库终态。其他轨迹只要产出等价状态同样可以通过校验。 | 仅当 `reward_basis` 包含 `RewardType.ACTION`；此时智能体工具调用必须和列表一一匹配，该列表会被当作唯一可接受轨迹。 |
| `env_assertions` | 仿真结束后，在智能体运行后的环境上执行的断言列表。 | 仅当 `reward_basis` 包含 `RewardType.ENV_ASSERTION`。 |
| `communicate_info` | 智能体必须向用户说出的字符串集合（子串匹配）。 | 仅当 `reward_basis` 包含 `RewardType.COMMUNICATE`。 |
| `nl_assertions` | 由大模型做判断的自然语言断言。 | 仅当 `reward_basis` 包含 `RewardType.NL_ASSERTION`。（实验性功能 / 开发中） |
| `reward_basis` | 参与计算最终奖励的 `RewardType` 列表。 | 始终生效。默认值：`[DB, COMMUNICATE]`。 |

各奖励分量对应的评估器与校验逻辑：

| `RewardType` | 评估器 | 校验内容 |
|---|---|---|
| `DB` | `EnvironmentEvaluator` | 仿真结束后，智能体侧环境数据库哈希值，是否和目标数据库哈希一致；目标数据库 = 全新环境 + 复现 `evaluation_criteria.actions`。任意轨迹只要产出等价终态即可通过。 |
| `ENV_ASSERTION` | `EnvironmentEvaluator` | 智能体运行后的环境中，所有 `env_assertions` 断言全部成立。 |
| `COMMUNICATE` | `CommunicateEvaluator` | `communicate_info` 中每一条字符串都出现在智能体输出消息内。 |
| `NL_ASSERTION` | `NLAssertionsEvaluator` | 大模型判定 `nl_assertions` 每一条断言都成立。（开发中） |
| `ACTION` | `ActionEvaluator` | 智能体工具调用必须与 `actions` 中每一条动作匹配（由 `Action.compare_with_tool_call` 完成比对）。仅在确定该列表是全部可接受路径时才使用。 |

最终奖励为各分量相乘。例如 `reward_basis = [DB, COMMUNICATE]`，奖励 = `db_reward * communicate_reward`。
`EnvironmentEvaluator` 在后台读取 `actions`，用于构建目标环境；**不会直接拿智能体轨迹和参考动作做比对**。

## 为什么 `actions` 容易被误认为强制要求（实际并不是）
`evaluation_criteria.actions` 保存的是**某一条**任务求解工具调用序列，一般是任务作者或标注人员的求解路径。
并不代表它是唯一正确路径；很多任务有多条不同轨迹，都能产出完全等价的数据库终态。
举些例子：查询用户信息可以放在查询预订记录之前或之后；部分只读查询，智能体可以选择跳过。

任务文件中保留参考轨迹的原因：对很多任务，直接写“在全新环境回放这组动作”，比手动逐条描述目标数据库状态要简单得多。

但调试界面会展示这份动作列表，看起来很像任务检查清单，很容易被误解为智能体必须执行的动作。
打分真正关心的两点：数据库最终状态是否匹配目标；是否输出指定文本。
智能体可以选择完全不同但逻辑正确的工具路径；甚至任务正确行为是拒绝操作时，智能体可以不调用任何工具，依旧拿到满分。

这是刻意设计：原始 τ‑bench 论文基于**结果**（`DB + COMMUNICATE`）打分，而不是看智能体是否照脚本执行。如果改成按 `actions` 校验，就无法和论文指标对齐，还会惩罚那些路径不同但解法正确的智能体。

## 实操示例：航空任务 1
来自 [`data/tau2/domains/airline/tasks.json`](../data/tau2/domains/airline/tasks.json)
```json
{
  "id": "1",
  "evaluation_criteria": {
    "actions": [
      {
        "action_id": "1_0",
        "name": "get_user_details",
        "arguments": { "user_id": "raj_sanchez_7340" }
      },
      {
        "action_id": "1_1",
        "name": "get_reservation_details",
        "arguments": { "reservation_id": "Q69X3R" }
      }
    ],
    "communicate_info": [],
    "nl_assertions": ["Agent should not approve the cancellation."],
    "reward_basis": ["DB", "COMMUNICATE"]
  }
}
```

实际含义：
- `reward_basis = ["DB", "COMMUNICATE"]` → 奖励 = `db_reward * communicate_reward`。
- `actions` 包含 `get_user_details`、`get_reservation_details`，二者均为**只读工具**（查看 `tools.py` 装饰器），在干净环境回放不会修改数据库。因此目标数据库哈希就等于环境初始哈希。
- 当智能体运行后的数据库哈希与初始哈希完全相同时，`db_reward = 1.0`。也就是智能体**不能写入修改数据库**。该任务正确行为是拒绝取消订单，正确智能体不会修改数据库，哈希匹配，拿到满分数据库奖励。
- `communicate_info = []` → 无需输出特定文本，`communicate_reward` 直接为 1.0。
- 配置了 `nl_assertions`，但 `reward_basis` 没有加入 `NL_ASSERTION`，该断言只会作为诊断信息输出，**不参与最终打分**。

最终效果：智能体只要礼貌拒绝请求，即可拿到满分 `1.0`，哪怕完全没有调用 `get_user_details` 和 `get_reservation_details`。
这就是该任务 τ‑bench 的打分逻辑。`actions` 记录了一条合法信息获取路径，但不是强制约束。智能体使用其他只读查询，或是直接跳过查询，同样判定正确。

## 如何查看动作匹配情况（仅用于调试）
即使 `reward_basis` 没有配置 `ACTION`，仍然可以运行 `ActionEvaluator` 做诊断分析：

- 命令行运行器默认使用 `EvaluationType.ALL_WITH_NL_ASSERTIONS`。每次仿真都会在 `RewardInfo` 填充 `action_checks`；`partial_action_reward` 统计参考动作命中数量。
`tau2 view` 命令可以查看（字段：`Partial Action Reward: m/n`），对应实现在 `src/tau2/metrics/agent_metrics.py`。
> ⚠️ 注意：该指标只衡量与**某一条参考轨迹**相似度，不等于正确性。哪怕 `partial_action_reward` 得 0/n，如果智能体通过别的工具序列正确完成任务，依旧是完全正确。

- 如果希望评估“智能体是否完整复现参考轨迹”，使用 `EvaluationType.ALL_IGNORE_BASIS` / `ALL_WITH_NL_ASSERTIONS_IGNORE_BASIS`。此时会忽略任务 `reward_basis`，把环境、动作、沟通（可选自然语言断言）合并为综合分数。同样，该分数衡量与参考路径相似度，不等价于任务正确性。参考 [`src/tau2/evaluator/evaluator.py`](../src/tau2/evaluator/evaluator.py)。

- `RewardInfo` 的 `partial_action_reward` 还区分了读工具 `ToolType.READ` /写工具 `ToolType.WRITE` 的匹配率，方便排查“数据库校验通过，但完全没有执行写操作”这类现象。

官方排行榜分数严格使用任务自身的 `reward_basis`，不受上述调试选项影响。

## 什么时候启用 `RewardType.ACTION`
内置任务中，只有少量 `banking_knowledge` 任务（约100个任务中的9个），会把 `ACTION` 加入 `reward_basis`；这类任务评估的就是工具调用路径本身。
航空、零售、电信域**完全不启用 `ACTION`**，设计上只看最终状态。

如果你自定义领域和任务，想要强制约束智能体工具调用必须和参考动作保持一致，就可以在 `reward_basis` 中加入 `RewardType.ACTION`。
注意：一旦开启，`actions` 就从“参考轨迹”升级为“唯一合法轨迹”。只有你已经枚举全部合法解法，或者任务确实仅有一条可行路径，才适合开启。

## 参考资源
- 模式定义：[`src/tau2/data_model/tasks.py`](../src/tau2/data_model/tasks.py)
- 评估器源码目录：[`src/tau2/evaluator/`](../src/tau2/evaluator/)
- 评估器架构总览：[`src/tau2/evaluator/AGENTS.md`](../src/tau2/evaluator/AGENTS.md)
- 奖励数据结构（`RewardInfo`、`ActionCheck`等）：[`src/tau2/data_model/simulation.py`](../src/tau2/data_model/simulation.py)
- 关于 `actions` 和 `reward_basis` 的讨论：[issue #224][issue‑224] 与 [RFC #129][issue‑129]
