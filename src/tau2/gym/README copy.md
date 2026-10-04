主要介绍 τ² Gym 如何把 τ-bench 的对话任务包装成兼容 Gymnasium API 的交互环境，并讲解两种交互模式：
- AgentGymEnv：调用者控制 Agent；用户由环境中的用户模拟器生成回复。可用于运行或评测 Agent，也可作为训练时的交互环境。
- UserGymEnv：调用者控制用户；Agent 由环境配置的模型自动运行。可用于从用户视角测试、人工评估对话，也可用于调试或研究用户模拟器。
Gym 模块提供的是交互环境接口，AgentGymEnv向外暴露agent侧可调用的接口，UserGymEnv向外暴露user侧调用接口

---
---

# Tau2 Gym
> 兼容Gymnasium的环境，用于在 τ‑bench 框架中评估对话智能体。该模块提供标准化Gym接口，允许你在受控仿真环境中分步运行智能体。

## 概述
Tau2 Gym 模块对标准的 `gym.Env` 接口进行扩展，实现对话智能体评估环境。它遵循强化学习环境的 [Gymnasium API 标准](https://gymnasium.farama.org/)。

## 使用方法
### 基础环境搭建
提供两个Gym环境：
1. **AgentGymEnv**（`TAU_BENCH_ENV_ID`）——扮演智能体，与模拟用户进行交互
2. **UserGymEnv**（`TAU_BENCH_USER_ENV_ID`）——扮演用户，与自动化智能体交互

```python
import gymnasium as gym
from tau2.gym import register_gym_agent, TAU_BENCH_ENV_ID, TAU_BENCH_USER_ENV_ID

# 注册环境（仅需执行一次）
register_gym_agent()

# 创建 AgentGymEnv —— 扮演智能体
domain = "mock"
task_id = "create_task_1"

agent_env = gym.make(TAU_BENCH_ENV_ID, domain=domain, task_id=task_id)

# 或者创建 UserGymEnv —— 扮演用户
user_env = gym.make(TAU_BENCH_USER_ENV_ID, domain=domain, task_id=task_id)

# AgentGymEnv 附加配置
agent_env = gym.make(
    TAU_BENCH_ENV_ID, 
    domain=domain, 
    task_id=task_id,
    solo_mode=True,
    user_llm="gpt-4",
    user_llm_args={"temperature": 0.7}
)

# UserGymEnv 附加配置
user_env = gym.make(
    TAU_BENCH_USER_ENV_ID,
    domain=domain,
    task_id=task_id,
    agent_llm="gpt-4o",
    agent_llm_args={"temperature": 0.7}
)
```

### 环境配置
`AgentGymEnv` 支持多种配置项，用于自定义仿真行为：

```python
from tau2.gym.gym_agent import AgentGymEnv

# 基础环境（普通模式，使用默认模拟用户）
env = AgentGymEnv(domain="retail", task_id="0")

# 独立模式 —— 智能体独立处理任务工单
env = AgentGymEnv(
    domain="retail", 
    task_id="0",
    solo_mode=True
)

# 自定义模拟用户的大模型配置
env = AgentGymEnv(
    domain="retail", 
    task_id="0",
    user_llm="gpt-4",
    user_llm_args={"temperature": 0.7, "max_tokens": 1000}
)

# 组合配置
env = AgentGymEnv(
    domain="telecom",
    task_id="[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]", 
    solo_mode=False,
    user_llm="claude-3-sonnet",
    user_llm_args={"temperature": 0.5}
)
```

#### 配置参数
**`solo_mode`（布尔，可选）**
- **默认值**：`False`
- **说明**：当为 `True`，智能体独立处理任务工单，无用户交互。当为 `False`，智能体与模拟用户交互。
- **使用场景**：`True` 用于独立问题求解场景；`False` 用于对话交互场景。

**`user_llm`（字符串，可选）**
- **默认值**：使用系统默认的用户侧大模型
- **说明**：指定模拟用户所使用的大模型（仅当 `solo_mode=False` 生效）。
- **示例**：`"gpt-4"`、`"claude-3-sonnet"`、`"gpt-3.5-turbo"`

**`user_llm_args`（字典，可选）**
- **默认值**：使用系统默认大模型参数
- **说明**：传给模拟用户大模型的附加参数（仅当 `solo_mode=False` 生效）。
- **常用参数**：`temperature`、`max_tokens`、`top_p`、`frequency_penalty` 等。

### 模式对比
#### 普通模式（`solo_mode=False`）
- **交互逻辑**：智能体与模拟用户对话
- **适用场景**：对话场景、客服业务、交互式问题求解
- **任务输入**：携带人设与指令的用户场景
- **示例**：客服人员帮助用户预订机票

```python
env = AgentGymEnv(domain="airline", task_id="0", solo_mode=False)
observation, info = env.reset()
# 观测包含用户消息与回复
action = "您好！我可以帮您预订机票。您打算去哪里旅行？"
observation, reward, terminated, truncated, info = env.step(action)
```

#### 独立模式（`solo_mode=True`）
- **交互逻辑**：智能体独立处理任务工单（无模拟用户）
- **适用场景**：技术故障排查、独立问题求解、工单处理
- **任务输入**：任务配置中定义的工单（不会出现在初始观测中）
- **初始观测**：空 / None（无对话历史）
- **示例**：IT支持智能体根据工单排查网络故障

```python
env = AgentGymEnv(domain="telecom", task_id="[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]", solo_mode=True)
observation, info = env.reset()
# 观测包含任务工单以及初始上下文
action = "check_network_status(user_id='user_123')"
observation, reward, terminated, truncated, info = env.step(action)
```

### 智能体简单交互示例
#### 普通模式示例
```python
# 初始化普通模式环境（默认）
env = AgentGymEnv(domain="mock", task_id="create_task_1", solo_mode=False)
observation, info = env.reset()
print(f"初始观测: {observation}")

# 从info中读取可用工具与策略
print(f"可用工具数量: {len(info['tools'])}")
print(f"智能体策略: {info['policy'][:100]}...")

# 分步执行智能体动作
action = "您好！我来帮您处理您的请求。"
observation, reward, terminated, truncated, info = env.step(action)

print(f"观测: {observation}")
print(f"奖励: {reward}")
print(f"是否终止: {terminated}")

# 调用工具
next_action = "create_task(user_id='user_1', title='Important Meeting')"
observation, reward, terminated, truncated, info = env.step(next_action)
print(f"观测: {observation}")
print(f"本步奖励: {reward}")
```

#### 独立模式示例
```python
# 初始化独立模式环境
env = AgentGymEnv(domain="telecom", task_id="[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]", solo_mode=True)
observation, info = env.reset()
print(f"初始观测: {observation}")  # solo模式下为空/None

# 在独立模式中，智能体独立处理工单
# 工单信息从任务定义获取，不在观测内
action = "get_user_details(user_id='customer_123')"
observation, reward, terminated, truncated, info = env.step(action)

print(f"获取用户详情后: {observation}")

# 继续故障排查
next_action = "check_mobile_data_usage(user_id='customer_123')"
observation, reward, terminated, truncated, info = env.step(next_action)
print(f"流量查询结果: {observation}")
```

#### 自定义大模型配置示例
```python
# 为用户模拟器指定大模型
env = AgentGymEnv(
    domain="airline", 
    task_id="0",
    solo_mode=False,
    user_llm="gpt-4",
    user_llm_args={
        "temperature": 0.8,
        "max_tokens": 500,
        "top_p": 0.9
    }
)

observation, info = env.reset()
# 用户模拟器将使用上述参数的GPT‑4
action = "很高兴帮您查询预订机票。您打算去哪里？"
observation, reward, terminated, truncated, info = env.step(action)
```

### UserGymEnv —— 扮演用户
使用 `UserGymEnv`，你可以控制用户侧动作，由自动化的LLM智能体给出回复。适用场景：
- 从用户视角测试智能体行为
- 人工评估智能体性能
- 调试对话流程
- 通过强化学习训练用户模拟器

#### UserGymEnv基础示例
```python
from tau2.gym.gym_agent import UserGymEnv

# 创建环境 —— 你控制用户，智能体自动运行
env = UserGymEnv(domain="mock", task_id="create_task_1")
observation, info = env.reset()

# 观测返回智能体的开场白
print(f"智能体输出: {observation}")
# 输出: "assistant: Hello! How can I help you today?"

# 你作为用户进行回复
action = "I need to create a new task"
observation, reward, terminated, truncated, info = env.step(action)

# 观测返回智能体回复
print(f"智能体输出: {observation}")
# 输出: "assistant: I can help you create a task. What would you like to name it?"

# 继续扮演用户对话
action = "Call it 'Important Meeting'"
observation, reward, terminated, truncated, info = env.step(action)
```

#### 使用自定义智能体大模型的UserGymEnv
```python
# 为自动化智能体指定大模型
env = UserGymEnv(
    domain="airline",
    task_id="0",
    agent_llm="gpt-4o",
    agent_llm_args={
        "temperature": 0.7,
        "max_tokens": 1000,
    }
)

observation, info = env.reset()
# 智能体会使用上述参数的GPT‑4o
print(f"智能体: {observation}")

# 扮演用户
action = "I need to book a flight from NYC to LAX"
observation, reward, terminated, truncated, info = env.step(action)
print(f"智能体: {observation}")
```

#### UserGymEnv配置
**`agent_llm`（字符串，可选）**
- **默认值**：使用系统默认的智能体大模型
- **说明**：指定自动化智能体使用的大模型
- **示例**：`"gpt-4o"`、`"claude-3-sonnet"`、`"gpt-4"`

**`agent_llm_args`（字典，可选）**
- **默认值**：使用系统默认大模型参数
- **说明**：传给智能体大模型的附加参数
- **常用参数**：`temperature`、`max_tokens`、`top_p` 等。

**`all_messages_as_observation`（布尔，可选）**
- **默认值**：`False`
- **说明**：`False` 时仅展示智能体回复；`True` 返回完整对话历史。

>注意：UserGymEnv **不支持** `solo_mode`，因为独立模式不存在用户交互。

### info 字典
`reset()` 和 `step()` 都会返回 `info` 字典，保存环境重要上下文信息：

- **`tools`**：当前领域中智能体可调用的工具/动作列表
- **`policy`**：定义智能体行为与约束的策略字符串
- **`simulation_run`**：当前仿真状态的JSON表示（当可用时）

```python
# 读取info内容示例
observation, info = env.reset()

# 查看可用工具
for tool in info['tools']:
    print(f"工具名: {tool.name}")
    print(f"描述: {tool.description}")
    print(f"参数: {tool.parameters}")

# 查看智能体策略
print(f"智能体必须遵循的策略: {info['policy']}")
```

### Action 动作输入格式
#### 工具调用
对于和环境交互的工具调用，可以使用JSON格式或者函数调用格式：

**JSON格式：**
```python
# JSON格式工具调用
action = '{"name": "search_flights", "arguments": {"origin": "NYC", "destination": "LAX"}}'
observation, reward, terminated, truncated, info = env.step(action)
```

**函数调用格式：**
```python
# 带关键字参数的函数式调用
action = "search_flights(origin='NYC', destination='LAX')"
observation, reward, terminated, truncated, info = env.step(action)

# 另一个参数示例
action = "create_task(user_id='user_1', title='Important Meeting', priority='high')"
observation, reward, terminated, truncated, info = env.step(action)
```

#### 向用户输出消息
和用户进行对话（非工具动作），直接传入普通字符串：
```python
# 给用户的纯文本消息
action = "您好！我可以帮您处理请求。"
observation, reward, terminated, truncated, info = env.step(action)

action = "我了解您需要预订机票，我帮您查询可用选项。"
observation, reward, terminated, truncated, info = env.step(action)
```

### Observation 观测格式
`reset()` 与 `step()` 返回的 `observation` 是字符串格式的对话历史。每条消息格式为 `"role: content"`，多条消息之间换行分隔。

>注意：独立模式（`solo_mode=True`）下初始观测为空（None），不存在用户交互。任务工单信息从任务定义获取。后续观测展示智能体动作与工具调用的返回结果，没有用户消息。

#### 消息类型
**用户消息：**
- 纯文本：`"user: Hello, I need help booking a flight"`
- 注意：观测中看不到用户的工具调用

**智能体消息：**
- 纯文本：`"assistant: I'll help you book a flight. Let me search for available options."`
- 工具调用：`"assistant: search_flights(origin='NYC', destination='LAX')"`

**工具返回结果：**
- 工具调用结果：`"tool: {\"name\": \"search_flights\", \"arguments\": {\"origin\": \"NYC\", \"destination\": \"LAX\"}, \"result\": \"Found 3 flights...\"}"`
- 工具返回结果以JSON格式，角色为 `tool`

#### 观测输出示例
```python
observation, info = env.reset()
print(observation)
# 输出示例：
# user: Hello, I need help booking a flight from New York to Los Angeles
# assistant: I'll help you book a flight. Let me search for available options.
# assistant: search_flights(origin='NYC', destination='LAX')
# tool: {"name": "search_flights", "arguments": {"origin": "NYC", "destination": "LAX"}, "result": "Found 3 flights: Flight 123, Flight 456, Flight 789"}
```

观测字符串包含完整对话上下文，便于理解交互当前状态，据此规划下一步动作。

## 架构与线程模型
### 概述
Tau2 Gym环境采用**双线程架构**，协调外部Gym接口与内部仿真系统。对于开发、调试线程相关问题，理解这套架构十分重要。

### 两大核心组件
1. **GymAgent** — 特殊智能体，不由自身做决策，可以由外部代码单步控制。
2. **AgentGymEnv** — Gym环境封装层，管理仿真生命周期与线程协同。

### 线程架构
环境同时运行两个线程：
1. **主线程（Gym接口）**
    - 处理外部的 `reset()` 和 `step()` 调用
    - 通过 `set_action()` 将动作传给智能体
    - 返回观测值与奖励

2. **调度器线程（仿真逻辑）**
    - 运行Tau2仿真主循环
    - 处理智能体与用户之间消息流转
    - 执行工具调用、更新环境状态
    - 后台守护线程运行

### 同步机制
线程之间依靠两个核心同步原语协同工作：

#### 锁（`self._lock`）
互斥锁，防止多线程访问共享数据时发生竞态条件：
```python
with self._lock:
    # 同一时刻，仅有一个线程可以执行该段代码
    self._observation = deepcopy(state.messages)
    self._next_action = action_msg
```

受保护变量包括：
- `self._next_action` — 外部代码传入的动作
- `self._observation` — 当前对话历史
- `self._agent_turn_finished` — 同步事件状态

#### 事件对象（`self._agent_turn_finished`）
线程事件，在线程之间充当信号标记：
- **`.clear()`** — 将事件置为false，线程调用 `.wait()` 将阻塞
- **`.set()`** — 将事件置为true，阻塞的线程可以继续执行
- **`.wait()`** — 阻塞，直到事件被置为true
- **`.is_set()`** — 返回事件状态，true / false

### 观测‑动作生命周期
典型step调用过程中的同步逻辑：

**阶段1：观测（调度器线程）**
```python
def generate_next_message(self, message, state):
    with self._lock:
        self._agent_turn_finished.clear()  # 信号标记：智能体正在处理
        state.messages.append(message)
        self._observation = deepcopy(state.messages)  # 更新观测
    
    # 在此阻塞，等待外部传入动作
    self._agent_turn_finished.wait()
```

**阶段2：动作（主线程）**
```python
def step(self, action):
    action_msg = parse_action_string(action)
    self._agent.set_action(action_msg)  # 将动作给到智能体
    
    # set_action内部逻辑：
    with self._lock:
        self._next_action = action_msg
        self._agent_turn_finished.set()  # 信号标记：动作就绪，可以继续！
```

**阶段3：处理（调度器线程）**
```python
    # 从wait阻塞处恢复执行
    with self._lock:
        response_message = self._next_action  # 获取外部传入动作
        self._next_action = None  # 重置，准备下一轮
    
    return response_message, state
```

### 该架构的设计目的
- **外部可控**：通过Gym接口对智能体动作做单步控制
- **线程安全**：访问共享状态时不会出现竞态条件
- **非阻塞仿真**：调度器在等待动作时可以独立运行
- **标准Gym接口**：兼容现有的强化学习框架与工具

### 关键属性
- **`is_agent_turn`**：当轮到智能体、等待外部传入动作时返回 `True`
- **`observation`**：以列表形式返回当前完整对话历史
- **异常处理**：如果非智能体回合调用 `set_action()`，抛出 `RuntimeError`

### 线程安全注意事项
开发使用代码时：
- 访问共享状态时，务必使用锁
- 只有当 `is_agent_turn` 为 `True` 时，才可以调用 `set_action()`
- 调度器线程是守护线程，主线程退出时调度器线程随之终止
- wait循环中设置超时，周期性检测终止条件

