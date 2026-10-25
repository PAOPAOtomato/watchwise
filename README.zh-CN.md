# Watchwise

[English](README.md) | [简体中文](README.zh-CN.md)

**让现有摄像头拥有一位 AI 守卫。** 减少打扰人的告警，让决策有据可查，并通过人工纠正持续学习。

[![持续集成](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml/badge.svg)](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml)
![许可证：MIT](https://img.shields.io/badge/license-MIT-blue.svg)

**[探索 Watchwise →](https://watchwise-six-ebon.vercel.app/#experience)**

选择插画呈现的摄像头事件，查看证据，再看看一次纠正如何影响后续的匹配事件。
网站使用确定性的浏览器情景；其中的计数和置信度仅属于这个故事。无需摄像头或
API 密钥。提问面板使用预先编写的回答；安装后可用的对话式聊天功能仍在
[规划中](#路线图)。

摄像头不贵，注意力却很宝贵。Watchwise 基于开源网络视频录像机（NVR）
[Frigate](https://frigate.video)，判断哪些事件需要人工关注。不调用模型的初筛关卡
（gate）处理配置规则和可信的重复模式，视觉模型筛查尚未解决的事件，再由可调用
工具的智能体调查有歧义的情况。模型默认在本地运行，也可选择云端服务商。你可以
查看决策历史，并通过 Telegram 或反馈 API 教会记忆系统哪些情况属于正常活动。

> **真实环境测试已完成，时间为 2026 年 10 月 22–23 日。** 真实 USB 摄像头向
> Frigate 提供视频；Watchwise 处理事件、向手机发送 Telegram 告警卡片，并从
> 纠正中学习，在不再次调用模型的情况下拦下后续匹配事件。这验证了这一特定的
> 端到端流程，并不代表长期可靠性、检测准确率或对所有摄像头的兼容性。具体范围如下。

## 已验证的内容

真实测试使用 Windows 笔记本连接外置 USB 摄像头，通过 go2rtc 1.9.14 和 Tailscale
传输 RTSP 视频，接入使用 OpenVINO CPU 检测器的 Frigate 0.18.2，以及 Mosquitto
消息代理。Watchwise 运行在 Ubuntu CPU 虚拟机上；另一台配备 24 GB 显存
NVIDIA L4 的 GPU 虚拟机通过 Ollama 0.40.8 提供 `qwen3-vl:8b-instruct`（筛查）
和 `qwen3:14b`（调查）模型。这些是由操作者管理的云端虚拟机，因此此次测试并非
完全在本地场所内部署。记录中没有摄像头的具体型号，不能据此确认某一摄像头品牌
受到支持。

| 能力 | 证据与适用范围 |
|---|---|
| 实时接入与筛查 | 真实的室内人员事件通过 Frigate 的 MQTT 和 HTTP 接口进入 Watchwise。在录像尚不可用时，早期筛查使用了快照。 |
| 手机告警 | 真实手机上的 Telegram 收到了卡片，包括最初的“Analyzing…”（正在分析）状态，以及之后的判定或调查更新。 |
| 从纠正中学习 | Expected 和 Real 反馈影响了模式信任度。之后一个具有相同摄像头／标签／区域特征签名的事件被初筛关卡降级，没有调用模型，也没有新发手机卡片。重置记忆后，重复的 Expected 反馈也教会了系统一个新模式。 |
| 深入调查 | 有歧义的真实事件调用了工具型调查智能体。这次测试不代表已验证每个取证工具，也不代表每条结论都正确。 |
| 录像链接 | 卡片显示了 Clip 按钮。手机端播放仍未验证：当时手机无法接入 Frigate 所在的私有网络。 |
| Frigate 事件复核摘要 | 摘要解析、持久化、卡片文字和只提高告警级别的决策逻辑均已实现，并有自动化测试覆盖。真实环境中较长事件的摘要生成超出了模型上下文容量，因此该配置关闭了 Frigate GenAI。 |
| 其他平台与服务商 | 已提供配置预设和服务商适配器。自动化替身与硬件测试夹具不能证明每个列出的选项都在真实环境中兼容。 |

测试也暴露出尚未解决的限制：记忆无法区分特征签名相同的不同动作，筛查提示词
没有包含布防状态，而事件初期的快照无法描述整个事件。详见[已知限制](#已知限制)。
本项目不据此宣称任何真实环境准确率、误报下降百分比或延迟保证。

### 代码基线与可用范围

下方安装说明和功能描述对应已发布的 Python 实现
[`7e7c4aa`](https://github.com/PAOPAOtomato/watchwise/commit/7e7c4aa9f0a6c1c8ade631fd06209fafcaaecf3f)。
截至 10 月 24 日，另有尚未提交的实现工作，正在加入感知布防状态的筛查、针对近期
Real 反馈的关卡记忆否决机制，以及 Watchwise 自行生成的事件摘要。这些更改仍在
验证中，**不包含在**本次文档／网站发布或这里介绍的安装版本中。10 月 22–23 日的
真实测试也没有验证这些更改。能够区分行为的模式匹配仍属于后续工作。

## 试用路由模拟（无需摄像头、GPU 或 API 密钥）

使用 Python 3.11 或更高版本：

```bash
git clone https://github.com/PAOPAOtomato/watchwise.git && cd watchwise
python -m venv .venv
. .venv/bin/activate              # Windows：.venv\Scripts\activate
pip install -e ".[anthropic]"

watchwise demo                    # 通过完整级联管线模拟一天的事件
```

`watchwise demo` 将 60 个合成告警送入真实处理管线两次：第一次不使用已学习的模式，
第二次则在模拟审核者提供纠正后运行。它使用假 NVR 和替身模型。这些可复现结果
衡量的是路由行为，而非视觉理解能力或真实环境安防效果：

| 模拟的一天（`watchwise demo`） | 纠正前 | 纠正后 |
|---|---|---|
| 需要人工处理的告警 | 60 个中有 38 个 | 60 个中有 6 个 |
| 未调用模型即可解决的告警 | 22 | 54 |
| 被隐藏的、情景预设为真实威胁的告警 | 0 | 0 |

## 按你的硬件进行设置

在仓库根目录运行 CLI，以便读取 `.env` 和生成的配置：

```bash
watchwise doctor     # 只读检测硬件、服务和服务商密钥是否存在
watchwise init       # 查看变更计划后写入 config/watchwise.yml
```

启动所选模型服务，并拉取 `init` 列出的模型。在 `config/watchwise.yml` 中配置
Frigate 的 HTTP API 和 MQTT 连接，或使用 [.env.example](.env.example) 中的环境
变量。已有 Frigate 安装时，请确认目标检测已开启，且两边的 `mqtt.topic_prefix`
一致。然后运行：

```bash
watchwise check      # 调用各个已配置模型；云端调用可能产生费用
watchwise serve      # 连接 Frigate 并处理其复核事件
```

若配置不在默认路径，请为 `check`、`serve` 和 Telegram 命令添加 `--config <path>`。
`check` 检查筛查和工具调用是否可用；探测成功并不等于摄像头识别准确率通过测试。
即使安装后的对话界面尚未实现，已配置的 `chat` 模型角色仍可接受探测。

`doctor` 检测 NVIDIA、AMD、Apple Silicon 或 CPU 硬件，探测本地模型服务以及
Frigate／MQTT 是否可达，并检查 API 密钥是否存在而不显示其值。配置预设只是起点，
并非硬件认证：

| 配置预设 | 适用起点 | 模型 | 发送给云端服务商的模型数据 |
|---|---|---|---|
| `cpu-local` | 没有可用 GPU | 通过 Ollama 让所有已配置角色使用 `qwen3-vl:4b` | 使用本地服务时不发送 |
| `cpu-hybrid` | CPU 筛查搭配云端调查 | 本地 `qwen3-vl:2b`；云端推理模型 | 文字和工具结果，不发送筛查图像 |
| `nvidia` | NVIDIA，约 8–15 GB | 所有已配置角色使用 `qwen3-vl:8b-instruct` | 使用本地服务时不发送 |
| `nvidia-16gb` | 16 GB 级别（报告内存达到 15 GB 即可能选中），须注意下方限制 | `qwen3-vl:8b-instruct` 筛查，`qwen3:14b` 推理 | 使用本地服务时不发送 |
| `apple-silicon` | Mac，约 16 GB 以上统一内存 | 通过原生 Ollama 使用 `qwen3-vl:8b-instruct` | 使用本地服务时不发送 |
| `apple-silicon-8gb` | Mac，约 8 GB 统一内存 | `qwen3-vl:4b` | 使用本地服务时不发送 |
| `amd-rocm` | Linux 上受支持的 AMD GPU | 通过 Ollama ROCm 使用 `qwen3-vl:8b-instruct` | 使用本地服务时不发送 |
| `cloud` | 无需本地模型服务 | 所有已配置角色使用云端模型 | 筛查图像、文字和工具结果 |
| `demo` | 路由模拟 | 确定性替身 | 不发送 |

**`nvidia-16gb` 这个名称高估了 16 GB 显存能容纳的工作负载。** L4 测试中，配置的
筛查模型和调查模型合计占用超过 16 GB；测试使用的是 24 GB GPU。自动选择逻辑尚未
修正。在显存较小的 GPU 上，应采用更小模型或单模型配置，并在该机器上验证，不能
假定两个模型都能常驻。CPU 筛查也可能较慢，或耗尽输出预算。

预设位于 [`watchwise/hw/profiles.yml`](watchwise/hw/profiles.yml)。任一角色都可以
使用不同的模型或服务。重新运行 `init` 会展示差异，并备份被替换的配置；应用前请
检查差异，保留自己的定制设置。平台设置详见 [docs/hardware.md](docs/hardware.md)，
配置说明见 [config.example.yml](config.example.yml)。

云端预设包括 Anthropic、OpenAI、Google Gemini、Groq 和 OpenRouter；也可手动
配置其他兼容 OpenAI 的端点。请用 `watchwise check` 验证模型标识符和所需能力。
存在预设不代表每个服务商都通过了真实摄像头测试。

**为 Watchwise 使用专用密钥。** 在 `.env` 中以 `WATCHWISE_<PROVIDER>_API_KEY`
命名，例如 `WATCHWISE_OPENAI_API_KEY`。设置时，它优先于服务商的常规变量；如果
没有专用密钥，设置流程仍可能选择常规变量。请检查生成配置中的 `api_key_env`。
已配置的密钥缺失时，系统会报告该问题，而不会自动改用其他凭据。

表格仅描述模型请求的数据流向。将某个角色指向远程服务，意味着它的输入会发送到
那里。Telegram、Webhook 和 MQTT 输出有各自的数据路径；仅使用本地推理，并不
代表全部通知数据也留在本地。

### 使用 Docker（Frigate + MQTT + Ollama + Watchwise）

```bash
watchwise init --deploy          # 同时生成 Compose 组合选择和 Frigate 配置
# 编辑 deploy/frigate/config.yml：添加摄像头视频流并启用检测。
cd deploy
# 检查生成的配置后，启动选定服务：
docker compose up -d
```

`deploy/.env` 选择相应硬件的叠加配置文件（`docker-compose.nvidia.yml`、`.amd.yml`、
`.apple.yml` 等）。请在该叠加配置实际使用的模型服务上拉取所选模型；macOS 上的
Ollama 在系统中原生运行。Frigate 模板现已显式设置 `detect.enabled: true`；以前
生成的配置需要手动检查该设置。启动前请替换视频流占位符。真实测试中，Frigate／
MQTT 运行在 Docker 内，而 Watchwise 运行在 Python 环境中；此次测试并未验证每套
完整 Compose 叠加配置。

### 手机告警（Telegram）

1. 在 Telegram 中联系 [@BotFather](https://t.me/BotFather)，发送 `/newbot`，
   复制它提供的令牌。
2. 将令牌写入 `.env`：`WATCHWISE_TELEGRAM_BOT_TOKEN=<token>`。
3. 运行 `watchwise telegram link`，然后向自己的机器人发送 `/start`。只有已关联的
   聊天可以接收卡片或操作卡片按钮。
4. 运行 `watchwise telegram test` 发送示例卡片，再运行 `watchwise serve` 处理
   真实事件。修改配置后，请重启已经运行的服务。

卡片显示快照、摄像头与时间、理由、置信度和决策阶段。筛查或调查结束后，卡片会
原位更新。置信度是模型或策略给出的评分，并非经过测量的正确概率。

| 按钮 | 作用 |
|---|---|
| ✅ Expected（符合预期） | 记录正常活动反馈并强化模式；要拦下后续事件，还须满足下文的纠正次数、相似度和信任度阈值 |
| 🚨 Real（确需关注） | 记录可疑活动反馈，并降低曾影响该事件的误报模式的信任度 |
| 🔍 Look closer（深入调查） | 尚未开展完整调查时，请求立即调查 |
| 🔇 Snooze（暂时静音） | 默认将该摄像头的卡片静音一小时；事件处理和记录继续进行 |
| ❓ Why（为什么） | 解释已保存的决策、匹配记忆和可用的调查证据 |
| ↩️ Undo（撤销） | 撤销纠正及其对次数和信任度的影响；已合并的描述文字可能保留 |
| ▶️ Clip（录像） | 配置 `nvr.web_url` 后，打开 Frigate 的事件复核页面 |

将 `nvr.web_url`（生成配置中也可使用 `FRIGATE_WEB_URL`）设为手机能够访问的
Frigate 网页地址。它与处理后端 API 请求的 `nvr.base_url` 不同。手机仍需要网络
访问权限，以及 Frigate 要求的登录凭据；这个链接不会代理视频，也不会绕过认证。

Telegram 会中转卡片文字、已配置的 Clip URL，以及快照；将
`notify.telegram.send_images` 设为 `false` 可关闭快照发送。长轮询无需公开入站
端点。默认只有 `alerted` 和 `escalated_human` 决策会创建卡片；在分析期间已经
发送的卡片，之后可以更新为降级结果。

## 工作原理

```text
RTSP -> [ Frigate ] -> 运动／目标检测 -> 复核事件
                            | MQTT <prefix>/reviews
                            v
GATE          配置规则、去重、可信的重复模式
                            | 尚未解决
                            v
TIER 2        视觉模型：benign / suspicious / uncertain + 置信度
                            | 尚未解决或不确定
                            v
TIER 3        异步工具调用调查：
              历史告警、摄像头健康、其他摄像头、值勤安排、录像
                            v
POLICY        suppress / downgrade / alert / escalate
                            v
HUMAN REVIEW  人工纠正 -> 模式记忆 -> 后续匹配决策
```

* **按需调用模型。** 初筛关卡能解决的事件不需要调用模型；未解决的事件先接受
  筛查，必要时再进行调查。
* **提前通知。** Frigate 新建复核事件时即开始筛查。默认情况下，筛查耗时较长会
  在 10 秒等待预算后发送“Analyzing…”卡片；离家布防和夜间布防使用零秒预算，
  但派发等待时间最低为 0.5 秒。这些是调度设置，不是送达保证。后续报告只有出现
  标签或区域变化时才会再次筛查。
* **模型预热。** `service.warm_up_models` 在启动时加载本地模型，让筛查
  模型保持已加载状态。服务启动和内存压力仍可能引起延迟。
* **故障处理。** 筛查错误、超时、拒绝回答或无法解析的回复会转为 uncertain；
  无法解决的调查会交给人工复核。这不能防止错误的正常活动判定或错误的关卡匹配。
* **可审计的决策。** 被拦下的事件仍有记录。决策包括理由、置信度、匹配记忆，
  以及实际开展过调查时的调查轨迹。模式保留纠正次数及同意／不同意次数。
* **只读集成 Frigate。** Watchwise 读取 Frigate 的 HTTP API 和 MQTT，写入自己的
  数据库，并提供自己的 API。移除 Watchwise 不会改变 Frigate 的录像或配置。

### 学习阈值与保护措施

默认情况下，误报模式至少需要 **3 次纠正**、**0.86 的匹配相似度**和 **0.70 的
信任度**，才能在初筛关卡解决事件。信任度计算为
`(agreements + 1) / (agreements + disagreements + 2)`；仅有纠正次数还不够。
Real 反馈可以降低信任度，Undo 则撤销其已记录的影响。

关卡置信度为“信任度 × 相似度”，上限为 `0.95`。策略只有在置信度达到 `0.85`
或更高时才会完全抑制已学习的匹配事件；符合关卡条件但分数较低的事件会降级为
检测记录。两种结果通常都不会新发手机卡片，但事件仍可检查。配置为始终告警的
标签优先于记忆和去重规则。

模式以摄像头、目标标签和区域为键。关卡不查看画面、不识别人，也不会学习某人的
时间安排。一次纠正不会修改值勤安排。时间衰减影响搜索排序，但**不会**使模式
到期，也不会单独取消模式拦下告警的资格。

### 完整事件描述

当 Frigate 提供 GenAI 事件复核摘要时，Watchwise 可以在卡片和决策历史中附上
“What happened”（发生了什么）。`decision.summary_can_raise: true` 为默认值；
此时，可疑或严重级别的摘要会将已拦下的事件提高为告警，但不能降低决策级别。
该集成已实现，并用替身测试过。真实配置中的较长事件摘要因图像输入超过模型上下文
而生成失败。Watchwise 自己的完整事件摘要器正在单独实现，不包含在上述已发布
基线中；该基线的早期判定可能只描述一张快照。

### 模型层

同一个与服务商无关的接口（`watchwise/llm/`）服务于各个已配置角色：

| 服务商类型 | 覆盖范围 |
|---|---|
| `ollama` | Ollama 原生 API，显式设置上下文长度 |
| `openai_compat` | 兼容 OpenAI 的 API，包括配置的 OpenAI、Groq、Gemini、OpenRouter、LM Studio、llama.cpp、vLLM 和 OpenVINO Model Server 端点 |
| `anthropic` | 通过官方 SDK 使用 Claude，支持提示缓存和拒绝回答时的回退机制 |
| `stub` | 用于测试和路由模拟的确定性替身 |

模型层可以恢复以普通文本形式输出的、受支持的工具调用，分离 `<think>` 区块，并将
格式错误的工具参数作为错误反馈，而不会执行它们。筛查器和 `watchwise check` 会
报告输出截断、空回复以及无法解析的回复。

## 接口

| 端点 | 用途 |
|---|---|
| `GET /health` | 各层状态、模型预热、队列深度、NVR 可达性 |
| `GET /metrics` | 路由／成本计数与已复核事件的一致性统计；不是无偏的准确率基准 |
| `POST /alerts/simulate` | 将一个合成告警送入真实级联管线 |
| `POST /alerts/ingest` | 通过 HTTP 接收 Frigate 复核载荷，替代 MQTT 接入 |
| `GET /alerts/{review_id}` | 获取事件及其决策历史 |
| `GET /investigations/{id}` | 获取结论背后的调查轨迹 |
| `POST /feedback` | 向学习循环提交人工判定 |
| `GET /patterns`, `/patterns/search` | 检查长期记忆 |
| `GET /tools`, `POST /tools/{name}` | 列出和调用取证工具 |
| `GET` / `POST /arming` | 查看或改变布防状态 |

启用 `notify.mqtt.enabled` 后，判定会发布到 `watchwise/verdicts` 和
`watchwise/<camera>/verdict`。`notify.webhook_url` 会以 JSON 格式向指定目标
POST 判定，例如用于接入 Home Assistant。

取证工具还通过 **MCP** 提供：智能体可在进程内使用，也可通过 stdio 服务
`watchwise-mcp` 提供给 Claude Desktop、Claude Code、Cursor 或其他 MCP 客户端；
后者需要 `pip install -e ".[mcp]"`。这些工具并不构成安装后可用的 Watchwise
聊天界面。

## 路线图

| 里程碑 | 状态 |
|---|---|
| 级联管线、调查智能体、取证／MCP 工具、学习循环 | 已实现；真实摄像头 → 手机 → 纠正 → 后续关卡决策这一核心流程已验证 |
| 模型适配器、硬件预设、`doctor` / `init` / `check` | 已实现；仍需更广泛的硬件／服务商验证 |
| Telegram 卡片、教学按钮、Undo、Snooze、Why、Look closer | 已实现并由自动化测试覆盖；真实送达及 Expected／Real 学习已验证 |
| Clip 链接和 Frigate 摘要接入 | 已实现；手机播放仍未验证，真实环境中较长事件的摘要生成失败 |
| 感知布防状态的筛查，以及针对近期 Real 反馈的关卡记忆否决机制 | 本地实现仍在进行中；不属于上述已发布安装版本 |
| 可区分行为的记忆匹配 | 规划中 |
| Watchwise 完整事件摘要 | 本地实现仍在进行中；尚未发布，也未经真实环境验证 |
| 每日摘要与复核抽查 | 规划中 |
| Ask Watchwise：安装后可用的对话式聊天，配置更改需确认 | 规划中；已配置的模型角色和网站提问回答不等于这个界面 |
| 设置助手、通过 SKILL.md 为智能体提供设置流程、更广泛的硬件实测基准 | 规划中 |
| [交互式产品网站](https://watchwise-six-ebon.vercel.app/#experience) | 已可访问；使用插画情景和浏览器本地状态 |

## 已知限制

以下限制适用于[代码基线与可用范围](#代码基线与可用范围)中标明的已发布基线。

* 真实测试是短时间、单摄像头的室内试验，不能证明长期可靠性、多摄像头性能或
  一般检测准确率。`watchwise demo` 中的替身模型无法理解真实画面。
* 记忆可能拦下与正常事件具有相同特征签名的真实异常事件。不同手势或行为不会
  被区分，Real 反馈也不是阻止后续抑制的硬性否决。信任度阈值、复核记录和 Undo
  有助于纠正，但无法消除这一风险。
* 筛查器不知道布防状态。即使处于离家布防，看起来普通的室内人员活动也可能被
  判为正常。布防状态影响临时通知的时间，也可被调查工具读取，但当前不会修正
  筛查提示词。没有布防状态文件时，默认状态为 `armed_away`。
* 早期筛查可能只用一张快照。结束报告若没有标签／区域变化，不会触发再次筛查。
  无法保证总有完整事件摘要，摄像头位置或画面模糊也会影响判定。
* 真实模型的调查可能较慢。预热有助于减少冷启动，但不能保证模型常驻或响应时间；
  `nvidia-16gb` 的双模型配置可能超过其名义显存档位。
* CPU 筛查和思考型模型可能耗尽时间或输出预算。
  `watchwise check --role screener` 可以帮助诊断所选机器上的情况。
* Telegram 送达需要网络连接。录像链接还要求手机能够访问 Frigate 并通过认证；
  这一播放检查尚未完成。
* 上述基线不提供安装后的聊天、自动修改时间安排或 Watchwise 自行生成的完整
  事件摘要。摘要工作仍在本地，尚未发布。网站提问面板不会向模型发送输入。

## 开发

```bash
pip install -e ".[dev,anthropic,mcp,agent]"
pytest -q
```

Python 自动化测试使用假的 Frigate、模型和 Telegram Bot API 服务（均为真实 ASGI
应用），并使用硬件检测夹具，无需 GPU、摄像头、机器人或 API 密钥。测试覆盖路由、
反馈影响与 Undo、通知更新、预热与超时行为、录像 URL 和摘要提级。这些检查补充
真实环境观察，不能取代真实测试。

前端与 Python 应用相互独立：

```bash
cd website
npm ci
npm run build
npx playwright install chromium
npm test
```

使用与 CI 一致的 Node.js 24。预览、浏览器检查和现有 Vercel 部署维护流程见
[website/README.md](website/README.md)。

```text
watchwise/    Python 包：tier2/、tier3/、tools/、memory/、llm/、hw/、
              setup/、notify/、cli.py
tests/        按层组织的 Python 测试
deploy/       Docker Compose 基础配置、硬件叠加配置、Frigate 模板
docs/         硬件与模型指南
website/      静态产品体验、本地资源、浏览器测试
```

## 致谢

Watchwise 提供智能体层，底层基础来自其他项目：

* [Frigate](https://github.com/blakeblackshear/frigate)（MIT，© Frigate, Inc.）：
  视频流接入、运动／目标检测、录像和事件数据库。Watchwise 通过其公开 HTTP API
  和 MQTT 与之通信。
* [Ollama](https://ollama.com) 及其提供服务的开放权重模型。
* [Model Context Protocol](https://modelcontextprotocol.io) 和 Anthropic SDK。
* 网站内置的 Manrope 字体遵循
  [SIL Open Font License](website/public/licenses/manrope.txt)，并使用原创 SVG
  插画，而非实时摄像头录像。

## 许可证

MIT。见 [LICENSE](LICENSE)。
