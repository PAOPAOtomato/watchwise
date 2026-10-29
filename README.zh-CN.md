# Watchwise

[English](README.md) | [简体中文](README.zh-CN.md)

**让现有摄像头拥有一位 AI 守卫。** 减少打扰人的告警，让决策有据可查，并通过人工纠正持续学习。

[![持续集成](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml/badge.svg)](https://github.com/PAOPAOtomato/watchwise/actions/workflows/ci.yml)
![许可证：MIT](https://img.shields.io/badge/license-MIT-blue.svg)

**[探索 Watchwise →](https://watchwise-six-ebon.vercel.app/#experience)**

选择插画呈现的摄像头事件，查看证据，再看看一次纠正如何影响后续的匹配事件。
网站使用确定性的浏览器情景；其中的计数和置信度仅属于这个故事。无需摄像头或
API 密钥。提问面板使用预先编写的回答；安装后可在 Telegram 中提问（见
[询问 Watchwise](#询问-watchwise)）。

摄像头不贵，注意力却很宝贵。Watchwise 基于开源网络视频录像机（NVR）
[Frigate](https://frigate.video)，判断哪些事件需要人工关注。不调用模型的初筛关卡
（gate）处理配置规则和可信的重复模式，视觉模型筛查尚未解决的事件，再由可调用
工具的智能体调查有歧义的情况。模型默认在本地运行，也可选择云端服务商。在
Telegram 中，你会收到能说明理由的告警卡片，一键即可教会记忆系统，还可以询问
发生了什么；反馈 API 也提供同样的纠正方式。

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

若配置不在默认路径，请为 `check`、`serve` 和 Telegram 命令添加 `--config <path>`；
Telegram 中的 `/setup` 会把回答保存到启动 `serve` 时使用的配置文件。`check` 检查
筛查、工具调用，以及负责回答问题的 `chat` 角色是否可用；探测成功并不等于摄像头
识别准确率通过测试。

`doctor` 检测 NVIDIA、AMD、Apple Silicon 或 CPU 硬件，探测本地模型服务以及
Frigate／MQTT 是否可达，并检查 API 密钥是否存在而不显示其值。配置预设只是起点，
并非硬件认证：

| 配置预设 | 适用起点 | 模型 | 发送给云端服务商的模型数据 |
|---|---|---|---|
| `cpu-local` | 没有可用 GPU | 通过 Ollama 让所有已配置角色使用 `qwen3-vl:4b` | 使用本地服务时不发送 |
| `cpu-hybrid` | CPU 筛查搭配云端调查和聊天 | 本地 `qwen3-vl:2b`；云端推理模型 | 文字和工具结果（包括聊天提问），不发送筛查图像 |
| `nvidia` | NVIDIA，约 8–15 GB | 所有已配置角色使用 `qwen3-vl:8b-instruct` | 使用本地服务时不发送 |
| `nvidia-16gb` | 16 GB 级别（报告内存达到 15 GB 即可能选中） | 所有已配置角色使用 `qwen3-vl:8b-instruct`，16k 上下文 | 使用本地服务时不发送 |
| `apple-silicon` | Mac，约 16 GB 以上统一内存 | 通过原生 Ollama 使用 `qwen3-vl:8b-instruct` | 使用本地服务时不发送 |
| `apple-silicon-8gb` | Mac，约 8 GB 统一内存 | `qwen3-vl:4b` | 使用本地服务时不发送 |
| `amd-rocm` | Linux 上受支持的 AMD GPU | 通过 Ollama ROCm 使用 `qwen3-vl:8b-instruct` | 使用本地服务时不发送 |
| `cloud` | 无需本地模型服务 | 所有已配置角色使用云端模型 | 筛查图像、文字、工具结果和聊天提问 |
| `demo` | 路由模拟 | 确定性替身 | 不发送 |

**每台机器一个模型。** GPU 和 Apple Silicon 预设让所有角色（筛查、调查、聊天）
共用一个视觉模型，因此只需常驻一个模型；请在自己的机器上验证其上下文长度和
内存占用。这些预设还会让模型核对已学习的常规活动（`gate.memory_mode:
model_checked`），并生成完整事件摘要。CPU 预设保留不调用模型的初筛关卡，不生成
摘要，而且可能较慢或耗尽输出预算。

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
Ollama 在系统中原生运行。Frigate 模板设置了 `detect.enabled: true`；请在自己的
配置中检查该设置。启动前请替换视频流占位符。

### 手机告警（Telegram）

1. 在 Telegram 中联系 [@BotFather](https://t.me/BotFather)，发送 `/newbot`，
   复制它提供的令牌。
2. 将令牌写入 `.env`：`WATCHWISE_TELEGRAM_BOT_TOKEN=<token>`。
3. 运行 `watchwise telegram link`，然后向自己的机器人发送 `/start`。只有已关联的
   聊天可以接收卡片、操作卡片按钮或提问。
4. 运行 `watchwise telegram test` 发送示例卡片，再运行 `watchwise serve` 处理
   真实事件。手动编辑配置后，请重启已经运行的服务；在 Telegram 中做出的更改会
   立即生效。

然后向机器人发送 `/setup`。它会询问每个摄像头拍摄的是什么（有多个摄像头时，
还会询问它所在的房间）、谁住在这里、哪些情况总要提醒你，以及没有设置布防状态时
默认采用哪种状态。它会展示变更计划，只有在你点击 ✅ Apply 后，才会先备份文件再
写入配置；↩️ Undo 可恢复旧文件。正在运行的服务会立即采用这些回答。之后它还会
提供向所有摄像头共用的人脸库添加照片的操作，以及识别检查：从摄像头前走过，
它会告诉你 Frigate 是否认出了你。

| 命令 | 作用 |
|---|---|
| `/setup` | 描述摄像头、房间、住户、总要提醒的规则和默认布防状态 |
| `/arm` | 查看或切换布防：Home、Away、Night、Off（也可用 `/arm away`） |
| `/face` | 列出共用人脸库中各住户的照片数量、少于 5 张时的提示，以及库中的其他名字 |
| `/face <name> [camera]` | 在 Telegram 中预览摄像头当前画面；点击 ✅ Add 后，将照片写入所有摄像头共用的 Frigate 人脸库 |
| `/new` | 与 Ask Watchwise 开始新的对话 |
| `/cancel` | 停止进行中的 `/setup` |
| `/help` | 按钮和命令说明 |

卡片显示快照、摄像头与时间、理由、置信度和决策阶段。筛查或调查结束后，卡片会
原位更新。置信度是模型或策略给出的评分，并非经过测量的正确概率。同一房间的
摄像头每次事件共用一张卡片：该房间另一个摄像头的事件会更新这张卡片，而不会
再次提醒。一张卡片只接收该房间首个事件之后五分钟内的事件；更晚的事件、更严重
的事件，或你已回复卡片后才出现的事件，会获得自己的卡片。

在 Away（离家）或 Night（夜间）布防时，如果调查将某个人员判为正常，卡片会说明
允许放行的依据：包含日期安排和时间段的值勤安排条目，或 Frigate 认出的住户。
匹配值勤安排只为调查提供背景，不会单独将人员判为正常。在安排好的来访期间，
Frigate 无法匹配的人脸会在被拦下事件的说明中注明，不会仅因此将事件提高为告警。

| 按钮 | 作用 |
|---|---|
| ✅ Expected（符合预期） | 记录正常活动反馈并强化模式；要拦下后续事件，还须满足下文的纠正次数、相似度和信任度阈值 |
| 🚨 Alert me（提醒我） | 记录可疑活动反馈（希望此类事件提醒你），并降低曾影响该事件的误报模式的信任度；随后询问“What was wrong?”（哪里不对？），你的回复会成为视觉模型要寻找的示例 |
| 🔍 Look closer（深入调查） | 尚未开展完整调查时，请求立即调查 |
| 🔇 Snooze（暂时静音） | 默认将该摄像头及其所在房间的卡片静音一小时；事件处理和记录继续进行 |
| ❓ Why（为什么） | 解释已保存的决策、匹配记忆和可用的调查证据 |
| 👤 That was &lt;resident&gt;（这是某位住户） | 在因未匹配人脸而提高为告警的卡片上，预览该事件的人脸裁剪图；点击 ✅ Add 后，将其加入 Frigate 的共用人脸库 |
| ↩️ Undo（撤销） | 撤销纠正及其对次数和信任度的影响；已合并的描述文字可能保留 |
| ▶️ Clip（录像） | 配置 `nvr.web_url` 后，打开 Frigate 的事件复核页面 |

在房间卡片上，✅、🚨 和 ↩️ 会作用于卡片上每个摄像头的事件，因此教会一个视角
也会教会另一个视角。

将 `nvr.web_url`（生成配置中也可使用 `FRIGATE_WEB_URL`）设为手机能够访问的
Frigate 网页地址。它与处理后端 API 请求的 `nvr.base_url` 不同。手机仍需要网络
访问权限，以及 Frigate 要求的登录凭据；这个链接不会代理视频，也不会绕过认证。

Telegram 会中转卡片文字、已配置的 Clip URL 和快照（将
`notify.telegram.send_images` 设为 `false` 可关闭快照发送），以及你的提问、
回答和你在 `/setup` 中输入的内容。长轮询无需公开入站端点。默认只有 `alerted` 和
`escalated_human` 决策会创建卡片；在分析期间已经发送的卡片，之后可以更新为降级
结果。

### 所有摄像头共用的人脸库

Frigate 为**所有摄像头维护同一个人脸库（Face Library）**。每个人的文件夹保存
人脸裁剪图，并将其特征取平均，得到所有摄像头用于比对的人脸嵌入向量
（face embedding）。用任意摄像头拍摄的清晰正面照片录入住户后，所有摄像头都能
使用这份录入信息。再逐步添加一些不同视角或光线下的清晰照片，有助于识别。
[Frigate 的指南](https://docs.frigate.video/configuration/face_recognition/#creating-a-robust-training-set)
建议以每人 5–10 张照片作为最低起点；照片质量很重要，录入不保证每个视角都能
匹配成功。请在 Frigate 中启用人脸识别，并通过 `/setup` 列出住户。

单独发送 `/face`，可查看每位住户的照片数量、少于 5 张时添加更多清晰照片的提示，
以及库中的其他名字。录入时，发送 `/face <name> [camera]`，在光线良好、画面中
只有自己的情况下正对摄像头，并检查当前画面的预览。点击 ✅ Add 会将该画面发送
到 Frigate 的注册 API，把人脸加入共用库；✖️ Cancel 不会更改人脸库。

因 Frigate 无法匹配人脸而提高为告警的卡片，会为最多 3 位已配置住户提供
**👤 That was &lt;resident&gt;** 按钮。点击名字后，可预览 Frigate 曾在该卡片事件中
尝试识别的一张可用人脸裁剪图；请确认它清楚地显示了正确的人。✅ Add 会调用
Frigate 的训练 API，将裁剪图移入该住户的人脸库文件夹；✖️ Cancel 不会添加任何
内容。如果 Frigate 已不再保留该事件的人脸裁剪图，请用 `/face` 重新拍照。
两种录入方式都会通过 Telegram 发送预览，除非将 `notify.telegram.send_images`
设为 `false`（此时只发送文字和按钮）。两者都只有在点击 ✅ Add 后才写入 Frigate。

### 询问 Watchwise

向机器人发送普通消息，或回复某张告警卡片来询问该事件：

* “今天厨房有什么动静吗？”：根据 Watchwise 的记录给出简短总结，必要时附上照片。
* “21:31 那次为什么没有提醒我？”：给出该决策、做出决策的阶段以及证据。
* “保洁员每周四下午 1–4 点来”：提议一条值勤安排规则。
* “删除保洁员规则”：提议删除该值勤安排规则。
* “包裹被取走时告诉我”：提议一项盯防（watch）。

`chat` 角色的模型通过查询来回答：Watchwise 自己的记录（它判定过的每个事件，
以及判定和摘要）和 Frigate 的事件列表。它自己不能更改任何东西。添加或删除值勤
安排规则、切换布防、将某个摄像头或房间静音，以及设置盯防，都会以带有 ✅ Confirm
和 ✖️ Cancel 按钮的提议形式出现。**只有点击 ✅ Confirm 才会执行提议的更改。**
输入“confirm”“yes”或“ok”不会更改任何内容；机器人会回复提示，要求使用按钮。
未确认的提议一小时后失效。

新值勤安排规则采用你的本地时区。任何规则都可以按名称删除；如果多条规则同名，
Watchwise 会询问你指的是哪一条。确认删除后，只会移除选中的规则。修改值勤安排
时会保留文件备份。规则告诉调查模块，在这些摄像头和时间段内出现人员是预期情况，
因此即使处于布防状态，调查也可以将来访判为正常，而不必询问你。规则本身不会
将人员判为正常；调查仍可能请求你处理。

连续 6 小时没有消息后，下次提问会开始新的对话。随时发送 `/new` 也可以重新开始。
对话较长时，最早的轮次会从模型提示中省略，为上下文预算腾出空间。你的提问、
回答及回答所附照片会经过 Telegram；如果 `chat` 角色使用云端服务商，其提示和
查询得到的文字／工具结果也会发送给该服务商。

盯防会在其指定的摄像头上，对每个 Frigate 事件检查一次，时间是在事件结束时：
视觉模型根据分布在整个事件中的几帧画面，判断是否清楚显示了你要求的情况。一旦
匹配，就提醒一次并结束该盯防；否则会持续到你选择的时间（默认 24 小时，最长
一周）。盯防也包括 Frigate 认出的住户。匹配后的卡片会写明
“👀 You asked to be told when …”（你要求在……时通知你）；满足你的要求，并不表示
该事件被判为可疑。

## 工作原理

```text
RTSP -> [ Frigate ] -> 运动／目标／人脸检测 -> 复核事件
                            | MQTT <prefix>/reviews
                            v
FACES         你在家时 Frigate 认出的住户：拦下
GATE          配置规则、去重、可信的重复模式
                            | 尚未解决
                            v
TIER 2        视觉模型：benign / suspicious / uncertain + 置信度，
              并获知布防状态、摄像头说明和你的总要提醒规则
                            | 尚未解决或不确定
                            v
TIER 3        异步工具调用调查：
              历史告警、摄像头健康、其他摄像头、值勤安排、录像
                            v
POLICY        suppress / downgrade / alert / escalate
                            v
EVENT END     完整事件摘要和盯防：可以提级，不会降级
                            v
HUMAN REVIEW  每个房间一张卡片 -> 人工纠正 -> 记忆；在聊天中提问
```

* **按需调用模型。** 初筛关卡能解决的事件不需要调用模型；未解决的事件先接受
  筛查，必要时再进行调查。
* **提前通知。** Frigate 新建复核事件时即开始筛查。默认情况下，筛查耗时较长会
  在 10 秒等待预算后发送“Analyzing…”卡片；离家布防和夜间布防使用零秒预算，
  但派发等待时间最低为 0.5 秒。这些是调度设置，不是送达保证。后续报告只有出现
  标签或区域变化时才会再次筛查。
* **感知布防状态。** 视觉模型会获知布防状态。离家布防或夜间布防时，画面中的
  人员会触发告警和调查。只有存在匹配的值勤安排条目，或 Frigate 认出该人员是
  住户时，判定为正常的调查才可以放行；卡片会显示这一依据。单凭值勤安排规则
  不会消除告警。没有布防状态时，Watchwise 假定为 `armed_away`；可用 `/arm`
  切换，或在 `/setup` 中设置默认值。
* **人脸。** 开启 Frigate 的人脸识别并在 `faces.household` 中列出住户（可由
  `/setup` 完成）后，你在家时 Frigate 认出的住户会被拦下，不调用模型。但盯防，
  以及措辞中包含“anyone”（任何人）或“any person”（任何人员）的总要提醒规则，仍会
  在事件结束时针对他们询问一次。Frigate 看到但无法匹配任何人的人脸，会在事件结束
  时把被拦下的事件提高为告警；但在安排好的来访期间，只会注明未匹配人脸。
  Frigate 使用共用人脸库进行识别；认出的名字会出现在卡片文字中。
* **房间。** `cameras.<name>.room` 将摄像头归入房间：每次事件每个房间一张卡片，
  点击一次即可教会卡片上的每个摄像头；调查会把同一房间的另一个摄像头视为另一个
  视角，而不是有人在住宅内移动。
* **模型预热。** `service.warm_up_models` 在启动时加载本地模型，让筛查
  模型保持已加载状态。服务启动和内存压力仍可能引起延迟。
* **故障处理。** 筛查错误、超时、拒绝回答或无法解析的回复会转为 uncertain；
  无法解决的调查会交给人工复核。这不能防止错误的正常活动判定或错误的关卡匹配。
* **可审计的决策。** 被拦下的事件仍有记录。决策包括理由、置信度、匹配记忆，
  以及实际开展过调查时的调查轨迹。模式保留纠正次数及同意／不同意次数。
* **视频仍由 Frigate 管理。** Watchwise 读取 Frigate 的 HTTP API 和 MQTT，写入
  自己的数据库，并提供自己的 API。`/face` 录入和 👤 人脸教学只有在点击 ✅ Add
  后才会写入 Frigate 的人脸库；启用图片发送时，预览会经过 Telegram。移除
  Watchwise 不会改变 Frigate 的录像或配置。

### 学习阈值与保护措施

默认情况下，误报模式至少需要 **3 次纠正**、**0.86 的匹配相似度**和 **0.70 的
信任度**，才能在初筛关卡解决事件。信任度计算为
`(agreements + 1) / (agreements + disagreements + 2)`；仅有纠正次数还不够。
🚨 Alert me 反馈可以降低信任度，Undo 则撤销其已记录的影响。对某类事件（摄像头、
标签、区域）点击 🚨 Alert me 后，30 天内记忆不会再拦下这类事件
（`gate.real_veto_days`）。

关卡置信度为“信任度 × 相似度”，上限为 `0.95`。策略只有在置信度达到 `0.85`
或更高时才会完全抑制已学习的匹配事件；符合关卡条件但分数较低的事件会降级为
检测记录。两种结果通常都不会新发手机卡片，但事件仍可检查。配置为始终告警的
标签优先于记忆和去重规则。

模式以摄像头、目标标签和区域为键。在 CPU 和云端预设中，初筛关卡会自行拦下可信
的常规活动，但不查看画面，因此无法区分不同的行为。在 GPU 和 Apple Silicon 预设
（`gate.memory_mode: model_checked`）中，关卡从不自行拦下：视觉模型会将每个事件与
你标记为正常或需要提醒的描述进行比较；如果你回答过“What was wrong?”，就使用你
自己的话。一次纠正不会修改值勤安排；值勤安排规则来自聊天，并需要你确认。时间
衰减影响搜索排序，但**不会**使模式到期，也不会单独取消模式拦下告警的资格。

### 完整事件描述

事件结束时，Watchwise 会根据分布在整个事件中的多帧画面，询问视觉模型发生了
什么（`vlm.summary_frames`；CPU 预设中关闭），并在卡片和决策历史中附上
“What happened”（发生了什么）。Frigate 发送的 GenAI 复核摘要也会以同样方式使用。
`decision.summary_can_raise: true` 为默认值；此时，可疑或严重级别的摘要会将已拦下
的事件提高为告警，但不能降低决策级别。摘要会在 Frigate 关闭复核事件后到达，即
活动停止后的若干秒；早期判定可能只描述一张快照。

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
后者需要 `pip install -e ".[mcp]"`。Telegram 中的 Ask Watchwise 使用自己的
工具集；它没有 HTTP 端点。

## 运行检查

```bash
pip install -e ".[dev,anthropic,mcp,agent]"
pytest -q
```

Python 自动化测试使用假的 Frigate、模型和 Telegram Bot API 服务（均为真实 ASGI
应用），并使用硬件检测夹具，无需 GPU、摄像头、机器人或 API 密钥。测试覆盖路由、
反馈影响与 Undo、通知更新、房间、`/setup` 对话及其文件变更、聊天回答和已确认的
更改（使用脚本化模型）、预热与超时行为、录像 URL 和摘要提级。这些检查验证应用
行为，不衡量视觉准确率。

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
              setup/、notify/、chat/、cli.py
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
