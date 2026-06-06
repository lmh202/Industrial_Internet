# IkAssemblySim — 智能制造 Agent 仿真包

基于 CoppeliaSim 的 Mirobot 机械臂 + ACOPOS 6D 滑块控制代码，并补充了“自然语言输入 → Agent 任务解析 → 产线规划 → 仿真控制执行”的主流程。

## 环境要求

- **仿真器**：[CoppeliaSim Edu V4.10.0](https://www.coppeliarobotics.com/) (Portable 版)
- **Python 依赖**：

```bash
pip install coppeliasim-zmqremoteapi-client numpy
```

默认使用本地 Ollama 模型解析自然语言，不调用云端 API。请先确认本机已安装并启动 Ollama，且已拉取当前使用的模型：

```bash
ollama pull qwen3.6
```

默认配置位于 `config.py`：

```bash
set BASE_MODEL=qwen3.6:latest
set BASE_URL=http://localhost:11434/v1
set API_KEY=ollama
set LLM_TIMEOUT=360
set LLM_KEEP_ALIVE=-1
```

其中 `API_KEY=ollama` 只是为了兼容 OpenAI-style 客户端，本地 Ollama 服务不会校验该值。`LLM_KEEP_ALIVE=-1` 表示普通调试运行后通过 Ollama 原生请求刷新模型常驻状态，避免每次运行都重新加载；需要释放显存/内存时再手动执行卸载命令。环境变量仍可覆盖这些默认值。模型不可用、输出无法校验或计划包含非法工具时，程序会直接报错停止，不再回退到固定的“造车/造手机”模板。

## 文件说明

```
Project/
├── models/
│   ├── ACOPOS6D_MotorSegment.ttm   # 电机段模型
│   └── ACOPOS6D_Shuttle.ttm        # 滑块模型
├── scenes/
│   └── assembly_line.ttt            # 双产线场景（7臂4滑块）
├── robot_arm.py                     # 机械臂控制器（pick/place）
├── mirobot_ik.py                    # 纯 Python 数值 IK 解算器 (DLS)
├── shuttle_controller.py            # 滑块运动控制器（梯形速度规划）
├── scene_config.py                  # 场景参数常量
├── config.py                        # LLM、CoppeliaSim、场景路径配置
├── llm_client.py                    # OpenAI 兼容 Chat Completions 客户端
├── agent.py                         # 自然语言任务解析与校验
├── collision_manager.py             # 滑块防穿模占用检测
├── factory_controller.py            # 车辆/手机确定性生产流程
└── main.py                          # 命令行主入口
```

## 快速上手

### 1. 直接运行 Agent 主流程

默认会自动启动：

`C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe`

并加载：

`scenes/assembly_line.ttt`

示例：

```bash
python main.py "生产一辆车"
python main.py "生产两部手机"
python main.py "连续生产一辆车和两部手机"
python main.py "把手机产线中的某个零件移到output中"
python main.py "把手机产线中的屏幕移到output中"
python main.py "把汽车产线中的车架送到输出区"
```

程序完成首条任务后不会自动退出，而是进入会话模式继续等待输入。后续每输入一条新的生产指令，程序会先让 LLM 输出工具调用计划书，再重新加载 `assembly_line.ttt` 复位 CoppeliaSim 场景并执行计划。输入 `exit`、`quit` 或 `q` 退出会话。

也可以不带初始任务，直接启动仿真会话：

```bash
python main.py
```

包含车辆和手机的混合任务会进入双产线调度：车辆进入 A 线队列，手机进入 B 线队列，两条线各保持一个活动产品并轮转推进。

只演示 `Robot_Put_Mid` 跨产线转运：

```bash
python main.py --demo-mid-transfer
```

只测试自然语言解析、不连接仿真：

```bash
python main.py --parse-only "连续生产一辆车和两部手机"
```

`--parse-only` 输出的是工具调用计划书，而不是旧版产品数量任务，例如：

每次生成的计划都会保存到项目根目录 `plan.json`，下次运行会直接覆盖上一份计划。

```json
{
  "plan_name": "one_car",
  "steps": [
    {"tool": "Load_A_Pick", "args": {"part": "car_frame"}},
    {"tool": "Transport_A_Pick_Assemble", "args": {}},
    {"tool": "Hold_A_Assemble", "args": {"part": "car_frame"}},
    {"tool": "Transport_A_Assemble_Pick", "args": {}},
    {"tool": "Load_A_Pick", "args": {"part": "car_base"}},
    {"tool": "Transport_A_Pick_Assemble", "args": {}},
    {"tool": "Place_A_Assemble", "args": {"part": "car_frame", "attach_to": "car_base", "layer": 1}},
    {"tool": "Transport_A_Assemble_Camera", "args": {}},
    {"tool": "Inspect_A", "args": {}},
    {"tool": "Transport_A_Camera_Output", "args": {}},
    {"tool": "Unload_A_Output", "args": {"part": "car_base"}}
  ]
}
```

对常见的单步操作型指令，Agent 会先走确定性解析再调用模型。例如“把手机产线中的某个零件移到 output 中”会默认选择手机机身 `phone_base`，生成 `Load_B_Pick(phone_base) -> ... -> Unload_B_Output(phone_base)`；如果明确说“屏幕”，则会使用 B2 辅助滑块装载屏幕后转移到 B 主滑块再送往 output。

允许的工具函数：

```text
Load_A_Pick(part, local_offset?)
Load_B_Pick(part, local_offset?)
Load_B2_Pick(screen)
Transport_A_Pick_Assemble
Transport_A_Assemble_Pick
Transport_A_Assemble_Camera
Transport_A_Camera_Output
Transport_A_Output_Pick
Transport_B_Pick_Assemble
Transport_B_Assemble_Pick
Transport_B_Assemble_Clear
Transport_B_Clear_Assemble
Transport_B_Assemble_Camera
Transport_B_Camera_Output
Transport_B_Output_Pick
Transport_B2_Clear_Pick
Transport_B2_Pick_Assemble
Transport_B2_Assemble_Clear
Hold_A_Assemble(part)
Hold_B_Assemble(part)
Hold_B2_Assemble(part)
Place_A_Assemble(part, attach_to?, layer?, local_offset?)
Place_B_Assemble(part, attach_to?, layer?, local_offset?)
Inspect_A
Inspect_B
Unload_A_Output(part)
Unload_B_Output(part)
Transport_A_B(part, target_offset?)
Transport_B_A(part, target_offset?)
```

手机产线使用 B1 作为主产品滑块、B2 作为辅助零件滑块。相机和底座组装完成后，B1 先从 Assemble 工位向内侧移动到刚好满足最小中心距的安全 clear 位；B2 从场景初始位回到 `Robot_Put_B` 下装载屏幕，再运输到 `Robot_Assemble_Phone` 下，由装配机械臂抓起屏幕。随后 B2 复位到场景初始位置，B1 回到 Assemble 工位，装配机械臂把屏幕放到底座上完成组装，再进入 Camera 质检。

连续生产同一条产线的多个产品时，上一件产品 `Unload_*_Output` 后必须先通过 `Transport_A_Output_Pick` 或 `Transport_B_Output_Pick` 将主滑块送回 pick 工位，再开始下一件产品的 `Load_*_Pick`。屏幕只能使用 B2 辅助滑块的 `Load_B2_Pick(screen)` 上料，不能使用 B 主滑块的 `Load_B_Pick(screen)`。

手动卸载当前 Ollama 模型：

```bash
python main.py --unload-model
```

如果已经手动打开 CoppeliaSim：

```bash
python main.py --no-launch "生产一辆车"
```

### 2. 手动启动 CoppeliaSim

```bash
# 启动 CoppeliaSim（portable 版）
"C:\Program Files\CoppeliaRobotics\CoppeliaSimEdu\coppeliaSim.exe"
```

### 3. 连接并加载场景

```python
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

# 加载预构建场景（使用绝对路径）
sim.loadScene('C:/path/to/minimal_sim/scenes/assembly_line.ttt')

# 关闭物理引擎、stepping 模式
sim.setBoolParam(sim.boolparam_dynamics_handling_enabled, False)
sim.setStepping(True)
sim.setFloatParam(sim.floatparam_simulation_time_step, 0.05)

# 启动仿真
sim.startSimulation()
```

### 4. 控制机械臂

```python
from robot_arm import RobotArmController

# 初始化（自动缓存运动链、验证 FK 精度）
arm = RobotArmController(sim, 'Robot_Put_A')

# 移动末端到目标位置（IK 解算 + 平滑动画）
arm.move_tip_to([0.1, 0.2, 0.12])

# 从指定位置取件
part_handle = sim.getObject('/Part_Bottom_A0')
part_pos = sim.getObjectPosition(part_handle, -1)
arm.pick_from_position(part_pos, part_handle)

# 放到指定位置
arm.place_at_position([0.0, -0.1, 0.08])

# 归位
arm.move_to_home()
```

### 5. 控制滑块

```python
from shuttle_controller import ShuttleController

handle = sim.getObject('/Shuttle_A1')
shuttle = ShuttleController(sim, handle, name='Shuttle_A1', dt=0.05)

# 平面移动（梯形速度规划）
shuttle.move_to(0.0, -0.5, speed=0.3)

# 旋转
shuttle.rotate_to(90)

# 沿路径移动
shuttle.move_along_path([(0.1, 0.0), (0.1, -0.3), (0.0, -0.5)])
```

### 6. 停止仿真

```python
sim.stopSimulation()
```

## 关键技术

| 模块 | 方案 |
|------|------|
| 逆运动学 | 纯 Python 数值 IK（Damped Least Squares），不依赖 simIK |
| 滑块运输 | 运动学模式，梯形速度规划（2 m/s / 20 m/s²） |
| Agent 规划 | LLM 输出受约束的工具调用 JSON 计划书，计划校验失败直接报错 |
| 任务执行 | Python 执行器只接受白名单工具函数，坐标、IK、避障和句柄由工具内部封装 |
| 双线调度 | 车辆任务进入 A 线队列、手机任务进入 B 线队列；调度器同时保持两条线各一个活动产品并按阶段轮转推进 |
| 中间转运 | `Robot_Put_Mid` 支持将工件从一侧滑块搬运到另一侧滑块，可用 `--demo-mid-transfer` 单独展示 |
| 摄像头质检 | 当前实现为滑块到达对应 Camera 工位后的停顿等待，不做图像识别判定 |
| 防穿模 | 基于滑块矩形占用与关键工位 reservation 的运行前检查；安全边距为 5mm，B2 作为手机辅助滑块参与调度 |

## 测试

```bash
python -m unittest discover -s tests
python main.py --parse-only "连续生产一辆车和两部手机"
python main.py --demo-mid-transfer
```

## Agent architecture notes

The planning stack is split into formal Agent layers:

- `tool_registry.py`: the single tool whitelist and metadata source, including
  routes, argument schemas, executor names, and prompt descriptions.
- `rules.py`: deterministic parsing for common commands that should not depend
  on the local model, such as moving a specific line part to `output`.
- `planner.py`: local-model planner with validation-first retries.
- `validator.py` and `state.py`: JSON shape validation plus abstract station,
  part, and holding-state simulation before CoppeliaSim execution.
- `factory_controller.py`: registered tool executor that returns a structured
  execution report for each step.

The current autonomy policy is validation-first: invalid model plans are retried
up to three times, but the runtime does not perform autonomous replanning after
an execution failure. Each user command reloads the scene before execution, so
the state model currently assumes a fresh factory state per command.
