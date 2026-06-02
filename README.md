# IkAssemblySim — 智能制造 Agent 仿真包

基于 CoppeliaSim 的 Mirobot 机械臂 + ACOPOS 6D 滑块控制代码，并补充了“自然语言输入 → Agent 任务解析 → 产线规划 → 仿真控制执行”的主流程。

## 环境要求

- **仿真器**：[CoppeliaSim Edu V4.10.0](https://www.coppeliarobotics.com/) (Portable 版)
- **Python 依赖**：

```bash
pip install coppeliasim-zmqremoteapi-client numpy
```

默认使用本地 Ollama 模型解析自然语言，不调用云端 API。请先确认本机已安装并启动 Ollama，且已拉取课程使用的模型：

```bash
ollama pull deepseek-r1:7b
```

默认配置位于 `config.py`：

```bash
set BASE_MODEL=deepseek-r1:7b
set BASE_URL=http://localhost:11434/v1
set API_KEY=ollama
set LLM_TIMEOUT=180
```

其中 `API_KEY=ollama` 只是为了兼容 OpenAI-style 客户端，本地 Ollama 服务不会校验该值。环境变量仍可覆盖这些默认值。模型不可用或输出无法校验时，程序会使用本地规则解析作为演示兜底，支持“生产一辆车”“生产两部手机”等常见表达。

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
```

只演示 `Robot_Put_Mid` 跨产线转运：

```bash
python main.py --demo-mid-transfer
```

只测试自然语言解析、不连接仿真：

```bash
python main.py --parse-only "连续生产一辆车和两部手机"
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
| Agent 解析 | OpenAI 兼容接口输出结构化 JSON，未配置 API Key 时规则兜底 |
| 任务执行 | LLM 只解析产品和数量，车辆/手机工艺路线由确定性控制器执行 |
| 中间转运 | `Robot_Put_Mid` 支持将工件从一侧滑块搬运到另一侧滑块，可用 `--demo-mid-transfer` 单独展示 |
| 摄像头质检 | 当前实现为滑块到达对应 Camera 工位后的停顿等待，不做图像识别判定 |
| 防穿模 | 基于滑块矩形占用与关键工位 reservation 的运行前检查；主生产使用 A/B 两个执行滑块，A2/B2 作为已占用滑块纳入避障 |

## 测试

```bash
python -m unittest discover -s tests
python main.py --parse-only "连续生产一辆车和两部手机"
python main.py --demo-mid-transfer
```
