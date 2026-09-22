# AIRBOT 视觉抓取与语音控制操作 SOP

适用目录：`/home/su/graspdemo`

适用功能：RealSense 视觉采集、YOLO 实时积木检测、蓝/绿颜色判定、MobileSAM 自动分割、抓取位姿计算、AIRBOT Play 抓取与放置，以及 FunASR 中文语音指令输入。鼠标选择保留为备用流程。

> 安全原则：首次联调必须关闭“语音识别后直接执行”。GUI 启动时会自动驱动机械臂到配置的观察位，因此启动 GUI 不是离线操作。急停只能使用硬件急停或既有控制器的停止方式，“取消语音”和语音“停止”都不是急停。

## 1. 系统组成与关键路径

| 项目 | 当前值/路径 |
| --- | --- |
| 工程目录 | `/home/su/graspdemo` |
| 抓取入口 | `app/airbot_interface.py` |
| 统一环境 | `/home/su/graspdemo/venv`（视觉、语音、SDK 共用） |
| 常驻语音工作进程 | `/home/su/graspdemo/app/voice_asr_worker.py` |
| 主配置入口 | `configs/config_file.yaml` |
| 当前实际配置 | `configs/sam_simplegrasp.yaml` |
| 当前机器人端口 | `50051` |
| 机械臂 SDK | `arm-sdk 5.2.2` |
| 机械臂服务 | `airbot-arm 5.2.2` |
| 当前服务硬件类型 | `airbot_play_g2`（G2 夹爪；必须与实物一致） |
| 分割模型 | `checkpoint/mobile_sam.pt` |
| YOLO 模型 | `checkpoint/yolo_best_0414.pt` |
| 运行数据 | `runtime/` |

语音识别使用本地 FunASR，不需要 OpenAI、DeepSeek、公司 API key 或 Ollama。
应用源码统一位于 `app/`，离线测试位于 `tests/`，文档位于 `docs/`；操作时仍从工程根目录执行脚本，避免相对配置和模型路径失效。`vendor/legacy/` 仅保存旧版 wheel，不参与 5.2.2 安装。

## 2. 5.2.2 软件环境准备

抓取、视觉和语音统一使用项目 `venv`。机械臂服务和 SDK 统一使用 5.2.2 版本。

### 2.1 新机器一条命令安装

将厂商提供的两个 5.2.2 安装包放入项目 `packages/`：

```text
arm_sdk-5.2.2-py3-none-any.whl
airbot-arm_5.2.2_amd64.deb
```

然后执行：

```bash
cd /home/su/graspdemo
./install.sh
```

该命令安装 Ubuntu 录音/Qt 依赖，创建 Python 3.12 `venv`，安装视觉推理、FunASR 和 arm-sdk 依赖，并预下载语音模型。安装包不在 `packages/` 时可在同一条命令中指定：

```bash
./install.sh \
  --sdk-wheel /path/to/arm_sdk-5.2.2-py3-none-any.whl \
  --arm-deb /path/to/airbot-arm_5.2.2_amd64.deb
```

已经安装系统服务时可增加 `--skip-system`。安装后运行 `./run_grasp.sh --check`；该检查不会连接或驱动机械臂。

### 2.2 手动安装主程序服务（仅排障）

```bash
sudo dpkg -i \
  '/home/su/下载/5.2.2软件包/airbot_arm_release/product/x86_64/noble/airbot-arm_5.2.2_amd64.deb'
dpkg-query -W -f='${Version}\n' airbot-arm
airbot-arm --version
```

`dpkg-query` 的预期输出为 `5.2.2`。当前机器已经检测到该 deb 已安装；重复安装同一包一般不需要。注意：你提供的 5.2.2 deb 中，`airbot-arm --version` 实测仍输出 `1.0`，因此判断 deb 版本应以 `dpkg-query` 为准；该现象属于安装包自身的版本字符串，不代表系统仍安装旧版。

本项目按你提供的 `5.2.2` deb 与 wheel 精确固定。本版本的包名、`Controller`、运动接口和 Lease 机制以官方 2026-03-13 的 V5.2 changelog 为依据：<https://docs.discover-robotics.com/document/airbot-play/changelog.html#20260313>。

### 2.3 手动安装 Python SDK（仅排障）

本项目使用 Ubuntu 的 Python 3.12。不要在 Conda `base` 中用未限定版本的 `python3 -m venv`；它可能选中 Conda 的 Python 3.14，导致视觉依赖不可用。

当前 `/home/su/graspdemo/venv` 若显示 Python 3.14 或缺少 `cv2`，按以下步骤重建。旧环境会被重命名保留，不会直接删除：

```bash
deactivate 2>/dev/null || true
cd /home/su/graspdemo
sudo apt install python3.12-venv libxcb-cursor0
mv venv "venv-python314-unused-$(date +%Y%m%d-%H%M%S)"
/usr/bin/python3.12 -m venv venv
source venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt \
  -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

`python --version` 应显示 `Python 3.12.x`。如果环境本来已经完整，只需激活：

```bash
cd /home/su/graspdemo
source venv/bin/activate
```

如果 Qt 报错 `xcb-cursor0 or libxcb-cursor0 is needed`，执行 `sudo apt install libxcb-cursor0`。

如果使用 Conda，则改为对应的 `conda activate <环境名>`。激活后确认解释器路径，再安装指定 wheel：

```bash
which python
python -m pip install --force-reinstall \
  '/home/su/下载/5.2.2软件包/sdk_client_release/dist/x86_64/arm_sdk-5.2.2-py3-none-any.whl'
python -c "import arm_sdk, cv2, sklearn; print('arm-sdk', arm_sdk.__version__); print('OpenCV', cv2.__version__); print('scikit-learn', sklearn.__version__)"
```

`arm-sdk` 预期输出必须是 `5.2.2`，并且应同时输出 OpenCV 和 scikit-learn 版本。项目适配层会拒绝其他 SDK 版本，避免混用不匹配接口。

5.2 SDK 接口总览：<https://docs.discover-robotics.com/document/airbot-play/sdk/api/reference.html>。

> 当前机器的 Conda `airbot` 环境虽然已安装 `arm-sdk 5.2.2`，但检查时缺少 PyQt6、Torch、pyrealsense2、Open3D、Ultralytics 和 MobileSAM。因此除非补齐这些视觉依赖，否则它不能直接运行本 demo；应使用已经跑通视觉 demo 的环境。

### 2.4 检查统一语音环境

```bash
./venv/bin/python -c \
  "import sounddevice, soundfile, funasr; print('语音依赖正常')"
```

若失败，执行 `./install.sh --skip-system` 修复项目环境。旧的独立语音环境不再被本项目使用。

## 3. 每次启动前检查

### 3.1 现场安全检查

- 机械臂、夹爪、相机安装牢固，线缆不会进入运动范围。
- 工作区内没有人员、工具或其他障碍物，硬件急停可立即触达。
- 待抓物体位于经过标定和验证的可抓取区域内。
- `observe_pose`、`place_pose` 和相机外参是当前设备的参数，不能直接照搬其他机械臂。
- 首次测试使用低速，并先测试观察位和夹爪，再测试完整抓取。

### 3.2 检查配置和模型

```bash
cd /home/su/graspdemo
sed -n '1,20p' configs/config_file.yaml
grep -En 'port:|server_arm_type:|gripper_width:|observe_pose:|place_pose:|resolution:|checkpoint:' \
  configs/sam_simplegrasp.yaml
test -f checkpoint/mobile_sam.pt && \
  python -c "import mobile_sam; print('MobileSAM 模块与权重: OK')"
test -f checkpoint/yolo_best_0414.pt && echo 'YOLO: OK'
```

`configs/config_file.yaml` 当前应指向 `configs/sam_simplegrasp.yaml`，该文件的 `ArmParams.port` 当前为 `50051`，`server_arm_type` 为 `airbot_play_g2`。服务端口和硬件类型必须与实物一致：G2 使用 `airbot_play_g2`，G2L 使用 `airbot_play_g2l`，不带夹爪才使用 `airbot_play`。

### 3.3 检查设备

```bash
ip link show can0
lsusb
./venv/bin/python -m sounddevice
```

麦克风选择有输入通道的设备。此前机器中 `4` 是板载模拟输入，`10` 是 PipeWire，`11` 是默认输入；设备编号在重启或插拔设备后可能变化，所以应以本次列表为准。HDMI 且显示 `0 in` 的设备不能录音。

## 4. 先做离线验证（不连接或驱动机械臂）

### 4.1 验证固定指令解析

```bash
cd /home/su/graspdemo
PYTHONPATH=app ./venv/bin/python app/voice_commands.py --text '打开夹爪'
PYTHONPATH=app ./venv/bin/python app/voice_commands.py --text '打开假爪'
./run_tests.sh
```

预期分别输出 `open_gripper`、`open_gripper`、`grasp_blue`，测试结果为 `OK`。这些命令不连接机械臂或相机，也不会运动。

### 4.2 验证录音和识别

```bash
./run_grasp.sh --check
```

`--check` 会验证 FunASR 与录音依赖，但不会打开麦克风或控制机械臂。实际录音在 GUI 中验证；首次保持自动执行关闭。

## 5. 启动系统

使用两个终端。启动前再次确认：运行 GUI 会立即移动到观察位。

### 5.1 终端 A：启动机械臂服务

```bash
cd /home/su/graspdemo
sudo airbot-arm -i can0 -t airbot_play_g2 \
  --address 0.0.0.0:50051 --no-return
```

保持终端 A 运行。官方要求使用 `sudo` 启动服务。`--no-return` 表示停止服务时不自动回零，避免退出终端触发未预期运动；如现场流程明确要求回零，可按规范移除此参数。若服务已在正确端口运行，不要重复启动。可在另一个终端检查：

```bash
ss -ltnp | grep ':50051'
```

### 5.2 终端 B：启动抓取 GUI

```bash
cd /home/su/graspdemo
./run_grasp.sh
```

脚本会固定使用项目的 `venv` 同时启动 GUI 和 FunASR 子进程，无需手动激活环境或设置变量。首次运行前可执行 `./run_grasp.sh --check` 做离线环境检查；这不会连接相机或机械臂。默认麦克风编号为 `11`，临时改用其他设备可执行 `./run_grasp.sh --device 10`，也可以在 GUI 面板中修改。

启动脚本不会自动运行需要 `sudo` 的 `airbot-arm` 服务，终端 A 仍需保持运行。

启动后的正常现象：

1. GUI 打开并显示 RealSense 实时画面。
2. 机械臂移动到 `observe_pose`。
3. 日志出现“已移动到观察姿态”。
4. 右侧语音面板先显示模型加载，约 8–10 秒后显示“语音模型已就绪”。
5. 实时画面中的积木显示检测框、颜色、检测置信度和颜色覆盖率。

任一动作异常时，不要继续抓取，先按第 9 节排查。

抓取下降高度受 `configs/sam_simplegrasp.yaml` 中的 `AirbotGrasp.min_grasp_z` 硬限制。当前值为修正后现场参考位姿的 Z 值 `0.01338870424854599 m`；预测高度低于该值时会自动抬高到此值。该数值是工具中心在机械臂基座坐标系中的 Z，不是积木高度。现场重新标定时必须关闭自动执行，以 `SLOW` 速度进行单次监督测试。

实时积木候选默认需要连续 3 帧确认，检测框使用指数平滑，短时漏检保持 3 个检测周期，颜色按最近 5 帧投票。对应参数位于 `RealtimeDetection.tracking_*`。语音中的常见同音错词会先校正；若从较长环境语音中提取出唯一白名单口令，界面会要求人工确认，即使勾选自动执行也不会直接动作。

颜色抓取位姿准备超时为 8 秒。运动时先规划到目标上方 10 cm，再直线下降；日志会分别显示“移动到预抓取位”和“直线下降到抓取位”。若某一步失败，程序会停止后续路径，并输出该阶段及目标 position/orientation，便于区分预抓取位不可达和直线下降失败。

机械臂离开观察位后若抓取失败，程序会执行失败恢复：抓取区先打开未确认夹持的夹爪并直线上抬到预抓取位；放置低位失败时先释放物体并上抬到预放置位；随后规划返回初始观察位。若已经夹住物体但尚未到达低位放置区，返回观察位时保持夹爪闭合，避免从高处掉落，并要求人工处理。任何恢复步骤失败都会单独记录，且仍会继续尝试返回观察位。

蓝色和绿色积木使用同一个公共放置位。当前 `place_pose.position` 为 `[0.2512875752470927, 0.20641115004740498, 0.08319538687229266]`。抓取后先直线抬升，再移动到该位置正上方 10 cm 的 `pre_place_pose`，直线下降到公共放置位后松爪，然后直线抬升并返回观察位。

自动夹持使用视觉估算宽度减去 `ArmParams.gripper_grasp_compression`，并通过 `gripper_grasp_effort` 发送夹爪前馈力。程序以 50 ms 周期读取 `eef_pos/eef_vel/eef_eff`；开度受阻或目标附近出现强作用力、速度接近零且电机无故障，连续 3 个样本成立即判定“已夹住”。未接触时会按配置分级收紧两次，每级 3 mm；全部尝试仍未确认时停止抬升和放置。

## 6. 首次现场验证顺序

### 6.1 只测试夹爪

1. 确保夹爪周围没有手、物体或线缆。
2. 不勾选“语音识别后直接执行”。
3. 等待语音模型就绪，点击“开始录音”并说“打开夹爪”；说完停顿约 0.8 秒即自动结束。
4. 核对文字是“打开夹爪”，再点“确认并执行文字指令”。
5. 观察夹爪打开和日志“夹爪已打开”。
6. 用同样流程说“关闭夹爪”，确认夹爪关闭。

重复说“打开夹爪”仍然是打开，不会切换成关闭。

### 6.2 只测试实时视觉，不抓取

1. 把一个蓝色积木和一个绿色积木放入工作区，不要点击颜色抓取按钮。
2. 检查实时画面是否出现 `BLUE BLOCK` / `GREEN BLOCK` 框，右侧计数是否正确。
3. 记录右侧的检测置信度和颜色覆盖率。若颜色不稳定，在不启动抓取的情况下调整 `RealtimeDetection.colors`、`min_color_coverage` 和现场照明。
4. 分别移动积木和暂时移出画面，确认框与数量及时更新，不使用旧目标。

### 6.3 分阶段执行颜色抓取

1. 首次仅放置一个蓝色积木，机械臂选择 `SLOW`，不勾选直接执行。
2. 说“抓取蓝色积木”，核对文字和右侧目标信息后手动确认。
3. 程序会固定同一帧 RGB、深度、位姿和检测框，使用 MobileSAM 自动分割；目标过期、分割/深度无效时不会运动。
4. 持续观察机械臂：打开夹爪 → 目标上方 → 下探 → 闭合 → 抬起 → 放置 → 返回观察位。
5. 蓝色成功后，仅放置一个绿色积木，重复同样的低速验证。
6. 最后放入两个同色积木并发出对应指令，确认界面提示目标不唯一且机械臂不运动。

## 7. 日常操作流程

```text
安全检查
  → 启动 airbot-arm 5.2.2（gRPC 50051）
  → 启动 GUI（自动到观察位）
  → 等待 FunASR 就绪和实时检测稳定
  → 语音“抓取蓝色积木”或“抓取绿色积木”
  → 核对文字与唯一颜色目标，手动确认
  → 观察抓取/放置/返回
  → 等待新的实时检测结果
```

确认多次识别、颜色检测和工作区安全后，才可勾选“高可信识别后直接执行”。弱模糊匹配仍必须人工确认。

## 8. 支持的语音指令

| 类别 | 示例 | 前置条件 | 是否运动 |
| --- | --- | --- | --- |
| 图像采集 | 拍照、捕获图像 | 相机正常 | 否 |
| 位姿预测 | 预测抓取、预测抓取位姿 | 已拍照并正确选择目标 | 否 |
| 完整抓取 | 开始抓取、抓取选中物体、抓取并放置 | 已拍照并正确选择目标 | 是 |
| 蓝色积木 | 抓取蓝色积木 | 实时画面中恰好一个有效蓝色目标 | 是 |
| 绿色积木 | 抓取绿色积木 | 实时画面中恰好一个有效绿色目标 | 是 |
| 夹爪 | 打开夹爪、关闭夹爪 | 抓取线程空闲 | 是 |
| 观察位 | 回到观察位、返回观察位 | 抓取线程空闲 | 是 |

一次只说一条指令。支持将“假爪/甲爪”等已审核错词校正为“夹爪”。否定句、疑问句、转述、多条指令、环境对话和不支持的颜色会被拒绝。

## 9. 异常处理

| 现象 | 检查方法 | 处理 |
| --- | --- | --- |
| 提示找不到语音 Python/工作脚本 | `test -x venv/bin/python` 和 `test -f app/voice_asr_worker.py` | 执行 `./install.sh --skip-system` |
| 麦克风无输入 | 重新执行 `--list-devices` | 换成有输入通道的设备，或清除 `GRASP_VOICE_DEVICE` 使用默认设备 |
| FunASR 首次很慢 | 先在终端执行第 4.2 节 | 等模型下载及缓存完成，再启动 GUI |
| GUI 缺少 PyQt6/SDK/视觉模块 | `./venv/bin/python -c "import PyQt6, arm_sdk, cv2, torch"` | 执行 `./install.sh --skip-system` 修复项目统一环境 |
| 无 RealSense 画面 | `lsusb` 并检查线缆 | 重插相机，关闭占用相机的其他程序后重启 GUI |
| 无法连接机械臂服务 | 对照配置端口并执行 `ss -ltnp \| grep ':50051'` | 确认 CAN、服务进程和端口一致 |
| 颜色积木未识别/识别错 | 查看右侧颜色覆盖率 | 保持稳定照明，现场校准 `RealtimeDetection.colors` 与覆盖率阈值 |
| 提示目标不唯一 | 查看同色目标数量 | 移除多余同色积木，系统不会自行猜测 |
| 识别成功但没有动作 | 看是否等待确认、目标过期/不唯一、分割/深度失败或线程忙 | 查看目标状态与诊断日志；失败时程序不会启动抓取 |
| 分割区域错误 | 右键补背景或中键清除 | 不得继续抓取，重新选择直到区域正确 |
| 预测失败 | 查看日志、深度图和目标是否在有效区域 | 重新摆放物体、拍照、选择；预测失败时程序不会启动抓取 |
| 运动异常或有碰撞风险 | 立即使用硬件急停 | 排除原因前不得重试；“取消语音”不能停止已开始的运动 |

## 10. 正常停机

1. 等待当前抓取完整结束，机械臂回到观察位。
2. 关闭 GUI 窗口，等待相机线程和抓取线程退出。
3. 回到终端 B，确认程序已结束。
4. 在终端 A 按 `Ctrl+C` 停止 `airbot-arm`；本 SOP 使用了 `--no-return`，不会因退出服务自动回零。
5. 按设备现场规范断开机械臂动力和相机连接。

不要在机械臂运动过程中直接关闭终端或断电；紧急情况使用硬件急停。

## 11. 验收记录

每次首次部署或更改标定、模型、SDK、机械臂、相机安装位置后，建议逐项记录：

- [ ] 固定指令离线测试通过
- [ ] 麦克风录音与中文识别通过
- [ ] RealSense 彩色图和深度图正常
- [ ] 蓝色/绿色积木实时检测与 HSV 覆盖率已现场校准
- [ ] 两个同色目标时正确拒绝抓取
- [ ] 观察位运动正确
- [ ] 夹爪打开/关闭正确
- [ ] 手动分割正确
- [ ] 抓取位姿预测合理
- [ ] 单次低速抓取与放置成功
- [ ] 硬件急停经过现场规范检查

软件架构及模块职责见 [SOFTWARE_ARCHITECTURE.md](SOFTWARE_ARCHITECTURE.md)。
