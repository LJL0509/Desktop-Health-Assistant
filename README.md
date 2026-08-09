# Desktop Health Assistant

本地运行的桌面健康助手。项目使用摄像头和个人姿势校准，观察长期使用电脑时的脖子前倾状态。摄像头画面只在内存中处理，不保存、不上传。

当前版本：`v0.1.0`，脖子前倾持续监控原型。

## v0.1.0 功能

- 15 秒个人正常坐姿校准；
- MediaPipe 人脸、耳部和虚拟肩部锚点；
- 最近 5 秒窗口平滑；
- `NECK NORMAL` 与 `NECK FORWARD` 二态判断；
- 3 秒状态确认，减少阈值附近闪烁；
- 数据连续不足 10 秒后的应用内提示；
- 暂停时释放摄像头；
- 本地记录状态变化和人工正确/错误反馈；
- 不把整体靠近屏幕误认为脖子前倾。

## 环境

- Windows 11
- Python 3.11
- MediaPipe 1.0.0
- 内置摄像头，优先使用 Media Foundation 共享接口

## 首次运行

```powershell
cd D:\Github\Desktop-Health-Assistant
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe scripts\neck_monitor.py
```

## 操作

- `C`：开始正常坐姿校准；
- `P`：暂停并释放摄像头，再按一次恢复；
- `Y`：标记当前判断正确；
- `N`：标记当前判断错误；
- `Q` 或 `Esc`：退出。

## 当前判定逻辑

虚拟锚点模式比较脸宽与虚拟肩宽相对个人基线的变化：

- 脸宽增长且脸宽/虚拟肩宽比例同时增长，持续满足条件后判断为 `NECK FORWARD`；
- 整个身体靠近时，脸宽和虚拟肩宽同步增长，脖子仍判断为 `NECK NORMAL`；
- 屏幕距离与脖子姿势是两个独立系统，v0.1.0 暂不判断距离是否健康。

当前 8% 阈值来自个人原型实验，不是医学标准。

## 验证结果

受控测试中共记录 25 次人工反馈，其中 23 次正确、2 次错误，人工反馈准确率为 92%。数据不足提醒触发 3 次，画面恢复后均自动清除。

详细记录：

- [需求文档](docs/requirements.md)
- [环境兼容性检查](docs/environment-check.md)
- [姿势识别设计](docs/posture-detection-design.md)
- [姿势区分实验](docs/posture-experiment-results.md)
- [v0.1 持续监控验证](docs/neck-monitor-validation.md)
- [v0.1 开发与调试过程](docs/v0.1-development-log.md)

## 限制

- 不是医疗设备，不诊断颈椎病、圆肩或其他疾病；
- 目前仅基于一名用户和一台内置摄像头验证；
- 轻微前倾可能漏检；
- 摄像头位置、屏幕角度或座椅变化后需要重新校准；
- v0.1.0 只有应用内提示，没有 Windows 系统通知；
- 肩膀不在画面内时不判断圆肩和肩膀倾斜。
