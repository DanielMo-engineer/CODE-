#!/usr/bin/env python3
"""
摔倒监测脚本 - 配合 RDK X5 hobot_falldown_detection 使用

功能:
  1. 检测到摔倒且10秒无反应 → 终端打印 "警告"（仅一次）
  2. 摔倒目标重新站起来   → 终端打印 "正常"

用法:
  # 方式一：先手动启动 fall detection，再运行本脚本
  source /opt/tros/humble/setup.bash
  ros2 launch hobot_falldown_detection hobot_falldown_detection.launch.py &
  python3 fall_detection_monitor.py

  # 方式二：自动启动 fall detection + 本脚本
  source /opt/tros/humble/setup.bash
  python3 fall_detection_monitor.py --auto

用法参考:
  source /opt/tros/humble/setup.bash
  python3 /root/.openclaw/workspace/fall_detection_monitor.py
"""

import sys
import time
import subprocess
import threading
import signal

import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets

FALL_DETECTION_TOPIC = "/hobot_falldown_detection"
TIMEOUT_SECONDS = 10  # 摔倒后等待秒数

# ===== 状态定义 =====
STATE_NORMAL = 0      # 正常站立/行走
STATE_FALLEN = 1      # 已摔倒，等待计时
STATE_WARNED = 2      # 已发出警告


def get_falldown_status(target) -> int:
    """
    从 Target 的 attributes 中提取摔倒状态。
    返回: 0=正常, 1=摔倒, -1=未找到
    """
    for attr in target.attributes:
        attr_type = attr.type.strip().lower()
        # 常见摔倒属性名
        if attr_type in ("falldown_status", "falldown", "fall_status", "fall"):
            return int(round(attr.value))
    return -1


class FallMonitorNode(Node):
    def __init__(self, auto_launch=False):
        super().__init__("fall_monitor")

        # 每人一个状态机: track_id -> {"state": STATE_*, "fall_time": float, "warned": bool}
        self.persons = {}
        self.warning_active = False  # 是否已发出警告（全局）

        # 是否自动启动 fall detection pipeline
        self.auto_launch = auto_launch
        self.ros_process = None

        if auto_launch:
            self._launch_falldown_pipeline()

        # 订阅摔倒检测结果
        self.create_subscription(
            PerceptionTargets,
            FALL_DETECTION_TOPIC,
            self.callback,
            10
        )
        self.get_logger().info(f"已订阅 {FALL_DETECTION_TOPIC}，等待摔倒检测消息…")

    def _launch_falldown_pipeline(self):
        """自动启动 fall detection pipeline"""
        self.get_logger().info("正在启动 hobot_falldown_detection pipeline…")
        self.ros_process = subprocess.Popen(
            [
                "ros2", "launch",
                "hobot_falldown_detection",
                "hobot_falldown_detection.launch.py"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=lambda: None,  # 不继承信号
        )
        # 等几秒让系统启动
        time.sleep(5)
        self.get_logger().info("fall detection pipeline 已启动")

    def callback(self, msg: PerceptionTargets):
        now = time.time()
        current_ids = set()

        for target in msg.targets:
            tid = target.track_id
            current_ids.add(tid)
            status = get_falldown_status(target)

            if tid not in self.persons:
                self.persons[tid] = {
                    "state": STATE_NORMAL,
                    "fall_time": None,
                    "warned": False,
                }

            p = self.persons[tid]

            if status == 1:
                # === 摔倒状态 ===
                if p["state"] == STATE_NORMAL:
                    # 首次检测到摔倒
                    p["state"] = STATE_FALLEN
                    p["fall_time"] = now
                    p["warned"] = False
                    print(f"[跟踪ID {tid}] 检测到摔倒，开始计时 {TIMEOUT_SECONDS}s…")
                elif p["state"] == STATE_FALLEN:
                    # 持续摔倒
                    elapsed = now - p["fall_time"]
                    if elapsed >= TIMEOUT_SECONDS and not p["warned"]:
                        p["state"] = STATE_WARNED
                        p["warned"] = True
                        self.warning_active = True
                        print("\n⚠️  警告")
                # STATE_WARNED 状态保持

            elif status == 0:
                # === 正常站立 ===
                if p["state"] in (STATE_FALLEN, STATE_WARNED):
                    p["state"] = STATE_NORMAL
                    p["fall_time"] = None
                    if self.warning_active:
                        self.warning_active = False
                        print("\n✅  正常")
                    else:
                        # 摔倒后10秒内就站起来了，不发"正常"
                        print(f"[跟踪ID {tid}] 目标已恢复（10秒内）")

        # 清理消失的人（保持内存可控）
        # 注意：如果人走出画面又回来，track_id 可能不同
        for tid in list(self.persons.keys()):
            if tid not in current_ids:
                # 注意：disappeared_targets 也会由 PerceptionTargets 携带
                # 如果目标完全消失 30 秒以上，清理状态
                pass  # 暂不清理，留给 disappeared_targets 处理

        # 处理 disappeared_targets（目标消失）
        for target in msg.disappeared_targets:
            tid = target.track_id
            if tid in self.persons:
                p = self.persons[tid]
                if p["state"] in (STATE_FALLEN, STATE_WARNED):
                    # 摔倒状态下消失 → 也当恢复处理
                    p["state"] = STATE_NORMAL
                    if self.warning_active:
                        self.warning_active = False
                        print("\n✅  正常（目标消失）")

    def destroy_node(self):
        if self.ros_process:
            self.ros_process.terminate()
            self.ros_process.wait(timeout=5)
        super().destroy_node()


def main():
    auto_launch = "--auto" in sys.argv

    rclpy.init()
    node = FallMonitorNode(auto_launch=auto_launch)

    def sigint_handler(sig, frame):
        print("\n正在退出…")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
