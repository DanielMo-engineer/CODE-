#!/usr/bin/env python3
"""
人体摔倒监测节点
- 订阅 /hobot_falldown_detection (ai_msgs/msg/PerceptionTargets)
- 检测人体摔倒，若摔倒后持续10秒无动静 → 终端输出警告
"""

import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
import time
import sys

FALL_DOWN_ATTR = "falldown"
FALL_DOWN_THRESHOLD = 0.5  # 认定为摔倒的阈值
ALARM_DELAY_SEC = 10       # 摔倒后等待秒数


class FallMonitor(Node):

    def __init__(self):
        super().__init__('fall_monitor')
        self.sub = self.subscription = self.create_subscription(
            PerceptionTargets,
            '/hobot_falldown_detection',
            self.topic_callback,
            10)
        # 跟踪每个目标的摔倒起始时间 {track_id: fall_start_time}
        self.fall_start_time = {}
        self.alarm_triggered = {}  # {track_id: bool} 避免重复报警
        self.get_logger().info('摔倒监测节点已启动 ✓')

    def topic_callback(self, msg):
        now = time.time()
        current_fall_ids = set()

        for target in msg.targets:
            track_id = target.track_id
            is_fall = False

            # 遍历该目标的属性，查找"falldown"属性
            for attr in target.attributes:
                if attr.type == FALL_DOWN_ATTR and attr.value >= FALL_DOWN_THRESHOLD:
                    is_fall = True
                    break

            if is_fall:
                current_fall_ids.add(track_id)
                # 如果是新检测到的摔倒，记录时间
                if track_id not in self.fall_start_time:
                    self.fall_start_time[track_id] = now
                    self.alarm_triggered[track_id] = False
                    self.get_logger().info(
                        f'⚠️ 人体摔倒检测 (track_id={track_id})，开始计时...')

                elapsed = now - self.fall_start_time[track_id]
                # 超过10秒且未触发过报警
                if elapsed >= ALARM_DELAY_SEC and not self.alarm_triggered[track_id]:
                    self.alarm_triggered[track_id] = True
                    warn_msg = (
                        f'\n'
                        f'{"=" * 50}\n'
                        f'  ⚠️  警  告  ⚠️\n'
                        f'{"=" * 50}\n'
                        f'  检测到人体摔倒，持续{ALARM_DELAY_SEC}秒无动静！\n'
                        f'  目标ID: {track_id}\n'
                        f'  时间: {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
                        f'{"=" * 50}\n'
                    )
                    self.get_logger().warn(warn_msg)
                    # 同时在 stdout 打印（确保终端可见）
                    print(warn_msg, flush=True)

        # 清理已消失的目标（离开视野或重新站起）
        for track_id in list(self.fall_start_time.keys()):
            if track_id not in current_fall_ids:
                elapsed = time.time() - self.fall_start_time[track_id]
                if elapsed >= ALARM_DELAY_SEC and self.alarm_triggered.get(track_id, False):
                    self.get_logger().info(f'目标 (track_id={track_id}) 已移动或消失')
                del self.fall_start_time[track_id]
                self.alarm_triggered.pop(track_id, None)


def main(args=None):
    rclpy.init(args=args)
    node = FallMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
