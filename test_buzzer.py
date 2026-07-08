#!/usr/bin/env python3
"""蜂鸣器测试：鸣响 1 秒后停止"""
import Hobot.GPIO as GPIO
import time

BUZZER_PIN = 11  # BOARD 编号

GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

print("蜂鸣器测试 (低电平触发)")
print("→ 蜂鸣 1 秒...")
GPIO.output(BUZZER_PIN, GPIO.LOW)  # 低电平 → 响
time.sleep(1)
GPIO.output(BUZZER_PIN, GPIO.HIGH)  # 高电平 → 停
print("→ 停止")

GPIO.cleanup()
print("完成")
