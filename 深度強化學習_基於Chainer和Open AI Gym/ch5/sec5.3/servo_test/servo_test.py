# -*- coding: utf-8 -*-
"""
RaspberryPi用
RCサーボモータテスト用プログラム
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import Adafruit_PCA9685

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
while True:
    angle = input('[200-600]:')
    pwm.set_pwm(0,0,int(angle))
