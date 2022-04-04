# coding:utf-8
"""
カメラテスト用プログラム
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
"""
import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    if cv2.waitKey(10) == 115:
        cv2.imwrite('camera.png', gray)
    if cv2.waitKey(10) == 113:
        break
cap.release()
