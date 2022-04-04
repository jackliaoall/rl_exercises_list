/*
ネズミ学習問題用スケッチ（センサ．RCサーボモータ）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
*/
#include <Servo.h>

Servo myservo;

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ;
  }
  pinMode(12, OUTPUT);
  pinMode(13, OUTPUT);
  myservo.attach(9);
  myservo.write(60);
}

void loop() {
  static int state = 0;
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    int reward = 0;
    char action = Serial.read();
    if (action == 'a') {
      myservo.write(0);
      delay(1000);
      int a = digitalRead(3);
      if (a == LOW) {
        if (state == 0) {
          state = 1;
        }
        else {
          state = 0;
        }
      }
      digitalWrite(13, state);
      Serial.print(state);
      Serial.print(reward);
      myservo.write(60);
      delay(1000);
    }
    else if (action == 'b') {
      myservo.write(120);
      delay(1000);
      int b = digitalRead(4);
      if (b == LOW) {
        if (state == 1) {
          reward = 1;
        digitalWrite(12, HIGH);
        }
      }
      digitalWrite(12, LOW);
      Serial.print(state);
      Serial.print(reward);
      myservo.write(60);
      delay(1000);
    }
    else if (action == 'c') {
      state = 0;
      digitalWrite(13, state);
      delay(1000);
    }
  }
}
