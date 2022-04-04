/*
ネズミ学習問題用スケッチ（全部入り）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
*/
#include <Servo.h>

Servo myservo;

void setup() {
  Serial.begin(9600);
  while(!Serial){;}
  pinMode(4,INPUT);
  pinMode(5,INPUT);
  pinMode(6,OUTPUT);

  myservo.attach(9);
  myservo.write(60);
}

void loop() {
  static int state=0;
  if(digitalRead(4)==HIGH){
      if(state==0)state=1;
      else state=0;
      digitalWrite(13,state);
  }
  if(digitalRead(5)==HIGH){
      if(state==1){
        myservo.write(120);
        delay(2000);
        myservo.write(60);
      }
  }
}

