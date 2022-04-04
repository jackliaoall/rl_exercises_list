#include <Servo.h>

Servo myservo;

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ;
  }
  myservo.attach(9);
  myservo.write(60);
}

void loop() {
  myservo.write(60);
  delay(500);
  myservo.write(30);
  delay(500);
  myservo.write(90);
  delay(500);
}
