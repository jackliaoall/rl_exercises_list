/*
シリアル通信テスト用スケッチ
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
*/
void setup() {
  Serial.begin(9600);//Leonardoだけ必要
  while(!Serial){;}
  pinMode(LED_BUILTIN,OUTPUT);
}

void loop() {
  static int led=0;
  if(Serial.available()>0){
    char a = Serial.read();
    digitalWrite(LED_BUILTIN,led);
    if(led==0)led=1;
    else led=0;
    Serial.print(led);
  }
}

