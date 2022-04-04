/*
ネズミ学習問題用スケッチ（外部機器なし）
Copyright(c) 2018 Koji Makino and Hiromitsu Nishizaki All Rights Reserved.
*/
void setup() {
  Serial.begin(9600);
  while(!Serial){;}
  pinMode(LED_BUILTIN,OUTPUT);

}

void loop() {
  static int state=0;
  if(Serial.available()>0){
    int reward=0;
    char action = Serial.read();
    if(action=='a'){
      if(state==0)state=1;
      else state=0;
      digitalWrite(LED_BUILTIN,state);
    Serial.print(state);
    Serial.print(reward);
    }
    else if(action=='b'){
      if(state==1){
        reward=1;
      }
    Serial.print(state);
    Serial.print(reward);
    }
    else if(action=='c'){
      state=0;
      digitalWrite(LED_BUILTIN,state);
    }
  }
}

