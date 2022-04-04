void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ;
  }
}

void loop() {
  Serial.print(digitalRead(4));
  Serial.print("\t");
  Serial.println(digitalRead(5));
  delay(500);
}

