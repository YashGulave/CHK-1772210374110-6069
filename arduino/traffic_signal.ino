int redLED = 5;
int greenLED = 12;

String input = "";

void setup() {

  pinMode(redLED, OUTPUT);
  pinMode(greenLED, OUTPUT);

  Serial.begin(9600);

  // Start with RED
  digitalWrite(redLED, HIGH);
  digitalWrite(greenLED, LOW);
}

void loop() {

  if (Serial.available()) {

    input = Serial.readStringUntil('\n');

    int signalTime = input.toInt();

    Serial.print("Received Time: ");
    Serial.println(signalTime);

    // GREEN SIGNAL
    digitalWrite(greenLED, HIGH);
    digitalWrite(redLED, LOW);

    delay(signalTime * 1000);

    // RED SIGNAL
    digitalWrite(greenLED, LOW);
    digitalWrite(redLED, HIGH);
  }
}
