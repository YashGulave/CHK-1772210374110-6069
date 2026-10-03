/*
  AI Traffic Light Controller for Arduino IDE
  
  Description:
  This Arduino sketch interfaces with the Python YOLOv8 Traffic Detection system (`main.py`).
  It receives dynamic GREEN light durations (in seconds) over Serial at 9600 baud rate.

  Circuit Wiring:
  - Red LED    : Digital Pin 12 (with 220 ohm resistor -> GND)
  - Yellow LED : Digital Pin 11 (with 220 ohm resistor -> GND)
  - Green LED  : Digital Pin 10 (with 220 ohm resistor -> GND)
  - Buzzer (Opt): Digital Pin 8 (Optional for violation alert)
*/

// Pin Definitions
const int RED_PIN    = 12;
const int YELLOW_PIN = 11;
const int GREEN_PIN  = 10;
const int BUZZER_PIN = 8; // Optional audio indicator

// Yellow light transition duration in seconds
const int YELLOW_DURATION_SEC = 2;

void setup() {
  // Initialize Serial Communication at 9600 Baud
  Serial.begin(9600);
  
  // Set LED pins as Outputs
  pinMode(RED_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Default state: RED ON (Waiting for Python signal)
  setTrafficLight(HIGH, LOW, LOW);
  
  Serial.println("Arduino Traffic Controller Ready.");
}

void loop() {
  // Check if signal time duration data is received over Serial
  if (Serial.available() > 0) {
    // Read signal time sent from Python main.py (in seconds)
    int greenDuration = Serial.parseInt();

    // If valid green duration is received
    if (greenDuration > 0) {
      Serial.print("Received Green Duration: ");
      Serial.print(greenDuration);
      Serial.println(" seconds.");

      // --- 1. GREEN LIGHT ACTIVE ---
      setTrafficLight(LOW, LOW, HIGH);
      
      // Countdown for Green Light
      for (int i = greenDuration; i > 0; i--) {
        delay(1000); // Wait 1 second
      }

      // --- 2. YELLOW LIGHT TRANSITION ---
      setTrafficLight(LOW, HIGH, LOW);
      delay(YELLOW_DURATION_SEC * 1000);

      // --- 3. RED LIGHT RETURN ---
      setTrafficLight(HIGH, LOW, LOW);
      Serial.println("Cycle completed. Signal returned to RED.");
    }
  }
}

// Helper function to set traffic light states
void setTrafficLight(int redState, int yellowState, int greenState) {
  digitalWrite(RED_PIN, redState);
  digitalWrite(YELLOW_PIN, yellowState);
  digitalWrite(GREEN_PIN, greenState);
}
