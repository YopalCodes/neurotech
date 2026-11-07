"""Checks the stepper wiring directly on the Pi, without the laptop or the command server.

    python3 motor_test.py

1. Lights LEDs A, B, C, D on the ULN2003 board one at a time (one per GPIO pin).
2. Spins the motor one way for 3 seconds, then the other way.
"""

from time import sleep

from gpiozero import OutputDevice

PINS = {17: "IN1 / LED A", 18: "IN2 / LED B", 27: "IN3 / LED C", 22: "IN4 / LED D"}
SEQUENCE = [(1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0),
            (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 1)]

outputs = {pin: OutputDevice(pin) for pin in PINS}

print("Step 1: one LED at a time")
for pin, name in PINS.items():
    print(f"  GPIO{pin} on: {name} should light")
    outputs[pin].on()
    sleep(1.5)
    outputs[pin].off()

for label, direction in (("one way", 1), ("the other way", -1)):
    print(f"Step 2: spinning {label} for 3 s")
    for i in range(1500):
        for pin, value in zip(PINS, SEQUENCE[(i * direction) % 8]):
            outputs[pin].value = value
        sleep(0.002)

for output in outputs.values():
    output.off()
print("Done")
