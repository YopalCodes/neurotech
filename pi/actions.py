"""What the Pi does for each command. Edit this file for your hardware.

Each function handles one command. Values in the message's "args" are passed as
keyword arguments, so {"cmd": "move", "args": {"speed": 0.4}} calls move(speed=0.4).

stop() is also what the failsafe calls, so it must always leave the hardware safe.

Hardware: a stepper motor on a ULN2003 driver board (the board usually sold with
28BYJ-48 motors), wired like this:

    Driver board    Raspberry Pi 4
    IN1             GPIO17 (physical pin 11)
    IN2             GPIO18 (physical pin 12)
    IN3             GPIO27 (physical pin 13)
    IN4             GPIO22 (physical pin 15)
    +               5V (pin 2 or 4) for a 5 V motor, or a separate supply
    -               GND (pin 6), joined to the separate supply's ground if you use one

"left" spins the motor one way and "right" the other until another command arrives.
"stop", "center" (target centred, from the vision app) and the failsafe stop it.
"""

import logging
import threading
import time

log = logging.getLogger("actions")

PINS = (17, 18, 27, 22)   # IN1, IN2, IN3, IN4
STEP_DELAY = 0.002        # seconds per half-step: 2 ms is about 7 RPM on a 28BYJ-48. Smaller = faster;
                          # below about 1 ms it stalls.
REVERSE = False           # set to True if "left" and "right" spin the wrong way

# Half-step sequence for a unipolar stepper on a ULN2003: smoother and stronger than full steps.
SEQUENCE = (
    (1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0),
    (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 1),
)


class Stepper:
    """Steps the motor in a background thread: a command starts it spinning and it keeps going
    until told otherwise. When stopped, every coil is switched off so the motor and driver don't
    heat up (the shaft can then be turned by hand)."""

    def __init__(self, pins, step_delay):
        from gpiozero import OutputDevice

        self.coils = [OutputDevice(pin) for pin in pins]
        self.step_delay = step_delay
        self.direction = 0   # +1, -1, or 0 for stopped
        self._phase = 0
        self._wake = threading.Event()
        threading.Thread(target=self._run, name="stepper", daemon=True).start()

    def spin(self, direction):
        """+1 or -1 to spin, 0 to stop."""
        self.direction = direction
        self._wake.set()

    def _run(self):
        next_step = time.perf_counter()
        while True:
            direction = self.direction
            if direction == 0:
                for coil in self.coils:
                    coil.off()
                self._wake.wait()
                self._wake.clear()
                next_step = time.perf_counter()
                continue
            self._phase = (self._phase + direction) % len(SEQUENCE)
            for coil, on in zip(self.coils, SEQUENCE[self._phase]):
                coil.value = on
            next_step += self.step_delay
            delay = next_step - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_step = time.perf_counter()  # fell behind: carry on rather than rush to catch up


try:
    motor = Stepper(PINS, STEP_DELAY)
except Exception as e:  # not on a Pi, or gpiozero missing
    motor = None
    log.warning("GPIO unavailable (%s): motor commands will only be logged", e)


def _spin(direction, description):
    log.info("Motor: %s", description)
    if motor is not None:
        motor.spin(-direction if REVERSE else direction)


def stop():
    _spin(0, "stopped")


def left():
    _spin(-1, "spinning left")


def right():
    _spin(+1, "spinning right")


def center():
    _spin(0, "stopped (target centred)")


# From the EEG app (eye gestures). left and right above are shared with the vision app.

def up():
    log.info("[placeholder] up")


def down():
    log.info("[placeholder] down")


def squint():
    log.info("[placeholder] squint")


# Command name -> function. Commands not listed here are ignored.
HANDLERS = {
    "stop": stop,
    "left": left,
    "right": right,
    "center": center,
    "up": up,
    "down": down,
    "squint": squint,
}
