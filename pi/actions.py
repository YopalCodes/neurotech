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
REVERSE = True            # True: swapped for this motor's wiring ("right" was spinning left)
HALF_STEPS_PER_REV = 4096 # 28BYJ-48 output shaft, half-stepping (about 0.09 degrees per half-step)

# Half-step sequence for a unipolar stepper on a ULN2003: smoother and stronger than full steps.
SEQUENCE = (
    (1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0),
    (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 1),
)


class _NoCoil:
    """Stands in for a GPIO pin when there's no GPIO (e.g. testing on a laptop)."""
    value = 0

    def off(self):
        pass


class Stepper:
    """Steps the motor in a background thread: a command starts it spinning and it keeps going
    until told otherwise. When stopped, every coil is switched off so the motor and driver don't
    heat up (the shaft can then be turned by hand, which the step count can't see).

    `position` counts half-steps from "forward" (0). It is exact as long as the motor doesn't
    skip steps; zero() redefines forward. go_to() moves back to a position and stops there.
    """

    def __init__(self, pins, step_delay, simulated=False):
        if simulated:
            self.coils = [_NoCoil() for _ in pins]
        else:
            from gpiozero import OutputDevice
            self.coils = [OutputDevice(pin) for pin in pins]
        self.simulated = simulated
        self.step_delay = step_delay
        self.direction = 0   # +1, -1, or 0 for stopped
        self.position = 0    # half-steps from forward, in the motor's own direction
        self.target = None   # position go_to() is heading for, or None
        self._phase = 0
        self._wake = threading.Event()
        threading.Thread(target=self._run, name="stepper", daemon=True).start()

    def spin(self, direction):
        """+1 or -1 to spin, 0 to stop. Cancels go_to()."""
        self.target = None
        self.direction = direction
        self._wake.set()

    def go_to(self, position):
        """Moves to `position` (half-steps) the way it came, without wrapping around, then stops."""
        self.direction = 0
        self.target = position
        self._wake.set()

    def _run(self):
        next_step = time.perf_counter()
        while True:
            target = self.target
            if target is not None:
                gap = target - self.position
                direction = (gap > 0) - (gap < 0)
                if direction == 0:
                    self.target = None   # arrived
            else:
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
            self.position += direction
            next_step += self.step_delay
            delay = next_step - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_step = time.perf_counter()  # fell behind: carry on rather than rush to catch up


try:
    motor = Stepper(PINS, STEP_DELAY)
except Exception as e:  # not on a Pi, or gpiozero missing: simulate it, so the position still counts
    motor = Stepper(PINS, STEP_DELAY, simulated=True)
    log.warning("GPIO unavailable (%s): simulating the motor (its position is tracked, nothing moves)", e)


def _logical(steps):
    """Motor direction -> our direction (+ = right), allowing for REVERSE."""
    return -steps if REVERSE else steps


def angle():
    """Degrees from forward: positive = right of forward, negative = left."""
    return _logical(motor.position) * 360.0 / HALF_STEPS_PER_REV


def status():
    """Reported to the laptop 10 times a second by command_server.py."""
    if motor.target is not None:
        doing = "homing"
    elif motor.direction == 0:
        doing = "stopped"
    else:
        doing = "right" if _logical(motor.direction) > 0 else "left"
    return {"stepper_angle": round(angle(), 1), "stepper": doing}


def _spin(direction, description):
    log.info("Motor: %s", description)
    motor.spin(_logical(direction))


def stop():
    _spin(0, "stopped")


def left():
    _spin(-1, "spinning left")


def right():
    _spin(+1, "spinning right")


def center():
    _spin(0, "stopped (target centred)")


def zero():
    """'Set forward here': the motor's current direction becomes forward (0 degrees)."""
    log.info("Motor: forward set here (was %+.1f deg)", angle())
    motor.position = 0


def home():
    """'Return to forward': spins back to 0 degrees the way it came, then stops."""
    log.info("Motor: returning to forward from %+.1f deg", angle())
    motor.go_to(0)


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
    "zero": zero,
    "home": home,
    "up": up,
    "down": down,
    "squint": squint,
}
