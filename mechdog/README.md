# MechDog head control

Steer a [Hiwonder MechDog](https://docs.hiwonder.com/projects/MechDog/en/latest/) with
head tilt from the NeuroPawn Knight IMU (see [../eog](../eog)).

| Head | Dog |
|---|---|
| tilt forward | start walking (keeps going after you return to neutral) |
| tilt back | stop |
| tilt left / right while walking | turn left / right; more tilt = sharper turn |

## 1. Put the receiver on the dog (once)

[main.py](main.py) is a MicroPython program for the dog. It advertises over Bluetooth LE as
`MechDog_Head` and walks with `mechdog.move(stride, angle)` from Hiwonder's `HW_MechDog`
library.

1. Connect the dog to the PC with USB and open the **Hiwonder Python Editor**
   (docs section 4.1).
2. Open `main.py` and download it to the dog, then restart the dog.

This replaces the factory app-control program, so the Wonderbot phone app won't
connect while it's installed. To restore it, download `main.py` from Hiwonder's
"App Control Programs" zip the same way.

## 2. Drive it

Close the NeuroPawn Visualizer, then:

```
cd ../eog
python dog_control.py --port COM5            # finds the dog, calibrates head tilt, then drives
python dog_control.py --port COM5 --dry-run  # same, but only prints commands (no dog needed)
```

Follow the calibration prompts: hold still, then tilt forward, then tilt right.

Safety: the PC sends a command every 0.2 s. If the dog hears nothing for 1 s, or
the Bluetooth link drops, it stops. Speed and steering sensitivity are
`DOG_STRIDE`, `DOG_MAX_TURN`, `STEER_DEADZONE` and `STEER_FULL` in
[../eog/config.py](../eog/config.py).

## Protocol

Newline-terminated ASCII, written to characteristic
`6E400002-B5A3-F393-E0A9-E50E24DCCA9E` (service `6E400001-…`, the Nordic UART layout):

- `M,<stride>,<angle>` walks. Stride is in mm (−100…100) and angle is in degrees (−30…30; positive turns left).
- `S` stops.
