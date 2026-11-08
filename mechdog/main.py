# MechDog head-control receiver (MicroPython, runs on the MechDog's ESP32).
#
# Advertises over Bluetooth LE as "MechDog_Head" with a Nordic-UART-style
# service and accepts newline-terminated ASCII commands:
#
#   M,<stride>,<angle>   walk: stride -100..100 mm, angle -30..30 deg (+ = left)
#   S                    stop
#
# Safety: if no command arrives for TIMEOUT_MS (the PC sends one every 200 ms),
# or the PC disconnects, the dog stops.
#
# Upload this as main.py with the Hiwonder Python Editor. It replaces the
# factory app-control program; to restore that, re-upload main.py from
# Hiwonder's "App Control Programs" download.

import time

import bluetooth
from micropython import const

from HW_MechDog import MechDog

NAME = "MechDog_Head"
TIMEOUT_MS = 1000

_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)

_UART_UUID = bluetooth.UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
_UART_TX = (bluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E"), bluetooth.FLAG_READ | bluetooth.FLAG_NOTIFY)
_UART_RX = (bluetooth.UUID("6E400002-B5A3-F393-E0A9-E50E24DCCA9E"), bluetooth.FLAG_WRITE | bluetooth.FLAG_WRITE_NO_RESPONSE)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def parse(line):
    """Return (stride, angle) for a command line, or None if it isn't valid."""
    parts = line.strip().split(",")
    if parts[0] == "S":
        return (0, 0)
    if parts[0] == "M" and len(parts) == 3:
        try:
            return (clamp(int(parts[1]), -100, 100), clamp(int(parts[2]), -30, 30))
        except ValueError:
            return None
    return None


def advertising_payload(name):
    n = name.encode()
    return bytes((2, 0x01, 0x06)) + bytes((len(n) + 1, 0x09)) + n


class Receiver:
    def __init__(self):
        self.ble = bluetooth.BLE()
        self.ble.active(True)
        self.ble.irq(self.irq)
        ((self.tx, self.rx),) = self.ble.gatts_register_services(((_UART_UUID, (_UART_TX, _UART_RX)),))
        self.ble.gatts_set_buffer(self.rx, 64, True)
        self.buffer = ""
        self.command = None
        self.connected = False
        self.last_ms = time.ticks_ms()
        self.advertise()

    def advertise(self):
        self.ble.gap_advertise(100000, adv_data=advertising_payload(NAME))

    def irq(self, event, data):
        if event == _IRQ_CENTRAL_CONNECT:
            self.connected = True
        elif event == _IRQ_CENTRAL_DISCONNECT:
            self.connected = False
            self.command = (0, 0)
            self.advertise()
        elif event == _IRQ_GATTS_WRITE:
            self.buffer += self.ble.gatts_read(self.rx).decode()
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                cmd = parse(line)
                if cmd is not None:
                    self.command = cmd
                    self.last_ms = time.ticks_ms()


def main():
    dog = MechDog()
    dog.set_default_pose()
    rx = Receiver()
    current = (0, 0)
    while True:
        target = rx.command if rx.command is not None else current
        if time.ticks_diff(time.ticks_ms(), rx.last_ms) > TIMEOUT_MS:
            target = (0, 0)
        if target != current:
            dog.move(target[0], target[1])
            current = target
        time.sleep_ms(50)


main()
