"""Drive a Hiwonder MechDog with head tilt from the Knight IMU.

    tilt forward     -> start walking (keeps walking after you return to neutral)
    tilt back        -> stop
    tilt left/right  -> steer while walking, more tilt = sharper turn

    python dog_control.py --port COM5
    python dog_control.py --port COM5 --dry-run   # print commands, no dog

The dog must be running ../mechdog/main.py. Commands go over Bluetooth LE
every 0.2 s; the dog stops by itself if they stop arriving for 1 s, so it
halts if this script crashes or the laptop goes out of range.
"""

import argparse
import asyncio
import time

import numpy as np

import board as eog_board
import config
from imu import HeadTilt

RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # dog's command characteristic
STEP = 0.1
HEARTBEAT = 0.2


def steer_angle(roll):
    """Head roll (deg, + = right) -> MechDog turn angle (deg, + = left)."""
    span = config.STEER_FULL - config.STEER_DEADZONE
    amount = np.clip((abs(roll) - config.STEER_DEADZONE) / span, 0.0, 1.0)
    return int(round(-np.sign(roll) * amount * config.DOG_MAX_TURN))


class Steering:
    def __init__(self):
        self.walking = False

    def update(self, change, roll):
        """Return the (stride, angle) to send for this head state."""
        if change == "forward":
            self.walking = True
        elif change == "back":
            self.walking = False
        if not self.walking:
            return (0, 0)
        return (config.DOG_STRIDE, steer_angle(roll))


class DogLink:
    def __init__(self, dry_run):
        self.dry_run = dry_run
        self.client = None

    async def connect(self):
        if self.dry_run:
            print("Dry run: commands are printed, not sent.")
            return
        from bleak import BleakClient, BleakScanner

        print(f"Looking for {config.DOG_NAME} over Bluetooth...")
        device = await BleakScanner.find_device_by_name(config.DOG_NAME, timeout=15)
        if device is None:
            raise RuntimeError(f"{config.DOG_NAME} not found. Is the dog on and running mechdog/main.py?")
        self.client = BleakClient(device)
        await self.client.connect()
        print(f"Connected to {config.DOG_NAME} ({device.address})")

    async def send(self, stride, angle):
        line = "S\n" if stride == 0 else f"M,{stride},{angle}\n"
        if self.client:
            await self.client.write_gatt_char(RX_UUID, line.encode(), response=False)

    async def close(self):
        if self.client and self.client.is_connected:
            await self.send(0, 0)
            await self.client.disconnect()


async def run(args):
    dog = DogLink(args.dry_run)
    await dog.connect()
    board, board_id = eog_board.connect(args.port)
    try:
        tilt = HeadTilt(board, board_id)
        tilt.calibrate()
        print("Tilt forward to walk, back to stop, left/right to steer. Ctrl+C to quit.")
        steering = Steering()
        last_cmd, last_sent = None, 0.0
        while True:
            await asyncio.sleep(STEP)
            change = tilt.update()
            pitch, roll = tilt.angles()
            cmd = steering.update(change, roll)
            now = time.monotonic()
            if cmd != last_cmd or now - last_sent >= HEARTBEAT:
                await dog.send(*cmd)
                last_sent = now
            if cmd != last_cmd:
                action = "STOP" if cmd[0] == 0 else ("WALK" if cmd[1] == 0 else
                         f"TURN {'LEFT' if cmd[1] > 0 else 'RIGHT'} {abs(cmd[1])} deg")
                print(f"{time.strftime('%H:%M:%S')}  {action:18s} (head forward/back {pitch:+.0f}, left/right {roll:+.0f})")
                last_cmd = cmd
    finally:
        await dog.close()
        board.stop_stream()
        board.release_session()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=config.DEFAULT_PORT)
    parser.add_argument("--dry-run", action="store_true", help="don't connect to the dog, just print")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
