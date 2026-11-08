"""Head tilt only: calibrate, then show tilt angles and report left/right/forward/back.

    python head_tilt.py --port COM5
"""

import argparse
import time

import board as eog_board
import config
from imu import HeadTilt


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=config.DEFAULT_PORT)
    args = parser.parse_args()

    board, board_id = eog_board.connect(args.port)
    try:
        tilt = HeadTilt(board, board_id)
        tilt.calibrate()
        print("Tracking head tilt (Ctrl+C to stop)")
        while True:
            time.sleep(0.1)
            change = tilt.update()
            pitch, roll = tilt.angles()
            status = f"forward/back {pitch:+6.1f} deg   left/right {roll:+6.1f} deg   [{tilt.state.upper():7s}]"
            if change:
                print(f"\r{status}   {time.strftime('%H:%M:%S')}  HEAD {change.upper()}")
            else:
                print(f"\r{status}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        board.stop_stream()
        board.release_session()


if __name__ == "__main__":
    main()
