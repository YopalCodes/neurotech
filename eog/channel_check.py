"""Makes sure the EOG channels are really on after board.connect().

board.connect() sends "chon_<ch>_<gain>" and "rldadd_<ch>" for each EOG channel, but the Knight
sometimes ignores one of them. That channel then sends exactly 0 for the whole session, and
nothing reports it: a 20-trials-per-gesture recording was made with the vertical channel off.

ensure_channels() checks every EOG channel has signal, re-sends the settings for any that
doesn't, and raises an error if it still doesn't come on.

    python record_checked.py --port COM10 --trials 20    # record.py, with this check first
"""

import time

import numpy as np
from brainflow.board_shim import BoardShim

import board as eog_board
import config

_aram_connect = eog_board.connect   # kept, so record_checked.py can swap eog_board.connect for ours


def dead_channels(board, board_id, channels, seconds=1.0):
    """EOG channels (1-based) whose last `seconds` of data are all exactly 0."""
    exg = BoardShim.get_exg_channels(board_id)
    fs = BoardShim.get_sampling_rate(board_id)
    data = board.get_current_board_data(int(seconds * fs))
    if data.shape[1] < fs // 2:
        return []   # not enough data to judge yet
    return [ch for ch in channels if np.all(data[exg[ch - 1]] == 0)]


def ensure_channels(board, board_id, channels=None, tries=3):
    channels = list(channels or (config.HORIZONTAL_CHANNEL, config.VERTICAL_CHANNEL))
    for attempt in range(tries + 1):
        time.sleep(1.5)
        dead = dead_channels(board, board_id, channels)
        if not dead:
            print(f"Channels {channels}: all sending data.", flush=True)
            return
        if attempt == tries:
            break
        print(f"Channel(s) {dead} not sending data; re-sending their settings...", flush=True)
        for ch in dead:
            eog_board.send_command(board, f"chon_{ch}_{config.CHANNEL_GAIN}")
            eog_board.send_command(board, f"rldadd_{ch}")
    raise RuntimeError(f"EOG channel(s) {dead} still send no data after {tries} retries. "
                       "Unplug and replug the board, and try again.")


def connect(port=config.DEFAULT_PORT, synthetic=False, configure=True, channels=None):
    """board.connect(), then ensure_channels(). Releases the board if the check fails."""
    board, board_id = _aram_connect(port, synthetic, configure, channels)
    if synthetic or not configure:
        return board, board_id
    try:
        ensure_channels(board, board_id, channels)
    except Exception:
        board.stop_stream()
        board.release_session()
        raise
    return board, board_id
