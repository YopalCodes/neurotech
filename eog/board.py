"""BrainFlow connection helpers for the NeuroPawn Knight board."""

import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

import config


def connect(port=config.DEFAULT_PORT, synthetic=False, configure=True):
    """Open a session and start streaming. Returns (board, board_id).

    `synthetic=True` uses BrainFlow's fake board so the pipeline can be tested
    without hardware.
    """
    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    if synthetic:
        board_id = BoardIds.SYNTHETIC_BOARD.value
    else:
        board_id = config.BOARD_ID
        params.serial_port = port

    board = BoardShim(board_id, params)
    board.prepare_session()
    if configure and not synthetic:
        configure_channels(board)
    board.start_stream(45000)
    return board, board_id


def configure_channels(board):
    """Turn on the EOG channels and add them to the right-leg-drive (bias)."""
    for ch in (config.HORIZONTAL_CHANNEL, config.VERTICAL_CHANNEL):
        board.config_board(f"chon_{ch}_{config.CHANNEL_GAIN}")
        time.sleep(0.5)
        board.config_board(f"rldadd_{ch}")
        time.sleep(0.5)


def eog_rows(board_id):
    """Row indices in BrainFlow's data array for (horizontal, vertical) EOG."""
    exg = BoardShim.get_exg_channels(board_id)
    return exg[config.HORIZONTAL_CHANNEL - 1], exg[config.VERTICAL_CHANNEL - 1]
