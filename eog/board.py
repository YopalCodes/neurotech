"""BrainFlow connection helpers for the NeuroPawn Knight board."""

import time

from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

import config


def connect(port=config.DEFAULT_PORT, synthetic=False, configure=True, channels=None):
    """Open a session and start streaming. Returns (board, board_id).

    `synthetic=True` uses BrainFlow's fake board so the pipeline can be tested
    without hardware. `channels` defaults to the horizontal and vertical EOG
    channels from config.
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
    try:
        if not synthetic:
            # Opening the port resets the board; commands sent while it boots are lost.
            time.sleep(config.BOOT_SECONDS)
        if configure and not synthetic:
            configure_channels(board, channels)
        board.start_stream(45000)
        if not synthetic:
            wait_for_data(board)
    except Exception:
        board.release_session()  # free the serial port so the next run can open it
        raise
    return board, board_id


def wait_for_data(board, timeout=5.0):
    start = time.monotonic()
    while board.get_board_data_count() == 0:
        if time.monotonic() - start > timeout:
            raise RuntimeError(
                "No data from the board. Close the NeuroPawn Visualizer, unplug and "
                "replug the board, and try again."
            )
        time.sleep(0.1)


def send_command(board, command, retries=10):
    """Send a config command, resending it while the board is still booting.

    The Knight's reply can contain non-UTF-8 bytes, which makes BrainFlow's
    Python wrapper fail to decode it even though the command was already sent,
    so an undecodable reply is ignored. Sometimes the board takes much longer
    than BOOT_SECONDS to find its IMU; a command sent then is answered with
    the boot message and lost.
    """
    for _ in range(retries):
        try:
            reply = board.config_board(command)
        except UnicodeDecodeError:
            reply = ""
        if "Scanning for IMU" not in reply:
            print(f"{command}: sent")
            time.sleep(0.5)
            return
        print(f"{command}: board still booting, retrying...")
        time.sleep(1.0)
    raise RuntimeError("Board never finished booting (stuck on 'Scanning for IMU'). Unplug and replug it.")


def configure_channels(board, channels=None):
    """Turn on the EOG channels and add them to the right-leg-drive (bias)."""
    for ch in channels or (config.HORIZONTAL_CHANNEL, config.VERTICAL_CHANNEL):
        send_command(board, f"chon_{ch}_{config.CHANNEL_GAIN}")
        send_command(board, f"rldadd_{ch}")


def eog_rows(board_id):
    """Row indices in BrainFlow's data array for (horizontal, vertical) EOG."""
    exg = BoardShim.get_exg_channels(board_id)
    return exg[config.HORIZONTAL_CHANNEL - 1], exg[config.VERTICAL_CHANNEL - 1]
