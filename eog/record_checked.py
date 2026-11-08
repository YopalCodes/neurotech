"""record.py, but first checks the EOG channels are really on (see channel_check.py).

    python record_checked.py --port COM10 --trials 20

Same options as record.py. Only the connect step differs: if a channel stays off, it stops
before the session instead of recording 6 minutes without it.
"""

import board as eog_board
import channel_check
import record

eog_board.connect = channel_check.connect   # record.py calls eog_board.connect(...)

if __name__ == "__main__":
    record.main()
