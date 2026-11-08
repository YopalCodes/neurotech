"""Shared settings for EOG recording, training, and live classification."""

from brainflow.board_shim import BoardIds

# Board
BOARD_ID = BoardIds.NEUROPAWN_KNIGHT_BOARD_IMU.value  # boot message says "IMU found"
DEFAULT_PORT = "COM5"
BOOT_SECONDS = 4  # the board reboots whenever the port opens; streams ~2.7 s later

# Knight board channels (1-based, as labelled on the board) that carry EOG.
# Horizontal: electrodes on the outer corners of the left and right eyes.
# Vertical:   electrodes above and below one eye (blinks show up here).
HORIZONTAL_CHANNEL = 6
VERTICAL_CHANNEL = 5
CHANNEL_GAIN = 12  # Knight PGA gain sent as "chon_<ch>_<gain>"

# Knight IMU rows in BrainFlow's data array. BrainFlow only lists them as
# "other" channels; these were identified from recordings (accel norm = 9.4 m/s²,
# gyro in rad/s). Rows 17-19 appear to be the magnetometer and aren't used.
ACCEL_ROWS = (11, 12, 13)  # m/s²
GYRO_ROWS = (14, 15, 16)   # rad/s

# Head tilt
TILT_ENTER_DEG = 15     # tilt past this to register left/right/forward/back
TILT_EXIT_DEG = 10      # come back inside this to return to neutral
CAL_HOLD_SECONDS = 3    # how long to hold still for zeroing
CAL_TILT_DEG = 15       # minimum tilt accepted during the calibration tilts
IMU_ALPHA = 0.02        # complementary filter: weight of the accelerometer per sample
HEAD_MOTION_GATE = 60   # deg/s; ignore eye events while the head turns faster than this

# Event labels. Marker values are written into BrainFlow's marker channel.
LABELS = {1: "rest", 2: "left", 3: "right", 4: "blink"}
MARKERS = {name: value for value, name in LABELS.items()}

# Epoch window relative to the cue, in seconds. Saccades start ~200 ms after
# the cue and the return-to-center saccade finishes within ~1.3 s.
EPOCH_START = 0.0
EPOCH_END = 1.5

# Filtering (Hz). EOG lives below ~10 Hz; the high-pass removes electrode drift.
# A 4th-order 10 Hz low-pass already knocks 60 Hz mains down by ~60 dB.
HIGHPASS = 0.05
LOWPASS = 10.0

# Training augmentation: extra copies of each epoch shifted by these amounts
# (seconds) so the classifier tolerates events that aren't perfectly aligned,
# which is what it sees on the sliding windows in live mode.
SHIFTS = (-0.2, -0.1, 0.0, 0.1, 0.2)
# Windows this far from the cue only catch part of an event (e.g. just the
# return-to-center saccade). They are labelled "rest" so live mode waits until
# a whole event is in view instead of guessing from a fragment.
MISALIGNED_SHIFTS = (-1.0, -0.8, -0.6, 0.5, 0.7, 0.9)

# Recording session timing (seconds).
CUE_DURATION = 1.0  # how long the target dot stays at the side
ITI_MIN = 2.0       # gap between trials, randomized to avoid anticipation
ITI_MAX = 3.0
TRIALS_PER_CLASS = 20
