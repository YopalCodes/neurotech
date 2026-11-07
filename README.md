# Neurotech

Two laptop apps that control a Raspberry Pi 4 over Wi-Fi:

- **Vision app** (`laptop/`): the Pi streams its webcam (EMEET SmartCam C960) to the laptop,
  which runs YOLO object detection on its NVIDIA GPU and sends commands back.
- **EEG app** (`eeg/`): reads a NeuroPawn Knight EEG headset, learns five eye gestures
  (glance left, right, up, down, and squint), and sends them to the Pi as commands.

```
Pi 4 + webcam ──MJPEG video, port 8080──▶ Vision app (YOLO on GPU) ─┐
      ▲                                                             │
      └───────────────JSON commands, port 5005──────────────────────┤
                                                                    │
NeuroPawn Knight ──USB serial──▶ EEG app (gesture classifier) ──────┘
```

Run **one app at a time** with the Pi. The Pi takes commands from the newest connection, so
two apps connected at once would keep taking control from each other.

| Path | What it does | Edit it? |
|---|---|---|
| `pi/start_stream.sh` | Streams the webcam with `ustreamer` | Only to change resolution or FPS |
| `pi/command_server.py` | Receives commands; failsafe stop | No |
| `pi/actions.py` | **What the Pi does for each command** | Yes |
| `laptop/app.py` | Vision app window | No |
| `laptop/engine.py` | Video → YOLO → command pipeline | No |
| `laptop/decide.py` | **Turns detections into commands** | Yes |
| `laptop/teach.py`, `laptop/video_view.py` | Teach tab: photos, boxes, training | No |
| `laptop/train.py` | Trains the model of your taught objects | No |
| `laptop/library.py` | Stores taught objects in `library/` | No |
| `library/` | **Your taught objects: photos and trained model** | Through the app |
| `eeg/app.py` | EEG app window | No |
| `eeg/signals.py` | Filters, features, classifier | Only to tune detection |
| `eeg/sources.py` | NeuroPawn board (BrainFlow) and a simulated headset | No |

## Pi setup (Raspberry Pi OS)

```bash
sudo apt update
sudo apt install -y ustreamer v4l-utils git
git clone https://github.com/YopalCodes/neurotech.git
cd neurotech/pi
```

Check the webcam is detected and supports MJPEG, and find the Pi's IP address:

```bash
v4l2-ctl --list-devices                         # the C960 is usually /dev/video0
v4l2-ctl --list-formats-ext -d /dev/video0      # look for 'MJPG' at 640x480
hostname -I
```

Run each in its own terminal (the EEG app only needs the second):

```bash
bash start_stream.sh          # video:    http://<pi-ip>:8080/stream
python3 command_server.py     # commands: port 5005
```

Open `http://<pi-ip>:8080/stream` in the laptop's browser to check the video. Stream settings
can be changed per run, e.g. `RESOLUTION=1280x720 FPS=30 bash start_stream.sh`.

## Laptop setup (Windows)

Create the virtual environment **outside** OneDrive or any other synced folder. PyTorch with
CUDA is several GB, and syncing it is slow and pointless.

```powershell
py -3.11 -m venv $HOME\venvs\neurotech
& $HOME\venvs\neurotech\Scripts\Activate.ps1
```

If PowerShell refuses to run the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Install the CUDA build of PyTorch **first**, so pip doesn't install the CPU-only version.
Check https://pytorch.org/get-started/locally/ for the current command; at the time of writing:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r laptop/requirements.txt
pip install -r eeg/requirements.txt
python -c "import torch; print(torch.cuda.is_available())"    # must print True
```

## Vision app

```powershell
python laptop/app.py --pi 192.168.1.50        # the Pi's IP from `hostname -I`
python laptop/app.py --source 0               # test with the laptop's own webcam, no Pi needed
```

The window shows the live video with detection boxes, connection status, frame rate and AI
time per frame, a list of detected objects, and a log of every command sent.

- **AI control** is off at start, so nothing moves until you click it. Esc turns it off.
- **Target** chooses which object the AI follows; **Confidence** sets how sure it must be.
- **Stepper motor** (top of the Drive tab): hold **◀ Spin left** or **Spin right ▶** to turn the
  motor; let go to stop. This overrides the AI. The line under the buttons says if the Pi isn't
  connected.

The default model is `yolo11m` in FP16 on the GPU (about 20 ms per frame on an RTX 4060 laptop).
Options: `--model yolo11s.pt` (faster) or `yolo11l.pt` / `yolo11x.pt` (more accurate),
`--imgsz 960` for small or distant objects (with a higher stream resolution).

To change what the AI does, edit `laptop/decide.py`. The example follows the largest object of
the target type and sends `left`, `right`, `center`, or `stop`.

### Teaching it your own objects

The **Teach** tab trains the camera to recognise things outside YOLO's 80 built-in objects,
such as your own mug or a particular toy:

1. Type a name and click **Add**.
2. Hold the object in view and press **Space** (or **Take photo**). The frame freezes: drag a
   box tightly around the object and press **Enter** to save, or **Esc** to cancel. The next
   photo starts with the same box, so if the object hasn't moved, Space then Enter is enough.
3. Take at least 10 photos; 30+ is better. Turn the object, move it nearer and further, and
   change the background and lighting. Keep your other taught objects out of view: anything in
   a photo without a box teaches the model that it's background.
4. Click **Train**. It takes a few minutes on the GPU, and the video slows down meanwhile.
   When it finishes, the new objects appear at the top of **Target** and are detected
   alongside the built-in ones. The score (0 to 1) measures how well the model finds each
   object in photos held back from training; aim for 0.85 or more.

Taught objects are stored in `library/` and load automatically each time the app starts:

```
library/objects/<name>/*.jpg    photos (full camera frames)
library/objects/<name>/*.txt    the object's box in the photo of the same name
library/model.pt                the trained model; model.json records what it was trained on
```

They are ordinary files, so copy the folder to back it up or share it. Git ignores it, because
the photos show your surroundings. Training scratch files go in `~/.neurotech/training`, outside
OneDrive. You can also train from a terminal: `python laptop/train.py` (add `--epochs 100` to
train longer).

## EEG app

```powershell
python eeg/app.py
```

No headset yet? Choose **Simulator** and click Connect. The arrow keys make it glance,
S squints, and B blinks. It also acts out calibration cues by itself.

### Why eye gestures

Electrodes near the eyes pick up eye movements (EOG) and squinting (muscle activity) as large,
clear signals. That makes these five gestures reliable. Detecting *imagined* movements
("think left") with an 8-channel headset is far less accurate, so the app doesn't attempt it.

### Electrodes and connection

Place the electrodes close to the eyes (also shown by the app's **Electrode placement** button):

| Channel | Position |
|---|---|
| Ch1 | Right temple, beside the outer corner of the right eye |
| Ch2 | Left temple, beside the outer corner of the left eye |
| Ch3 | Forehead, just above the right eyebrow |
| Ch4 | Cheekbone, just below the right eye |
| Ch5–Ch8 | Optional: forehead or anywhere on the scalp |

Reference and ground go where NeuroPawn's instructions say (usually an earlobe or the bone
behind the ear). If your headset has fixed positions, that's fine: the app learns from whatever
placement you calibrate with. Recalibrate whenever you move the electrodes.

Plug in the Knight board, choose **NeuroPawn Knight** (or **Knight IMU**), pick its COM port
(Windows Device Manager → Ports shows which one), and click Connect.

### Calibrate, then use

1. Click **Start calibration** (10 trials per gesture takes about 4 minutes). Keep your head
   still and look at the centre between cues. When cued, glance quickly and look straight
   back. On "rest" cues, relax and blink once or twice; this teaches the app to ignore blinks.
2. The app trains on the recording and shows its **accuracy on trials it didn't train on**,
   with a score for each gesture. Aim for 90%+. If it's low, check electrode contact and
   recalibrate.
3. Detection then runs live. A gesture fires once per glance or squint, about 0.3–0.5 s after
   you make it. Raise **Certainty** if it fires by mistake; lower it if it misses gestures.
   **Min. gap** sets the shortest time between commands.
4. Tick **Send detected commands to the Pi** and enter the Pi's address. Each detected
   gesture is sent for **Each command lasts** seconds, then the Pi gets `stop`. The
   **Left** / **Right** buttons below spin the motor while held.

The last calibration loads automatically next time. Each calibration's recording and model
are saved in `eeg/recordings/` and `eeg/models/`; git ignores these folders.

The app uses the standard `brainflow` package from pip, which already supports the Knight
board (125 Hz, 8 channels). NeuroPawn's fork (github.com/NeuroPawn/brainflow) differs only in
its microvolt scale factor; that changes the numbers on the plots, not detection, because the
app learns from your own recordings.

## Pi commands and safety

Commands are JSON objects, one per line, such as `{"cmd": "left"}`, or with arguments
`{"cmd": "move", "args": {"speed": 0.4}}`, which calls `move(speed=0.4)` in `pi/actions.py`.

| Command | Sent by | What the Pi does |
|---|---|---|
| `left`, `right` | Both apps | Spins the stepper motor one way or the other until the next command |
| `center` | Vision app | Stops the motor (the target is centred) |
| `up`, `down`, `squint` | EEG app | Nothing yet: placeholders in `pi/actions.py` |
| `stop` | Both apps, whenever nothing else is happening | Stops the motor and switches its coils off |

### Stepper motor wiring

`pi/actions.py` drives a stepper motor through a ULN2003 driver board (the 4-input board
usually sold with 28BYJ-48 motors):

| Driver board | Raspberry Pi 4 |
|---|---|
| IN1 | GPIO17 (physical pin 11) |
| IN2 | GPIO18 (pin 12) |
| IN3 | GPIO27 (pin 13) |
| IN4 | GPIO22 (pin 15) |
| + | 5V (pin 2 or 4) for a 5 V motor, or a separate supply |
| − | GND (pin 6), joined to the separate supply's ground if you use one |

To spin the motor by hand, hold a spin button in either app: in the Stepper motor panel at the
top of the vision app's Drive tab, or under Raspberry Pi in the EEG app (tick "Send detected commands to the
Pi" first). It spins while the button is held and stops when you let go.

At the top of `pi/actions.py`: `STEP_DELAY` sets the speed (2 ms per half-step is about 7 RPM
on a 28BYJ-48; it stalls below about 1 ms), and `REVERSE = True` swaps the directions if left
and right turn the wrong way. While stopped, the coils are switched off so the motor stays cool,
which also means the shaft can be turned by hand.

The apps resend the current command about 10–30 times a second. If the Pi hears nothing for
0.5 s (an app crashed or Wi-Fi dropped), or the app disconnects, it calls `stop()`. `stop()` must
always leave the hardware safe.
