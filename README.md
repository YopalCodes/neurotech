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
| `laptop/devices_panel.py`, `laptop/device_window.py` | Devices tab: connect to and control other computers and phones | No |
| `remote/` | Device control: phones over ADB, computers running `agent.py` (see `remote/README.md`) | No |
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

```
python laptop/app.py --pi powershell192.168.1.50        # the Pi's IP from `hostname -I`
python laptop/app.py --source 0               # test with the laptop's own webcam, no Pi needed
```

The window shows the live video with detection boxes, connection status, frame rate and AI
time per frame, a list of detected objects, and a log of every command sent.

- **AI control** is off at start, so nothing moves until you click it. Esc turns it off.
- **Target** chooses which object the AI follows; **Confidence** sets how sure it must be.
- **Stepper motor** (top of the Drive tab): hold **◀ Spin left** or **Spin right ▶** to turn the
  motor; let go to stop. This overrides the AI. The line under the buttons says if the Pi isn't
  connected. Below them, a dial and "Facing: 35° right of forward" show which way the stepper
  points (the Pi counts its steps; also in the Status box as "Stepper"). **Set forward here** makes
  the current direction forward; **↺ Return to forward** spins it back to forward and stops. Forward
  starts as wherever the stepper was when `command_server.py` started, and turning the shaft by hand
  while it's stopped isn't counted.
- **Eyes tab** (start the app with `--eog-port COM10`): turn on **Eye control** to steer the motor
  with the NeuroPawn headset: look right or left to spin that way, blink to stop. It uses Aram's
  EOG pipeline and your trained model (see the EOG section below), runs in its own process so it
  doesn't slow the camera AI, and connects to the headset once at startup. Eye control and AI
  control can't both be on; a held Spin button overrides both; Esc turns both off.
  `--eog-port playback:eog/data/<recording>.csv` replays a recording instead of the headset.
  Head tilt (the headset's IMU) works as in Aram's `live.py`: after connecting, follow the
  calibration prompts in the terminal (hold still, tilt forward, tilt right). The Eyes tab then
  shows your head tilt, and eye events are ignored while your head turns fast, so turning your
  head doesn't register as a glance. `--no-imu` skips it.
- **Dog tab** (start with `--eog-port COM10 --dog`): drive Aram's MechDog with head tilt. Turn on
  **Head drives the dog**, then tilt forward to walk (it keeps walking), back to stop, and left or
  right to steer (more tilt = sharper turn). **STOP DOG** or Esc stops it. Uses Aram's
  `eog/dog_control.py` over Bluetooth; the dog must be running `mechdog/main.py` (see below).
  `--dog-dry-run` shows the commands without a dog.

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

### Devices tab: control other computers and phones

The **Devices** tab lets the desktop app take over another computer or an Android phone. It
uses `remote/`, from Aram's device-control branch. When you connect, a window opens with the
device's live screen, and your mouse and keyboard drive it.

**Computers** (Windows, macOS, Linux), on the same network:
1. On the computer, run `pip install -r remote/requirements.txt`.
2. Then run `python remote/agent.py --password <at least 8 characters>`. It prints its IP address.
   On Windows, allow it through the firewall.
3. In the app, click **Add PC...** and enter that address.
4. Double-click the PC, or select it and click **Connect & open screen**, and enter the password.

The link is encrypted, and the agent refuses connections without the password. Its console shows
when someone is controlling it. Ctrl+C there stops sharing.

**Android phones:** no app is installed on the phone.
1. Install Android platform-tools on the laptop: `winget install Google.PlatformTools`.
2. On the phone, turn on USB debugging, then plug it in and allow this computer.
3. Click **Find phones**.

To control a phone over Wi-Fi, select it while it's plugged in, click **Phone → Wi-Fi**, then
unplug it. Alternatively, use **Pair phone...** with Android's Wireless debugging. In the phone
window:

| Mouse | Phone |
|---|---|
| click | tap |
| drag | swipe |
| right-click | Back |
| wheel | scroll |

Typing works too, and the toolbar has Home, Recents and Open app.

One device is controlled at a time. Closing its window disconnects it. `remote/README.md` has
more detail, and `python remote/controller.py` runs the same control on its own, without the
vision app.

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

## EOG motor control (NeuroPawn, Aram's classifier)

`eog/` is Aram's EOG pipeline from the `eog-brainflow` branch (see `eog/README.md` for
recording, training and electrode placement). `eog/motor_control.py` runs his live detection
(`board.py`, `features.py`, the model from `train.py`, and `live.py`'s timing) and sends the
result to the Pi. One difference from `live.py`: an event fires as soon as one check passes the
certainty threshold, instead of three in a row, so short blinks aren't missed.

| Eyes | Motor |
|---|---|
| Look right | Spins right, and keeps spinning |
| Look left | Spins left, and keeps spinning |
| Blink | Stops |

```powershell
cd eog
python motor_control.py --port COM10 --pi <the Pi's IP>   # the headset's COM port (Device Manager → Ports)
python motor_control.py --synthetic --pi <the Pi's IP>    # no headset
```

It handles the board exactly like `live.py`: it connects once at startup and releases the board
when you close it. If connecting fails, or the board stops sending data, close it, unplug and
replug the board, and run it again. Close the NeuroPawn Visualizer first (only one program can
use the port). The window shows the motor state, the detection probabilities and the Pi
connection. Its Left / Stop / Right buttons
(or the ← → keys, and B or Space to stop) control the motor without the headset. The motor state
is sent to the Pi 10 times a second, so the Pi's failsafe stops the motor if this program,
the headset or the Wi-Fi stops.

Head tilt (the IMU) works as in Aram's `live.py`: after connecting, follow the calibration prompts
in the terminal (hold still, tilt forward, tilt right). The window shows your head tilt, and eye
events are ignored while your head turns faster than 60°/s and for 1.5 s after. `--no-imu` skips it.

The model in `eog/models/` is trained on one person's recordings in `eog/data/` (Aram's original
examples and model are kept in `eog/data/aram_examples/` and `eog/models/aram_*`). For good
accuracy on someone else, record and train with `record.py` and `train.py` first, as his README
describes.

## MechDog (Aram's head-tilt control)

`mechdog/main.py` is Aram's MicroPython program for the Hiwonder MechDog. It shows up over
Bluetooth as `MechDog_Head`, walks on `M,<stride>,<angle>` commands, stops on `S`, and stops by
itself if no command arrives for 1 s. Put it on the dog once: connect the dog with USB, open it in
the **Hiwonder Python Editor**, download it, and restart the dog (details in `mechdog/README.md`;
it replaces the factory phone-app program until you re-upload that).

Then drive the dog from the desktop app's **Dog tab** (`--eog-port COM10 --dog`), or with Aram's
script on its own: `cd eog` then `python dog_control.py --port COM10`. Speed and steering are
`DOG_STRIDE`, `DOG_MAX_TURN`, `STEER_DEADZONE` and `STEER_FULL` in `eog/config.py`. The Dog tab
turns up to 30° (the dog's maximum) and shortens the stride while turning, down to
`DOG_TURN_STRIDE` (40 mm) at a full turn, for a tighter circle.

## Pi commands and safety

Commands are JSON objects, one per line, such as `{"cmd": "left"}`, or with arguments
`{"cmd": "move", "args": {"speed": 0.4}}`, which calls `move(speed=0.4)` in `pi/actions.py`.

| Command | Sent by | What the Pi does |
|---|---|---|
| `left`, `right` | Both apps | Spins the stepper motor one way or the other until the next command |
| `center` | Vision app | Stops the motor (the target is centred) |
| `zero` | Vision app's **Set forward here** | The stepper's current direction becomes forward (0°) |
| `home` | Vision app's **↺ Return to forward** (resent until it's back) | Spins back to forward the way it came, then stops |
| `up`, `down`, `squint` | EEG app | Nothing yet: placeholders in `pi/actions.py` |
| `stop` | Both apps, whenever nothing else is happening | Stops the motor and switches its coils off |

The Pi also sends the laptop a status line 10 times a second, e.g.
`{"status": {"stepper_angle": 35.2, "stepper": "right"}}` (degrees from forward, + = right). It
counts half-steps (4096 per turn on a 28BYJ-48), so it's exact unless the motor skips steps.

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
