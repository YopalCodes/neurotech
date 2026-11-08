# Remote device control

Control other computers and Android phones from the main computer. You get a live
view of the device's screen, and your mouse and keyboard drive it.

```
pip install -r requirements.txt
python controller.py      # on the main computer
python agent.py --password <pw>   # on each computer you want to control
```

The device selector is meant to be camera-based later. It plugs in through
`DeviceManager.select(name)` in [devices.py](devices.py), which every device
supports with the same calls: `pointer_down/up`, `pointer_move`, `scroll`,
`key_down/up`, `type_text` and `open_app`. All coordinates are normalized to 0–1.

## Computers (Windows, macOS, Linux)

1. Copy this folder to the computer and install the requirements.
2. Run `python agent.py --password <at least 8 characters>`. It prints the
   addresses to use. On Windows, allow it through the firewall when prompted.
3. In the controller, click **Add computer...**, enter that address, then
   **Connect** and type the password.

- **Mouse:** moves and clicks go straight through.
- **Keyboard:** keys go straight through, including Ctrl/Shift/Alt combos.
- **Open app...:** types the name into the Start menu (Spotlight on macOS).

The agent console says when someone is connected. Ctrl+C there stops sharing. Each
agent accepts one controller at a time, and any keys or buttons still held are
released when the controller disconnects.

Options: `--monitor 2` shares another screen, `--fps`, `--quality` and `--max-width`
trade sharpness for bandwidth, and `--bind 127.0.0.1` makes it local-only.

**Security:** the link is encrypted with AES-256-GCM, using a key derived from the
password. Without the password, a connection is refused and nothing it sends is
executed. Anyone who has the password can fully control the computer, so use
a strong one and only run the agent while you need it.

**Limitations:**
- Shortcuts your own OS intercepts (Alt+Tab, the Windows key, Ctrl+Alt+Del) act
  on the main computer, not the remote one.
- On a Windows remote, the agent can't control admin windows or the UAC
  prompt unless the agent is run as administrator.

## Android phones and tablets

No app is installed on the phone. The controller uses ADB (Android platform-tools must
be installed on the main computer). While you're connected, it runs
[scrcpy](https://github.com/Genymobile/scrcpy)'s small server from the phone's temp folder.
That gives a smooth ~25–30 fps live view and instant touch and typing. The server
(v3.1, Apache-2.0) downloads into `vendor/` on first use and is checked against
scrcpy's published SHA-256. If it can't run, the controller falls back to
`screenrecord` or screenshots plus `adb input`, which is slower.

1. On the phone, open Settings → About phone and tap **Build number** 7 times
   to unlock Developer options.
2. In Developer options, turn on **USB debugging**.
3. Plug the phone in, accept **Allow USB debugging**, and click
   **Refresh phones** in the controller.

| On the screen | Phone |
|---|---|
| click | tap |
| drag | swipe (follows the mouse live) |
| hold | long press |
| right-click / middle-click | Back / Home |
| wheel | scroll |
| typing | text input, plus Enter, Backspace, arrows and Ctrl+letter |

The toolbar has Back, Home, Recents, Power and Volume buttons. **Open app...**
lists the phone's apps.

### Wireless

There are two ways to connect. Both need the phone and the main computer on the same
network, and some networks block devices from reaching each other. Over Wi-Fi the
stream drops to 1024 px at 2 Mbit/s, which kept about 24 fps on eduroam.

- **USB phone → Wi-Fi (quickest):** with the phone plugged in, select it and click
  **USB phone → Wi-Fi**, then unplug it. A second entry, `(ip:5555)`, appears;
  connect to that one. This lasts until the phone reboots. Only computers the phone
  has already authorized can connect. To turn it off early, plug in and run
  `adb usb`, or reboot the phone.
- **Pair phone (Wi-Fi)... (no cable, Android 11+):** this uses Android's own
  encrypted wireless debugging.
  1. Open Developer options → **Wireless debugging** → **Pair device with pairing code**.
  2. Enter the pairing address and code, then the address shown on the main
     Wireless debugging screen.
