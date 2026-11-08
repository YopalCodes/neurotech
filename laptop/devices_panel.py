"""The Devices tab: control other computers and Android phones from the desktop app.

Devices come from remote/devices.py: computers running remote/agent.py, and Android phones
over ADB (USB or Wi-Fi). Connecting opens a window with the device's live screen, driven by
this computer's mouse and keyboard (device_window.py). One device is controlled at a time.
"""

import logging
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QVBoxLayout, QWidget,
)

from device_window import DeviceWindow

sys.path.append(str(Path(__file__).resolve().parent.parent / "remote"))
from devices import AndroidDevice, ComputerDevice, DeviceManager  # noqa: E402  (remote/)

log = logging.getLogger("devices")
GREEN, RED, GREY = "#2e9e4f", "#d64545", "#888888"


def short(name):
    """"DE2118 (3f93e4ec)" -> "DE2118"."""
    return name.split(" (")[0]


def dot(color, text):
    return f'<span style="color:{color}">●</span> {text}'


class DevicesPanel(QWidget):
    job_done = Signal(str, str)   # (device name or "", error or message) from a background thread

    def __init__(self):
        super().__init__()
        self.manager = DeviceManager()
        self.windows = {}
        self.passwords = {}       # agent passwords, remembered for this session only
        self.connecting = None
        self.job_done.connect(self._job_done)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda item: self.connect_device(item.data(Qt.UserRole)))
        layout.addWidget(self.list, 1)

        buttons = QHBoxLayout()
        connect = QPushButton("Connect && open screen")
        connect.clicked.connect(lambda: self.connect_device(self._selected()))
        disconnect = QPushButton("Disconnect")
        disconnect.clicked.connect(self.disconnect)
        buttons.addWidget(connect)
        buttons.addWidget(disconnect)
        layout.addLayout(buttons)

        for row in ((("Add PC...", self.add_computer), ("Remove", self.remove)),
                    (("Find phones", self.refresh_phones), ("Phone → Wi-Fi", self.usb_to_wifi),
                     ("Pair phone...", self.pair_phone))):
            more = QHBoxLayout()
            for text, slot in row:
                b = QPushButton(text)
                b.clicked.connect(slot)
                more.addWidget(b)
            layout.addLayout(more)

        hint = QLabel("Computers: run <b>python remote/agent.py --password &lt;pw&gt;</b> on them, then "
                      "Add PC with their IP address. Phones: turn on USB debugging and plug in. "
                      "Double-click a device to control it.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {GREY};")
        layout.addWidget(hint)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.refresh_list()
        self.refresh_phones(quiet=True)

    # ---- device list ----

    def _selected(self):
        item = self.list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _connected(self, name):
        d = self.manager.current
        return d is not None and d.name == name and d.connected

    def refresh_list(self):
        current = self._selected()
        self.list.clear()
        for name, d in self.manager.devices.items():
            kind = "Phone" if isinstance(d, AndroidDevice) else "PC"
            mark = "  ● connected" if self._connected(name) else ("  … connecting" if name == self.connecting else "")
            item = QListWidgetItem(f"[{kind}] {name}{mark}")
            item.setData(Qt.UserRole, name)
            self.list.addItem(item)
            if name == current:
                self.list.setCurrentItem(item)
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)

    def _run(self, work):
        threading.Thread(target=work, daemon=True).start()

    def refresh_phones(self, quiet=False):
        def work():
            try:
                found = self.manager.refresh_android()
                msg = "" if quiet else (f"Found {len(found)} phone(s)." if found else
                                        "No phones found. Turn on USB debugging, plug the phone in, "
                                        "and allow this computer.")
            except Exception as e:  # noqa: BLE001
                msg = "" if quiet else f"Couldn't look for phones: {e}"
            self.job_done.emit("", msg)
        self._run(work)

    def usb_to_wifi(self):
        name = self._selected()
        d = self.manager.devices.get(name)
        if not isinstance(d, AndroidDevice) or ":" in d.serial:
            self.status.setText("Select a phone connected by USB first.")
            return
        self.status.setText(f"Switching {short(name)} to Wi-Fi...")

        def work():
            try:
                addr = self.manager.android_to_wifi(name)
                msg = f"{short(name)} is on Wi-Fi at {addr}; you can unplug it and use its ({addr}) entry."
            except Exception as e:  # noqa: BLE001
                msg = f"Couldn't switch to Wi-Fi: {e}"
            self.job_done.emit("", msg)
        self._run(work)

    def pair_phone(self):
        info = "Phone: Developer options → Wireless debugging → Pair device with pairing code.\n\n"
        pair, ok = QInputDialog.getText(self, "Pair phone", info + "IP address & port shown under the code:")
        if not ok or not pair:
            return
        code, ok = QInputDialog.getText(self, "Pair phone", "6-digit pairing code:")
        if not ok or not code:
            return
        addr, ok = QInputDialog.getText(self, "Pair phone", "IP address & port on the main Wireless debugging screen:")
        if not ok or not addr:
            return
        self.status.setText("Pairing...")

        def work():
            try:
                msg = self.manager.pair_android_wifi(pair.strip(), code.strip(), addr.strip())
                self.manager.refresh_android()
            except Exception as e:  # noqa: BLE001
                msg = f"Pairing failed: {e}"
            self.job_done.emit("", msg)
        self._run(work)

    def add_computer(self):
        host, ok = QInputDialog.getText(self, "Add PC", "IP address or hostname of the computer running remote/agent.py:")
        if not ok or not host.strip():
            return
        name, ok = QInputDialog.getText(self, "Add PC", "Name:", text=host.strip())
        self.manager.add_computer((name or host).strip(), host.strip())
        self.refresh_list()

    def remove(self):
        name = self._selected()
        if name:
            self.manager.remove(name)
            self.refresh_list()

    # ---- connecting ----

    def connect_device(self, name):
        if not name or self.connecting:
            return
        password = None
        if isinstance(self.manager.devices[name], ComputerDevice):
            password = self.passwords.get(name)
            if password is None:
                password, ok = QInputDialog.getText(self, "Password", f"Agent password for {name}:",
                                                    QLineEdit.Password)
                if not ok:
                    return
        self.connecting = name
        self.status.setText(f"Connecting to {short(name)}...")
        self.refresh_list()

        def work():
            try:
                self.manager.select(name, password)
                if password:
                    self.passwords[name] = password
                self.job_done.emit(name, "")
            except Exception as e:  # noqa: BLE001
                self.passwords.pop(name, None)
                self.job_done.emit(name, str(e) or type(e).__name__)
        self._run(work)

    def _job_done(self, name, error):
        if not name:            # a background job that only reports a message
            if error:
                self.status.setText(error)
            self.refresh_list()
            return
        self.connecting = None
        if error:
            self.status.setText(dot(RED, f"Couldn't connect to {short(name)}: {error}"))
            log.error("Couldn't connect to %s: %s", name, error)
        else:
            log.info("Connected to %s", name)
            self.status.setText(dot(GREEN, f"Controlling {short(name)}."))
            for other in list(self.windows.values()):   # one device at a time
                other.close()
            window = DeviceWindow(self.manager.current, self.manager)
            window.closed.connect(self._window_closed)
            self.windows[name] = window
            window.show()
            window.raise_()
        self.refresh_list()

    def _window_closed(self, name):
        self.windows.pop(name, None)
        if self.manager.current is not None and self.manager.current.name == name:
            self.manager.deselect()
            log.info("Disconnected from %s", name)
        self.refresh_list()

    def disconnect(self):
        for window in list(self.windows.values()):
            window.close()
        self.manager.deselect()
        self.status.setText("Disconnected.")
        self.refresh_list()

    def shutdown(self):
        for window in list(self.windows.values()):
            window.close()
        self.manager.deselect()
