"""Control other computers and Android phones from this one.

    python controller.py

Left: your devices. Phones plugged in with USB debugging appear after
"Refresh phones"; add computers running agent.py with "Add computer".
Select a device to see its screen live. Click the screen to give it focus,
then your mouse and keyboard drive the device.

Phones: click = tap, drag = swipe, hold = long-press, right-click = Back,
middle-click = Home, wheel = scroll.
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from PIL import Image, ImageTk

import protocol
from devices import AndroidDevice, ComputerDevice, DeviceManager

# Tk keysym -> key name understood by agent.py / AndroidDevice
KEYSYMS = {
    "Return": "enter", "KP_Enter": "enter", "BackSpace": "backspace", "Tab": "tab", "Escape": "esc",
    "Delete": "delete", "space": "space", "Up": "up", "Down": "down", "Left": "left", "Right": "right",
    "Home": "home", "End": "end", "Prior": "page_up", "Next": "page_down", "Insert": "insert",
    "Shift_L": "shift", "Shift_R": "shift_r", "Control_L": "ctrl", "Control_R": "ctrl_r",
    "Alt_L": "alt", "Alt_R": "alt_r", "Win_L": "cmd", "Win_R": "cmd_r", "Super_L": "cmd", "Super_R": "cmd_r",
    "Caps_Lock": "caps_lock", **{f"F{i}": f"f{i}" for i in range(1, 13)},
}
BUTTON_NAMES = {1: "left", 2: "middle", 3: "right"}
MOVE_INTERVAL = 0.015  # s; cap pointer-move messages at ~60 per second


def key_name(event):
    if event.keysym in KEYSYMS:
        return KEYSYMS[event.keysym]
    if len(event.keysym) == 1:
        return event.keysym
    if len(event.char) == 1 and event.char.isprintable():
        return event.char  # punctuation like "comma" -> ","
    return None


class ControllerApp:
    def __init__(self, root):
        self.root = root
        self.manager = DeviceManager()
        self.passwords = {}  # remembered for this session only
        self.photo = None
        self.shown_frame = -1
        self.view = None     # (x offset, y offset, scale) of the image on the canvas
        self.last_move = 0.0

        root.title("Neurotech device control")
        root.geometry("1280x800")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        side = ttk.Frame(root, padding=8)
        side.grid(row=0, column=0, sticky="ns")
        ttk.Label(side, text="Devices", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.listbox = tk.Listbox(side, width=34, height=20, exportselection=False)
        self.listbox.pack(fill="y", expand=True, pady=4)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.select_highlighted())
        for text, cmd in (("Connect", self.select_highlighted), ("Disconnect", self.disconnect),
                          ("Refresh phones", self.refresh_phones), ("Phone over Wi-Fi...", self.add_wifi_phone),
                          ("Add computer...", self.add_computer), ("Remove", self.remove)):
            ttk.Button(side, text=text, command=cmd).pack(fill="x", pady=1)

        main = ttk.Frame(root)
        main.grid(row=0, column=1, sticky="nsew")
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)
        self.toolbar = ttk.Frame(main, padding=(8, 8, 8, 0))
        self.toolbar.grid(row=0, column=0, sticky="ew")
        self.canvas = tk.Canvas(main, bg="#111", highlightthickness=2, highlightbackground="#111",
                                highlightcolor="#4a9eff", takefocus=1)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.status = ttk.Label(main, text="Select a device.", padding=(8, 0, 8, 8))
        self.status.grid(row=2, column=0, sticky="ew")

        c = self.canvas
        c.bind("<Motion>", self.on_move)
        c.bind("<B1-Motion>", self.on_move)
        c.bind("<B3-Motion>", self.on_move)
        for n in (1, 2, 3):
            c.bind(f"<ButtonPress-{n}>", lambda e, n=n: self.on_button(e, n, True))
            c.bind(f"<ButtonRelease-{n}>", lambda e, n=n: self.on_button(e, n, False))
        c.bind("<MouseWheel>", self.on_wheel)                       # Windows / macOS
        c.bind("<Button-4>", lambda e: self.on_wheel(e, 120))       # Linux
        c.bind("<Button-5>", lambda e: self.on_wheel(e, -120))
        c.bind("<KeyPress>", self.on_key_down)
        c.bind("<KeyRelease>", self.on_key_up)

        self.refresh_list()
        self.refresh_phones(quiet=True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tick()

    # ---- device list -------------------------------------------------------

    def refresh_list(self):
        self.listbox.delete(0, "end")
        for name, d in self.manager.devices.items():
            kind = "Phone" if isinstance(d, AndroidDevice) else "PC"
            mark = "  ● connected" if d is self.manager.current and d.connected else ""
            self.listbox.insert("end", f"[{kind}] {name}{mark}")

    def highlighted(self):
        sel = self.listbox.curselection()
        return list(self.manager.devices)[sel[0]] if sel else None

    def refresh_phones(self, quiet=False):
        try:
            found = self.manager.refresh_android()
        except Exception as e:  # noqa: BLE001
            if not quiet:
                messagebox.showerror("Phones", str(e))
            return
        self.refresh_list()
        if not quiet and not found:
            messagebox.showinfo("Phones", "No phones found. Plug one in with USB debugging on and accept "
                                          "the 'Allow USB debugging' prompt on the phone.")

    def add_wifi_phone(self):
        addr = simpledialog.askstring("Phone over Wi-Fi",
                                      "Phone address (ip:port) from Wireless debugging.\n"
                                      "Pair it first once with:  adb pair ip:pairing_port", parent=self.root)
        if addr:
            self.status.config(text=self.manager.connect_android_wifi(addr.strip()))
            self.refresh_phones(quiet=True)

    def add_computer(self):
        host = simpledialog.askstring("Add computer", "IP address or hostname of the computer running agent.py:",
                                      parent=self.root)
        if not host:
            return
        name = simpledialog.askstring("Add computer", "Name for this computer:", initialvalue=host,
                                      parent=self.root) or host
        port = simpledialog.askinteger("Add computer", "Port:", initialvalue=protocol.DEFAULT_PORT,
                                       parent=self.root) or protocol.DEFAULT_PORT
        self.manager.add_computer(name, host.strip(), port)
        self.refresh_list()

    def remove(self):
        name = self.highlighted()
        if name:
            self.manager.remove(name)
            self.refresh_list()
            self.update_toolbar()

    # ---- selecting ---------------------------------------------------------

    def select_highlighted(self):
        name = self.highlighted()
        if name:
            self.select(name)

    def select(self, name):
        """Connect to a device by name. The camera-based selector can call this later."""
        device = self.manager.devices[name]
        password = None
        if isinstance(device, ComputerDevice):
            password = self.passwords.get(name) or simpledialog.askstring(
                "Password", f"Agent password for {name}:", show="•", parent=self.root)
            if password is None:
                return
        self.status.config(text=f"Connecting to {name}...")

        def work():
            try:
                self.manager.select(name, password)
                if password:
                    self.passwords[name] = password
                self.root.after(0, lambda: self.on_selected(name))
            except Exception as e:  # noqa: BLE001
                self.passwords.pop(name, None)
                self.root.after(0, lambda: self.status.config(text=f"Couldn't connect to {name}: {e}"))
        threading.Thread(target=work, daemon=True).start()

    def on_selected(self, name):
        self.shown_frame = -1
        self.refresh_list()
        self.update_toolbar()
        self.canvas.focus_set()
        self.status.config(text=f"Controlling {name}. Click the screen to give it keyboard focus.")

    def disconnect(self):
        self.manager.deselect()
        self.canvas.delete("all")
        self.photo = None
        self.view = None
        self.refresh_list()
        self.update_toolbar()
        self.status.config(text="Disconnected.")

    def update_toolbar(self):
        for w in self.toolbar.winfo_children():
            w.destroy()
        d = self.manager.current
        if d is None:
            return
        ttk.Label(self.toolbar, text=d.name, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(0, 12))
        for name in d.actions:
            ttk.Button(self.toolbar, text=name.replace("_", " ").title(),
                       command=lambda n=name: d.action(n)).pack(side="left", padx=1)
        ttk.Button(self.toolbar, text="Open app...", command=self.open_app).pack(side="left", padx=(12, 1))
        ttk.Button(self.toolbar, text="Type text...", command=self.type_text).pack(side="left", padx=1)

    def open_app(self):
        d = self.manager.current
        if isinstance(d, AndroidDevice):
            AppPicker(self.root, d)
        elif d:
            app = simpledialog.askstring("Open app", "App name (as you'd type it in the Start menu):", parent=self.root)
            if app:
                d.open_app(app)

    def type_text(self):
        d = self.manager.current
        text = d and simpledialog.askstring("Type text", "Text to type on the device:", parent=self.root)
        if text:
            d.type_text(text)

    # ---- screen ------------------------------------------------------------

    def tick(self):
        d = self.manager.current
        if d is not None:
            if d.frame is not None and d.frame_id != self.shown_frame:
                self.draw(d.frame)
                self.shown_frame = d.frame_id
            if not d.connected and d.error:
                self.status.config(text=f"{d.name}: {d.error}")
                self.manager.current = None
                self.refresh_list()
        self.root.after(15, self.tick)

    def draw(self, img):
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        scale = min(cw / img.width, ch / img.height)
        w, h = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        ox, oy = (cw - w) // 2, (ch - h) // 2
        self.photo = ImageTk.PhotoImage(img.resize((w, h), Image.BILINEAR))
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        self.view = (ox, oy, w, h)

    def to_device(self, event):
        """Canvas pixel -> normalized device coordinate, or None if outside the image."""
        if self.view is None:
            return None
        ox, oy, w, h = self.view
        x, y = (event.x - ox) / w, (event.y - oy) / h
        if 0 <= x <= 1 and 0 <= y <= 1:
            return x, y
        return None

    # ---- input -------------------------------------------------------------

    def on_move(self, event):
        d, pos = self.manager.current, self.to_device(event)
        now = time.monotonic()
        if d and pos and now - self.last_move >= MOVE_INTERVAL:
            d.pointer_move(*pos)
            self.last_move = now

    def on_button(self, event, number, pressed):
        self.canvas.focus_set()
        d, pos = self.manager.current, self.to_device(event)
        if not (d and pos):
            return
        if pressed:
            d.pointer_down(*pos, BUTTON_NAMES[number])
        else:
            d.pointer_up(*pos, BUTTON_NAMES[number])

    def on_wheel(self, event, delta=None):
        d, pos = self.manager.current, self.to_device(event)
        if d and pos:
            d.scroll(1 if (delta or event.delta) > 0 else -1, *pos)

    def on_key_down(self, event):
        d, key = self.manager.current, key_name(event)
        if d and key:
            d.key_down(key)
        return "break"  # keep Tab etc. from moving focus out of the screen

    def on_key_up(self, event):
        d, key = self.manager.current, key_name(event)
        if d and key:
            d.key_up(key)
        return "break"

    def on_close(self):
        self.manager.deselect()
        self.root.destroy()


class AppPicker(tk.Toplevel):
    """Searchable list of a phone's launchable apps."""

    def __init__(self, parent, device):
        super().__init__(parent)
        self.title("Open app")
        self.geometry("380x480")
        self.device = device
        self.apps = []
        self.query = tk.StringVar()
        self.query.trace_add("write", lambda *_: self.filter())
        entry = ttk.Entry(self, textvariable=self.query)
        entry.pack(fill="x", padx=8, pady=8)
        entry.focus_set()
        self.listbox = tk.Listbox(self)
        self.listbox.pack(fill="both", expand=True, padx=8)
        self.listbox.bind("<Double-Button-1>", lambda _e: self.launch())
        entry.bind("<Return>", lambda _e: self.launch())
        ttk.Button(self, text="Open", command=self.launch).pack(pady=8)
        self.listbox.insert("end", "Loading...")
        threading.Thread(target=self.load, daemon=True).start()

    def load(self):
        apps = self.device.list_apps()
        self.after(0, lambda: (setattr(self, "apps", apps), self.filter()))

    def filter(self):
        q = self.query.get().lower()
        self.listbox.delete(0, "end")
        for app in self.apps:
            if q in app.lower():
                self.listbox.insert("end", app)

    def launch(self):
        sel = self.listbox.curselection()
        app = self.listbox.get(sel[0]) if sel else (self.listbox.get(0) if self.listbox.size() else None)
        if app and app != "Loading...":
            self.device.open_app(app)
            self.destroy()


def main():
    root = tk.Tk()
    ControllerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
