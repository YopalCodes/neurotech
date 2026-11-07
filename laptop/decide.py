"""Turns what the AI sees into a command for the Pi. Edit this file for your project.

decide() gets the YOLO result for one frame and returns a command, such as
{"cmd": "left"} or {"cmd": "move", "args": {"speed": 0.5}}. Every command name you
return needs a matching function in pi/actions.py.

It runs on every frame, and while AI control is on its answer is sent to the Pi every
frame, so return the command you want in effect right now. If the laptop stops
sending, the Pi stops.

The example below follows the largest object of one type and reports whether it is
on the left, right, or centre of the camera image.
"""

TARGET = "person"   # default target; the app's Target dropdown changes it live, and also
                    # lists the objects you taught in the Teach tab
DEAD_ZONE = 0.15    # how far from the middle (fraction of image width) still counts as centre

# Commands decide() can return. Each needs a function in pi/actions.py.
COMMANDS = ["left", "center", "right", "stop"]


def decide(result, target=TARGET):
    box = largest_box(result, target)
    if box is None:
        return {"cmd": "stop"}

    x1, _, x2, _ = box
    image_width = result.orig_shape[1]
    offset = (x1 + x2) / 2 / image_width - 0.5   # -0.5 = far left, +0.5 = far right

    if offset < -DEAD_ZONE:
        return {"cmd": "left"}
    if offset > DEAD_ZONE:
        return {"cmd": "right"}
    return {"cmd": "center"}


def largest_box(result, class_name):
    """Returns [x1, y1, x2, y2] of the biggest detection of class_name, or None."""
    best, best_area = None, 0.0
    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        if result.names[int(cls)] != class_name:
            continue
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best, best_area = box, area
    return best
