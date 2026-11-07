"""Runs on the Pi. Receives commands from the laptop over Wi-Fi and carries them out.

Protocol: the laptop connects over TCP (port 5005 by default) and sends one JSON
object per line, for example
    {"cmd": "left"}
    {"cmd": "move", "args": {"speed": 0.4}}

The laptop resends the command it wants on every frame it processes. The matching
function in actions.py runs only when the command changes, so repeats cost nothing.

Failsafe: if no valid command arrives for --failsafe seconds (laptop crashed, Wi-Fi
dropped, AI stalled), or the laptop disconnects, actions.stop() runs.

    python3 command_server.py
"""

import argparse
import json
import logging
import select
import socket
import time

import actions

log = logging.getLogger("command_server")

STOP = ("stop", "{}")
MAX_LINE = 64 * 1024


class CommandRunner:
    """Runs each new command once, and stops the hardware when commands stop arriving."""

    def __init__(self, failsafe_seconds):
        self.failsafe_seconds = failsafe_seconds
        self.last_valid = time.monotonic()
        self._warned = set()
        actions.stop()  # start in a safe state
        self.active = STOP

    def handle_line(self, line):
        try:
            msg = json.loads(line)
        except ValueError:
            self._warn_once(f"Ignoring a message that isn't JSON: {line[:100]!r}")
            return
        if not isinstance(msg, dict) or not isinstance(msg.get("cmd"), str):
            self._warn_once(f'Ignoring a message without a "cmd" string: {msg!r}')
            return
        cmd = msg["cmd"]
        args = msg.get("args") or {}
        if not isinstance(args, dict):
            self._warn_once(f'Ignoring {cmd!r}: "args" must be an object, got {args!r}')
            return
        handler = actions.HANDLERS.get(cmd)
        if handler is None:
            self._warn_once(f"Ignoring unknown command {cmd!r} (add it to HANDLERS in actions.py)")
            return

        self.last_valid = time.monotonic()
        key = (cmd, json.dumps(args, sort_keys=True))
        if key == self.active:
            return
        log.info("Command: %s %s", cmd, args if args else "")
        self.active = key  # set first, so a failing command isn't retried on every frame
        try:
            handler(**args)
        except Exception:
            log.exception("Command %r failed", cmd)

    def check_failsafe(self):
        silent = time.monotonic() - self.last_valid
        if silent > self.failsafe_seconds:
            self.stop(f"no command for {silent:.1f} s")

    def stop(self, reason):
        if self.active == STOP:
            return
        log.warning("Stopping: %s", reason)
        self.active = STOP
        try:
            actions.stop()
        except Exception:
            log.exception("actions.stop() failed")

    def _warn_once(self, message):
        if message not in self._warned:
            self._warned.add(message)
            log.warning(message)


def serve(host, port, runner):
    """Accepts one laptop at a time. A new connection replaces the old one, so a laptop
    that reconnects after a Wi-Fi drop is never locked out by its own dead connection."""
    server = socket.create_server((host, port))
    log.info("Listening for the laptop on port %d", port)
    client, buffer = None, b""
    try:
        while True:
            watch = [server] if client is None else [server, client]
            readable, _, _ = select.select(watch, [], [], 0.05)

            if server in readable:
                new_client, addr = server.accept()
                if client is not None:
                    log.info("New connection from %s replaces the old one", addr[0])
                    client.close()
                else:
                    log.info("Laptop connected from %s", addr[0])
                client, buffer = new_client, b""

            if client is not None and client in readable:
                try:
                    data = client.recv(4096)
                except OSError:
                    data = b""
                if not data:
                    log.info("Laptop disconnected")
                    client.close()
                    client = None
                    runner.stop("laptop disconnected")
                else:
                    buffer += data
                    *lines, buffer = buffer.split(b"\n")
                    for line in lines:
                        if line.strip():
                            runner.handle_line(line)
                    if len(buffer) > MAX_LINE:
                        log.warning("Dropping an over-long message")
                        buffer = b""

            runner.check_failsafe()
    finally:
        if client is not None:
            client.close()
        server.close()


def main():
    parser = argparse.ArgumentParser(description="Receive commands from the laptop and run them.")
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on (default: all)")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--failsafe", type=float, default=0.5,
                        help="seconds without a command before stopping (default: 0.5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    runner = CommandRunner(args.failsafe)
    try:
        serve(args.host, args.port, runner)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop("shutting down")


if __name__ == "__main__":
    main()
