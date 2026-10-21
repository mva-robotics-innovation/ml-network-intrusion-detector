"""Tiny console logger with [+] / [!] / [i] prefixes."""

from __future__ import annotations

_ENABLED = True


def set_verbose(flag: bool) -> None:
    global _ENABLED
    _ENABLED = bool(flag)


def ok(msg: str) -> None:
    if _ENABLED:
        print(f"[+] {msg}", flush=True)


def warn(msg: str) -> None:
    if _ENABLED:
        print(f"[!] {msg}", flush=True)


def info(msg: str) -> None:
    if _ENABLED:
        print(f"[i] {msg}", flush=True)
