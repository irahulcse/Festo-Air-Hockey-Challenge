#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox


def resource_path(filename: str) -> Path:
    """Return absolute path for resources next to this script."""
    return Path(__file__).resolve().parent / filename


def run_main() -> None:
    """Launch `main.py` in a new Python process using the same interpreter."""
    script = resource_path("main.py")
    if not script.exists():
        messagebox.showerror("File not found", f"main.py not found at {script}")
        return
    subprocess.Popen([sys.executable, str(script)]) 


root = tk.Tk()
root.title("Festo Air‑Hockey Launcher")
root.minsize(520, 320)  
root.configure(background="#1e1e1e")

screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
win_w, win_h = 520, 320
root.geometry(f"{win_w}x{win_h}+{(screen_w - win_w) // 2}+{(screen_h - win_h) // 3}")
root.resizable(False, False)

style = ttk.Style(root)
style.theme_use("clam") 
ACCENT = "#007acc"     
BG_DARK = "#1e1e1e"
FG_LIGHT = "#f0f0f0"

style.configure("TFrame", background=BG_DARK)
style.configure("Header.TLabel", font=("Helvetica", 24, "bold"), foreground=FG_LIGHT, background=BG_DARK)
style.configure(
    "Accent.TButton",
    font=("Helvetica", 16, "bold"),
    foreground=FG_LIGHT,
    background=ACCENT,
    borderwidth=0,
    focusthickness=3,
    focuscolor="white",
    padding=(20, 10),
)
style.map(
    "Accent.TButton",
    background=[("active", "#005a9e"), ("disabled", "#5a5a5a")],
)
style.configure(
    "Secondary.TButton",
    font=("Helvetica", 12),
    foreground=FG_LIGHT,
    background="#333333",
    padding=(12, 6),
)
style.map(
    "Secondary.TButton",
    background=[("active", "#444444")],
)

container = ttk.Frame(root)
container.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

header = ttk.Label(container, text="Festo Air‑Hockey Challenge", style="Header.TLabel")
header.pack(pady=(0, 30))

start_btn = ttk.Button(container, text="Start Game", style="Accent.TButton", command=run_main)
start_btn.pack(fill="x")

quit_btn = ttk.Button(container, text="Quit", style="Secondary.TButton", command=root.destroy)
quit_btn.pack(pady=(20, 0))

start_btn.focus_set()
root.bind("<Return>", lambda *_: start_btn.invoke())

root.mainloop()
