import threading
from typing import Callable


POPUP_WIDTH = 420
POPUP_MIN_HEIGHT = 180
POPUP_RIGHT_MARGIN = 24
POPUP_TOP_MARGIN = 36
FIRST_REPEAT_REMINDER_SECONDS = 3 * 60.0
ONGOING_REPEAT_REMINDER_SECONDS = 10 * 60.0


def popup_geometry(screen_width: int, requested_height: int) -> str:
    height = max(POPUP_MIN_HEIGHT, requested_height)
    x = max(16, screen_width - POPUP_WIDTH - POPUP_RIGHT_MARGIN)
    return f"{POPUP_WIDTH}x{height}+{x}+{POPUP_TOP_MARGIN}"


def show_health_popup(title: str, message: str) -> None:
    import tkinter as tk

    root = tk.Tk(className="DesktopHealthAssistantAlert")
    root.withdraw()
    root.title(title)
    root.configure(background="#171b1f")
    root.resizable(False, False)
    root.attributes("-topmost", True)
    root.overrideredirect(True)

    border = tk.Frame(root, background="#ef5b6c", padx=2, pady=2)
    border.pack(fill="both", expand=True)
    panel = tk.Frame(border, background="#20262b")
    panel.pack(fill="both", expand=True)

    tk.Label(
        panel,
        text=title,
        font=("Microsoft YaHei UI", 14, "bold"),
        foreground="#ffffff",
        background="#20262b",
        anchor="w",
    ).pack(fill="x", padx=18, pady=(15, 5))
    tk.Label(
        panel,
        text=message,
        font=("Microsoft YaHei UI", 10),
        foreground="#d9dee2",
        background="#20262b",
        justify="left",
        wraplength=380,
        anchor="w",
    ).pack(fill="x", padx=18)
    tk.Button(
        panel,
        text="知道了",
        command=root.destroy,
        font=("Microsoft YaHei UI", 9),
        foreground="#ffffff",
        background="#c84455",
        activeforeground="#ffffff",
        activebackground="#a93645",
        relief="flat",
        borderwidth=0,
        padx=16,
        pady=5,
        cursor="hand2",
    ).pack(anchor="e", padx=18, pady=(10, 12))

    root.update_idletasks()
    root.geometry(
        popup_geometry(
            root.winfo_screenwidth(),
            border.winfo_reqheight(),
        )
    )
    root.deiconify()
    root.after(15000, root.destroy)
    root.after(50, root.lift)
    root.mainloop()


class HealthPopupNotifier:
    def __init__(
        self,
        popup: Callable[[str, str], None] = show_health_popup,
    ) -> None:
        self.popup = popup

    def show(self, title: str, message: str) -> None:
        threading.Thread(
            target=self.popup,
            args=(title, message),
            daemon=True,
        ).start()
