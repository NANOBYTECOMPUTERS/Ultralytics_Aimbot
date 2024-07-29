from tkinter import Canvas
import tkinter as tk
import tkinter.font as tkFont
import threading
import queue
import numpy as np

from logic.config_watcher import cfg

class Overlay:
    def __init__(self):
        self.queue = queue.Queue()
        self.thread = None
        self.square_id = None

        # Skip frames so that the figures do not interfere with the detector ¯\_(ツ)_/¯
        self.frame_skip_counter = 0

    def run(self, width, height):
        if cfg.show_overlay:
            self.root = tk.Tk()

            self.root.overrideredirect(True)

            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()

            x = (screen_width - width) // 2
            y = (screen_height - height) // 2

            self.root.geometry(f"{width}x{height}+{x}+{y}")

            self.root.attributes('-topmost', True)
            self.root.attributes('-transparentcolor', 'white')

            self.canvas = Canvas(self.root, bg='white', highlightthickness=0, cursor="none")
            self.canvas.pack(fill=tk.BOTH, expand=True)

            # Bindings for the root window
            self.root.bind("<Button-1>", lambda e: "break")
            self.root.bind("<Button-2>", lambda e: "break")
            self.root.bind("<Button-3>", lambda e: "break")
            self.root.bind("<Motion>", lambda e: "break")
            self.root.bind("<Key>", lambda e: "break")
            self.root.bind("<Enter>", lambda e: "break")
            self.root.bind("<Leave>", lambda e: "break")
            self.root.bind("<FocusIn>", lambda e: "break")
            self.root.bind("<FocusOut>", lambda e: "break")

            # Bindings for the canvas
            self.canvas.bind("<Button-1>", lambda e: "break")
            self.canvas.bind("<Button-2>", lambda e: "break")
            self.canvas.bind("<Button-3>", lambda e: "break")
            self.canvas.bind("<Motion>", lambda e: "break")
            self.canvas.bind("<Key>", lambda e: "break")
            self.canvas.bind("<Enter>", lambda e: "break")
            self.canvas.bind("<Leave>", lambda e: "break")
            self.canvas.bind("<FocusIn>", lambda e: "break")
            self.canvas.bind("<FocusOut>", lambda e: "break")

            if cfg.overlay_show_borders:
                self.square_id = self.canvas.create_rectangle(0, 0, width, height, outline='black', width=2)
            
            # Create a dictionary to store shape IDs and tags
            self.shape_ids = {}
            # Create an image item on the canvas
            self.shape_ids["image"] = self.canvas.create_image(0, 0, anchor=tk.NW)

            self.process_queue()  # Start processing the queue in the main thread
            self.root.mainloop()

    def process_queue(self):
        if not self.queue.empty():
            while not self.queue.empty():
                data = self.queue.get()
                if isinstance(data, np.ndarray):  # Handle image updates
                    self.update_image(data)
                else:                               # Handle other drawing commands
                    command, args = data
                    command(*args)

        self.root.after(2, self.process_queue)  # Schedule next update

    def update_image(self, image_np):
        # Convert the NumPy array to a PhotoImage
        photo = tk.PhotoImage(data=image_np.tobytes(), width=image_np.shape[1], height=image_np.shape[0])
        # Update the image on the canvas
        self.canvas.itemconfig(self.shape_ids["image"], image=photo)
        self.canvas.image = photo  # Keep a reference to avoid garbage collection

    def _draw_square(self, x1, y1, x2, y2, color='black', size=1):
            # Remove the fill attribute to draw only the outline
            if "square" not in self.shape_ids:
                self.shape_ids["square"] = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=size, tags="square")
            else:
                self.canvas.coords(self.shape_ids["square"], x1, y1, x2, y2)
                self.canvas.itemconfig(self.shape_ids["square"], outline=color, width=size)

    def _draw_oval(self, x1, y1, x2, y2, color='black', size=1):
        if "oval" not in self.shape_ids:
            self.shape_ids["oval"] = self.canvas.create_oval(x1, y1, x2, y2, outline=color, width=size, tags="oval")
        else:
            self.canvas.coords(self.shape_ids["oval"], x1, y1, x2, y2)
            self.canvas.itemconfig(self.shape_ids["oval"], outline=color, width=size)

    def _draw_line(self, x1, y1, x2, y2, color='black', size=1):
        if "line" not in self.shape_ids:
            self.shape_ids["line"] = self.canvas.create_line(x1, y1, x2, y2, fill=color, width=size, tags="line")
        else:
            self.canvas.coords(self.shape_ids["line"], x1, y1, x2, y2)
            self.canvas.itemconfig(self.shape_ids["line"], fill=color, width=size)

    def _draw_point(self, x, y, color='black', size=1):
        if "point" not in self.shape_ids:
            self.shape_ids["point"] = self.canvas.create_oval(x-size, y-size, x+size, y+size, fill=color, outline=color, tags="point")
        else:
            self.canvas.coords(self.shape_ids["point"], x-size, y-size, x+size, y+size)
            self.canvas.itemconfig(self.shape_ids["point"], fill=color, outline=color)

    def _draw_text(self, x, y, text, size=12, color='black'):
        if "text" not in self.shape_ids:
            self.shape_ids["text"] = self.canvas.create_text(x, y, text=text, font=('Arial', size), fill=color, state='', tags="text")
        else:
            self.canvas.coords(self.shape_ids["text"], x, y)
            self.canvas.itemconfig(self.shape_ids["text"], text=text, font=('Arial', size), fill=color)

    # Remaining methods:

    def draw_square(self, x1, y1, x2, y2, color='black', size=1):
        self.queue.put((self._draw_square, (x1, y1, x2, y2, color, size)))

    def draw_oval(self, x1, y1, x2, y2, color='black', size=1):
        self.queue.put((self._draw_oval, (x1, y1, x2, y2, color, size)))

    def draw_line(self, x1, y1, x2, y2, color='black', size=1):
        self.queue.put((self._draw_line, (x1, y1, x2, y2, color, size)))

    def draw_point(self, x, y, color='black', size=1):
        self.queue.put((self._draw_point, (x, y, color, size)))

    def draw_text(self, x, y, text, size=12, color='black'):
        self.queue.put((self._draw_text, (x, y, text, size, color)))

    def show(self, width, height):
        if self.thread is None:
            self.thread = threading.Thread(target=self.run, args=(width, height), daemon=True)
            self.thread.start()

overlay = Overlay()