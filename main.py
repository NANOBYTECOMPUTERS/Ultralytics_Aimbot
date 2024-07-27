import tkinter as tk
import cv2
from PIL import Image, ImageTk
import bettercam
import threading


class CaptureViewer:
    def __init__(self, master):
        self.master = master
        master.title("BetterCam Viewer")

        # Capture Settings (with adjustable width and height)
        self.width_label = tk.Label(master, text="Capture Width:")
        self.width_label.grid(row=0, column=0)
        self.width_var = tk.StringVar(master, value="350")  # Default width
        self.width_entry = tk.Entry(master, textvariable=self.width_var)
        self.width_entry.grid(row=0, column=1)

        self.height_label = tk.Label(master, text="Capture Height:")
        self.height_label.grid(row=1, column=0)
        self.height_var = tk.StringVar(master, value="400")  # Default height
        self.height_entry = tk.Entry(master, textvariable=self.height_var)
        self.height_entry.grid(row=1, column=1)

        # Video Feed
        self.video_label = tk.Label(master)
        self.video_label.grid(row=2, column=0, columnspan=2)

        # Buttons
        self.start_button = tk.Button(master, text="Start", command=self.start_capture)
        self.start_button.grid(row=3, column=0)
        self.stop_button = tk.Button(master, text="Stop", command=self.stop_capture, state=tk.DISABLED)
        self.stop_button.grid(row=3, column=1)

        self.running = False
        self.capture = None

    def start_capture(self):
        if not self.running:
            self.running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.capture_thread = threading.Thread(target=self.update_video_loop)
            self.capture_thread.start()

    def stop_capture(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
            self.capture.Quit()
            self.capture = None

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def update_video_loop(self):
        # Capture settings (from your GUI)
        capture_width = int(self.width_var.get())
        capture_height = int(self.height_var.get())
        
        # Initialize Capture object
        self.capture = bettercam.create(device_idx=0, output_idx=0 output_color="RGB", max_buffer_len=64)
  
        # Set capture settings
        self.capture.output_color("RGB")
        self.capture.set_target_fps(30)
        self.capture.set_region([0, 0, capture_width, capture_height])
        try:
            while self.running:
                frame = self.capture.grab()
                if frame is not None:
                    self.update_video_frame(frame)
        except Exception as e:
            print(f"Error during capture: {e}")

    def update_video_frame(self, frame):
        # Convert OpenCV frame to PIL Image
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)

        # Resize image to fit the label
        img = img.resize((int(self.width_entry.get()), int(self.height_entry.get())))
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the label with the new image
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

root = tk.Tk()
app = CaptureViewer(root)
root.mainloop()