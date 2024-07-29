import tkinter as tk
import wmi
import ctypes.util
import ctypes.wintypes
import ctypes
import threading
from ctypes import windll, Structure, POINTER, byref, c_ushort, c_uint, c_ulong, wstring_at, sizeof, CFUNCTYPE, c_int 
from ctypes.wintypes import DWORD, HANDLE, UINT, BOOL, WPARAM, LPARAM, POINT, MSG
import win32api
import win32.lib.win32con as win32con
from win32con import RELATIVE, MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE
import sys
import queue

class MouseThread(threading.Thread):
    def __init__(self):
        super(MouseThread, self).__init__()
        self.mouse = win32api
        self.mouse_x = 0
        self.mouse_y = 0

    def run(self):
        while True:
            if not mouse_queue.empty():
                mouse_x, mouse_y, mouse_event_type = mouse_queue.get()
                if mouse_event_type == win32con.WM_MOUSEMOVE:  # Check for mouse movement events
                    self.process_mouse_input(mouse_x, mouse_y, True)  # True indicates raw input
                else:
                    self.process_mouse_input(mouse_x, mouse_y, False) 

    def process_mouse_input(self, x, y, is_raw_input):
        if is_raw_input:
            self.mouse_x = x
            self.mouse_y = y
        else:
            self.mouse_x += x
            self.mouse_y += y

        self.mouse_x = max(0, min(self.mouse_x, 65535))  # Ensure x is within screen bounds
        self.mouse_y = max(0, min(self.mouse_y, 65535))  # Ensure y is within screen bounds

        self.mouse.SetCursorPos((self.mouse_x, self.mouse_y))  # Set the mouse position

    def click(self, button):
        self.mouse.mouse_event(button, 0, 0, 0, 0)  # Simulate a mouse click

    def scroll(self, amount):
        self.mouse.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, amount, 0)  # Simulate a mouse scroll

    def press_key(self, key):
        self.mouse.keybd_event(key, 0, 0, 0)  # Simulate a key press

    def release_key(self, key):
        self.mouse.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)  # Simulate a key release

    def press_key_combination(self, keys):
        for key in keys:
            self.press_key(key)
        for key in keys:
            self.release_key(key)

    def move(self, x, y):
        self.mouse_x += x
        self.mouse_y += y
        self.mouse.SetCursorPos((self.mouse_x, self.mouse_y))  # Set the mouse position

    def run(self):
        while True:
            if not mouse_queue.empty():
                mouse_x, mouse_y, mouse_event_type = mouse_queue.get()
                if mouse_event_type == win32con.WM_MOUSEMOVE:  # Check for mouse movement events
                    self.process_mouse_input(mouse_x, mouse_y, True)  # True indicates raw input

# Raw Input Structures
class RAWINPUTDEVICE(Structure):
    _fields_ = [
        ("usUsagePage", c_ushort),
        ("usUsage", c_ushort),
        ("dwFlags", DWORD),
        ("hwndTarget", HANDLE),
    ]

class RID_DEVICE_INFO(Structure):
    _fields_ = [
        ("cbSize", UINT),
        ("dwType", UINT),
        ("hid", ctypes.c_void_p),  # Pointer to device-specific information
    ]

# Hook Structures and Constants
class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", POINT),
        ("mouseData", DWORD),
        ("flags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", ctypes.wintypes.LPARAM),  # Corrected type
    ]

class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", DWORD),
        ("scanCode", DWORD),
        ("flags", DWORD),
        ("time", c_int),
        ("dwExtraInfo", ctypes.wintypes.LPARAM),  # Corrected type
    ]

HC_ACTION = 0
WH_MOUSE_LL = 14
WH_KEYBOARD_LL = 13

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Function to get mouse IDs using WMI
def get_mouse_ids():
    c = wmi.WMI()
    for mouse in c.Win32_PointingDevice():
        if mouse.DeviceID:
            yield mouse.DeviceID

HC_ACTION = 0
WH_MOUSE_LL = 14
WH_KEYBOARD_LL = 13

# Function to create the mouse selection GUI
def select_mouse():
    connected_mice = wmi.WMI().Win32_PointingDevice()
    window = tk.Tk()
    window.title("Select Mouse")

    selected_mouse_id = tk.StringVar(value="None selected")

    for mouse in connected_mice:
        if mouse.DeviceID:
            radio_button = tk.Radiobutton(window, text=mouse.Name, variable=selected_mouse_id, value=mouse.DeviceID)
            radio_button.pack(anchor="w")

    def confirm_selection():
        print("Selected mouse:", selected_mouse_id.get())
        window.destroy()

    confirm_button = tk.Button(window, text="Confirm", command=confirm_selection)
    confirm_button.pack()

    window.mainloop()
    return selected_mouse_id.get() if selected_mouse_id.get() != "None selected" else None


# Mouse hook procedure
def mouse_hook_proc(nCode, wParam, lParam):
    global mouse_queue
    global output_method
    global mouse
    if nCode == HC_ACTION:
        msllhook = ctypes.cast(lParam, POINTER(MSLLHOOKSTRUCT)).contents
        x = msllhook.pt.x
        y = msllhook.pt.y
        mouse_event_type = wParam

        mouse_queue.put((x, y, mouse_event_type))  # Put all mouse events in the queue

    return user32.CallNextHookEx(mouse_hook, nCode, wParam, lParam)

output_method = "win32"

# Function to capture and retransmit all mouse input
def capture_and_retransmit_mouse_input(selected_mouse_id, mouse_queue, output_method="win32"):
    global mouse_hook
    mouse_output = None

    # Initialize the appropriate output method (Win32 by default)
    if output_method == "ghub":
        from logic.ghub import gHub  # Assuming you have a gHub module
        mouse_output = gHub()
    elif output_method == "arduino":
        from logic.arduino import arduino  # Assuming you have an arduino module
        mouse_output = arduino()
    else:
        mouse_output = win32api  # Default to Win32 API


    # Install the mouse hook here, outside the function
    CMPFUNC = ctypes.WINFUNCTYPE(c_int, c_int, WPARAM, LPARAM)
    mouse_hook = user32.SetWindowsHookExA(WH_MOUSE_LL, CMPFUNC(mouse_hook_proc), 0, 0)
    if not mouse_hook:
        print(f"Error installing mouse hook: {ctypes.WinError()}")
        sys.exit(1)

    # Message loop to keep the hook active (placed inside the function)
    msg = MSG()
    while True:
        if user32.PeekMessageA(byref(msg), 0, 0, 0, 1) != 0:
            if msg.message == win32con.WM_QUIT:
                break
            user32.TranslateMessage(byref(msg))
            user32.DispatchMessageW(byref(msg))

    user32.UnhookWindowsHookEx(mouse_hook)  # Uninstall the hook when exiting


# Main script logic
selected_mouse_id = select_mouse()

if selected_mouse_id:
    mouse_queue = queue.Queue() 
    mouse = MouseThread()  # Create MouseThread instance
    
    # Start the mouse capture thread
    output_method = "win32" # or "ghub" or "arduino"
    mouse_capture_thread = threading.Thread(target=capture_and_retransmit_mouse_input, args=(selected_mouse_id, mouse_queue, output_method))
    mouse_capture_thread.daemon = True
    mouse_capture_thread.start()

    # ... other code that runs after the capture_and_retransmit_mouse_input has started.
    while not mouse_queue.empty():
        mouse_x, mouse_y, mouse_event_type = mouse_queue.get()
        if mouse_event_type == win32con.WM_MOUSEMOVE:  # Check for mouse movement events
            mouse.process_mouse_input(mouse_x, mouse_y, True)  # True indicates raw input
        else:
            mouse.process_mouse_input(mouse_x, mouse_y, False) 
    
else:
    print("No mouse selected or no mice detected.")



# Define RAWINPUTHEADER structure
class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", DWORD),
        ("dwSize", DWORD),
        ("hDevice", HANDLE),
        ("wParam", ctypes.wintypes.WPARAM),
    ]

# Define RAWMOUSE structure
class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", ctypes.c_ushort),
        ("ulButtons", ctypes.c_ulong),
        ("ulRawButtons", ctypes.c_ulong),
        ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long),
        ("ulExtraInformation", ctypes.c_ulong),
    ]

 
# Function to get mouse IDs using raw input (you might not need this if using WMI exclusively)
def get_mouse_ids():
    rid_count = c_uint()
    windll.user32.GetRawInputDeviceList(None, byref(rid_count), ctypes.sizeof(RAWINPUTDEVICE))
    rid_list = (RAWINPUTDEVICE * rid_count.value)()
    windll.user32.GetRawInputDeviceList(rid_list, byref(rid_count), ctypes.sizeof(RAWINPUTDEVICE))

    mouse_ids = []
    for rid in rid_list:
        if rid.usUsagePage == 1 and rid.usUsage == 2:  # Filter for mice
            device_info = RID_DEVICE_INFO()
            device_info.cbSize = ctypes.sizeof(RID_DEVICE_INFO)
            windll.user32.GetRawInputDeviceInfoW(
                rid.hwndTarget,
                0x2000000B,  # RIDI_DEVICEINFO
                byref(device_info),
                byref(c_ulong(ctypes.sizeof(RID_DEVICE_INFO))),
            )
            device_id = wstring_at(device_info.hid).split("#")[-1]
            mouse_ids.append(device_id)
    return mouse_ids

# Main script logic
selected_mouse_id = select_mouse()

if selected_mouse_id:
    print("Selected mouse:", selected_mouse_id)
    # ... your custom logic here (use the selected_mouse_id)
else:
    print("No mouse selected or no mice detected.")
# Get list of connected mice
connected_mice = wmi.WMI().Win32_PointingDevice()

# Print connected mice
if connected_mice:
    print("Connected mice:")
    for mouse in connected_mice:
        print(mouse.DeviceID)
else:
    print("None of your target mice were detected.")
# Get list of connected mice
connected_mice = wmi.WMI().Win32_PointingDevice()


