
import tkinter as tk
import ctypes.util
import ctypes.wintypes
import ctypes
from ctypes import windll, Structure, POINTER, byref, c_ushort, c_uint, c_ulong, wstring_at
from ctypes.wintypes import DWORD, HANDLE, UINT, BOOL
from ctypes.wintypes import DWORD, HANDLE, UINT, BOOL
import wmi
import subprocess
import sys


c = wmi.WMI()
for mouse in c.Win32_PointingDevice():
    if mouse.DeviceID:
        print(mouse.DeviceID)

class RAWINPUTDEVICE(Structure):
    _fields_ = [("usUsagePage", c_ushort),
                ("usUsage", c_ushort),
                ("dwFlags", DWORD),
                ("hwndTarget", HANDLE)]

class RID_DEVICE_INFO(Structure):
    _fields_ = [("cbSize", UINT),
                ("dwType", UINT),
                ("hid", ctypes.c_void_p)]

def get_mouse_ids():
    rid_count = c_uint()
    result = windll.user32.GetRawInputDeviceList(None, byref(rid_count), ctypes.sizeof(RAWINPUTDEVICE))
    if result < 0:
        raise Exception("GetRawInputDeviceList failed")
    
    rid_list = (RAWINPUTDEVICE * rid_count.value)()
    result = windll.user32.GetRawInputDeviceList(rid_list, byref(rid_count), ctypes.sizeof(RAWINPUTDEVICE))
    if result < 0:
        raise Exception("GetRawInputDeviceList failed")

    mouse_ids = []
    for rid in rid_list:
        if rid.usUsagePage == 1 and rid.usUsage == 2:  # Mouse usage page and usage
            device_info = RID_DEVICE_INFO()
            device_info.cbSize = ctypes.sizeof(RID_DEVICE_INFO)
            result = windll.user32.GetRawInputDeviceInfoW(rid.hwndTarget, 0x2000000b, byref(device_info), byref(c_ulong(ctypes.sizeof(RID_DEVICE_INFO))))
            if result >= 0:  # Check for success
                device_id = wstring_at(device_info.hid).split("#")[-1]
                mouse_ids.append(device_id)
            else:
                print(f"Failed to get info for device: {rid.hwndTarget}")
    return mouse_ids


# Target mouse ID (you'd obtain this beforehand)
target_mouse_id = "USB\VID_XXXX&PID_YYYY&REV_ZZZZ" 



# Logic based on detected mouse
connected_mice = [mouse_id for mouse_id in wmi.WMI().Win32_PointingDevice()]

if connected_mice:
    print("Connected mice:")
    for mouse_id in connected_mice:
        print(mouse_id)
else:
    print("None of your target mice were detected.")
def select_mouse():
    # Get list of connected mice
    connected_mice = wmi.WMI().Win32_PointingDevice()

    # Create main window
    window = tk.Tk()
    window.title("Select Mouse")

    # Create a variable to store the selected mouse ID
    selected_mouse_id = tk.StringVar(value="None selected")

    # Create radio buttons for each detected mouse
    for mouse in connected_mice:
        if mouse.DeviceID:
            radio_button = tk.Radiobutton(window, text=mouse.Name, variable=selected_mouse_id, value=mouse.DeviceID)
            radio_button.pack(anchor="w")

    # Create a button to confirm the selection
    def confirm_selection():
        print("Selected mouse:", selected_mouse_id.get())
        window.destroy()

    confirm_button = tk.Button(window, text="Confirm", command=confirm_selection)
    confirm_button.pack()

    # Start the GUI event loop
    window.mainloop()

    # Return the selected mouse ID (or None if none selected)
    return selected_mouse_id.get() if selected_mouse_id.get() != "None selected" else None

# Main script logic
selected_mouse_id = select_mouse()

if selected_mouse_id:
    print("Selected mouse:", selected_mouse_id)
    # ... your custom logic here (use the selected_mouse_id)
else:
    print("No mouse selected or no mice detected.")
mouse_scripts = {
    "HID\\ELAN0406&COL01\\5&362E3420&0&0000": r"C:\path\to\your\script.py",
    # Add more mappings for other mouse IDs and scripts
}

# Main script logic
selected_mouse_id = select_mouse()

if selected_mouse_id:
    if selected_mouse_id in mouse_scripts:
        script_path = mouse_scripts[selected_mouse_id]
        subprocess.Popen([sys.executable, script_path])  # Run the associated script
    else:
        print("No script associated with this mouse.")
else:
    print("No mouse selected or no mice detected.")    