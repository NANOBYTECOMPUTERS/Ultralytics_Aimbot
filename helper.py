# python default
import re
import os, ctypes
import os.path
import sys, subprocess
import time 
import shutil
import zipfile
import winreg

reload_prestart = False

try:
    from tqdm import tqdm
except:
    os.system('pip install tqdm')
    reload_prestart = True
try:
    import requests
except:
    os.system('pip install requests')
    reload_prestart = True
try:
    import cuda
except:
    os.system('pip install cuda_python')
    reload_prestart = True
try:
    import bettercam
except:
    os.system('pip install bettercam')
    reload_prestart = True
try:
    import numpy
except:
    os.system('pip install numpy')
    reload_prestart = True
try:
    import win32gui, win32ui, win32con
except:
    os.system('pip install pywin32')
    reload_prestart = True
try:
    import ultralytics
    from ultralytics import YOLO
except:
    os.system('pip install ultralytics')
    reload_prestart = True
try:
    import screeninfo
except:
    os.system('pip install screeninfo')
    reload_prestart = True
try:
    import asyncio
except:
    os.system('pip install asyncio')
    reload_prestart = True
try:
    import onnxruntime
except:
    os.system('pip install onnxruntime onnxruntime-gpu')
    reload_prestart = True
try:
    import serial
except:
    os.system('pip install pyserial')
try:
    import torch
except:
    os.system('pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124')
    reload_prestart = True
try:
    import cv2
except:
    os.system('pip install opencv-python')
    reload_prestart = True
try:
    import filterpy
except:
    os.system('pip install filterpy')
    reload_prestart = True
        
if reload_prestart:
    os.system('py helper.py')
    print('restarting...')
    quit()
    
def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(filename, 'wb') as file:
        for data in response.iter_content(1024):
            progress_bar.update(len(data))
            file.write(data)
    
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("Error with downloading file.")

def get_system_path():
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_READ) as key:
        return winreg.QueryValueEx(key, 'Path')[0]

def set_system_path(new_path):
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SYSTEM\CurrentControlSet\Control\Session Manager\Environment', 0, winreg.KEY_WRITE) as key:
        winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path)
    from ctypes import windll
    windll.user32.SendMessageTimeoutA(0xFFFF, 0x001A, 0, None, 0x02, 1000, None)

def upgrade_ultralytics():
    print('Checks new ultralytics version...')
    ultralytics_current_version = ultralytics.__version__

    ultralytics_repo_version = requests.get('https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/__init__.py').content.decode('utf-8')
    ultralytics_repo_version = re.search(r"__version__\s*=\s*\"([^\"]+)", ultralytics_repo_version).group(1)

    if ultralytics_current_version != ultralytics_repo_version:
        print('The versions of ultralytics do not match\nAn update is in progress...')
        os.system('pip install ultralytics --upgrade')
    else:
        os.system('cls')

def upgrade_pip(): # TODO newest version checks
    print('Checks new pip version...')
    ver = os.popen('pip -V').read().split(' ')[1]
    if ver != '24.0':
        print('The pip version does not match the required one.\nAn update is in progress...')
        os.system("python -m pip install --upgrade pip")
    else:
        os.system('cls')


def Install_TensorRT():
    cuda = find_cuda_path()
    if cuda is not None:
        if not os.path.isfile('TensorRT-10.0.0.6.Windows10.win10.cuda-12.4.zip') and os.path.isdir('TensorRT-10.0.0.6') == False:
            print('TensorRT in not downloaded\nDownloading TensorRT...')
            download_file('https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.0.0/zip/TensorRT-10.0.0.6.Windows10.win10.cuda-12.4.zip', 'TensorRT-10.0.0.6.Windows10.win10.cuda-12.4.zip')
        
        if not os.path.isdir('TensorRT-10.0.0.6'):
            print('Unpacking the TensorRT archive, please wait...')
            with zipfile.ZipFile(r'./TensorRT-10.0.0.6.Windows10.win10.cuda-12.4.zip', 'r') as zip_ref:
                zip_ref.extractall('./')
        
        os.system('pip install ./TTensorRT-10.0.0.6/python/tensorrt-10.0.0b6-cp311-none-win_amd64.whl')

        current_path = get_system_path()
        tensorrt_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'TensorRT-10.0.0.6\\lib')

        if tensorrt_lib_path not in current_path:
            new_path = current_path + ';' + tensorrt_lib_path
            set_system_path(new_path)
            print(f'New path added: {tensorrt_lib_path}')
        else:
            print(f'Env path already exists: {tensorrt_lib_path}')

        tensorrt_lib_files = ['nvinfer.dll', 'nvinfer.lib', 'nvinfer_builder_resource.dll', 'nvinfer_dispatch.dll', 'nvinfer_dispatch.lib', 'nvinfer_lean.dll', 'nvinfer_lean.lib', 'nvinfer_plugin.dll', 'nvinfer_plugin.lib', 'nvinfer_vc_plugin.dll', 'nvinfer_vc_plugin.lib', 'nvonnxparser.dll', 'nvonnxparser.lib']
        
        for cuda_path in cuda:
            if 'bin' in cuda_path:
                for lib in tensorrt_lib_files:
                    shutil.copy2('{0}\TensorRT-10.0.0.6\lib\\{1}'.format(os.path.join(os.path.dirname(os.path.abspath(__file__))), lib), cuda_path)
    else:
        print('First install cuda 12.4.')
        
def Install_cuda():
    os.system('cls')
    print('Cuda 12.4 is being downloaded, and installation will begin after downloading.')
    download_file('https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_551.61_windows.exe', './cuda_12.4.0_551.61_windows.exe')
    subprocess.call('{}/cuda_12.4.0_551.61_windows.exe'.format(os.path.join(os.path.dirname(os.path.abspath(__file__)))))
    
def Test_detections():
    cuda_support = ultralytics.utils.checks.cuda_is_available()
    if cuda_support == True:
        print('Cuda support True')
    else:
        print('Cuda is not supported\nTrying to reinstall torch with GPU support...')
        force_reinstall_torch()
        
    model = YOLO('models/{}'.format(cfg.AI_model_name), task='detect')
    cap = cv2.VideoCapture('media/tests/test_det.mp4')
    cv2.namedWindow('Model: {0} imgsz: {1}'.format(cfg.AI_model_name, cfg.ai_model_image_size))
    debug_window_hwnd = win32gui.FindWindow(None, 'Model: {0} imgsz: {1}'.format(cfg.AI_model_name, cfg.ai_model_image_size))
    win32gui.SetWindowPos(debug_window_hwnd, win32con.HWND_TOPMOST, 100, 100, 200, 200, 0)
    
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            result = model(frame, stream=False, show=False, imgsz=cfg.ai_model_image_size, device=cfg.AI_device, verbose=False, conf=0.40)
            annotated_frame = result[0].plot()

            cv2.putText(annotated_frame, 'TEST 1234567890', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imshow('Model: {0} imgsz: {1}'.format(cfg.AI_model_name, cfg.ai_model_image_size), annotated_frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    
def force_reinstall_torch():
    os.system('pip uninstall torch torchvision torchaudio')
    os.system('pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')

def print_menu():
    os.system('cls')
    print('Run this script as an administrator for install TensorRT correctly.')
    print('Installed version: {0}, online version: {1}\n'.format(get_aimbot_current_version()[0], get_aimbot_online_version()[0]))

    print("1: null")
    print("2: Download Cuda 12.4")
    print("3: Install TensorRT 10")
    print("4: Test the object detector")
    print("5: Force reinstall Torch (GPU)")
    print("0: Exit")

def main():
    try:
        while True:
            print_menu()
            choice = input("Select an option: ")

            if choice == "1":
               print("Incorrect input, try again.")
            
            elif choice == "2":
                Install_cuda()
            
            elif choice == "3":
                Install_TensorRT()
                
            elif choice == "4":
                Test_detections()
                
            elif choice == "5":
                force_reinstall_torch()
                
            elif choice == "0":
                print("Exiting the program...")
                break
            
            else:
                print("Incorrect input, try again.")
    except:
        quit()

if __name__ == "__main__":
    try:
        from logic.config_watcher import cfg
    except:
        print('File config_watcher.py not found, reinstalling...')
    upgrade_pip()
    upgrade_ultralytics()
    main()
