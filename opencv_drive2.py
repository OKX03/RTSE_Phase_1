import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes

try:
    import keyboard
    KEYBOARD_ENABLED = True
except ImportError:
    print("WARNING: keyboard library not installed. Some events may require manual key presses.")
    KEYBOARD_ENABLED = False

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081

shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,        
    'acceleration_input' : 1.0,    
    'tap_state': 'IDLE',
    'debug_info': "",
    'debug_tokens': [],
}
data_lock = threading.Lock()
is_running = True

# Aggressive Tapping Control (Tuned for High Speed)
tap_state = 'IDLE'            
tap_timer = 0
active_steering_value = 0.0
TAP_HOLD_FRAMES = 8          # Reduced for sharper, faster lane switches
COOLDOWN_FRAMES = 12         # Reduced to allow rapid double-lane changes

class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class RTTask(threading.Thread):
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True

    def run(self):
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH: ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.LOW: ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except: pass

        while is_running:
            start_time = time.time()
            self.execute_func()
            sleep_time = self.period - (time.time() - start_time)
            if sleep_time > 0: time.sleep(sleep_time)

# ---------------------------------------------------------
# Network Setup
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None

def setup_cameras():
    global front_camera_sock, back_camera_sock
    front_connected, back_connected = False, False
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock, front_connected = s, True
            except: pass
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock, back_connected = s, True
            except: pass
        if not (front_connected and back_connected): time.sleep(1)

def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    while is_running:
        try:
            control_conn, _ = server_sock.accept()
            break
        except: pass

def read_single_camera(sock, data_key):
    if sock is None: return
    try:
        sock.settimeout(None)
        length_bytes = sock.recv(4)
        if not length_bytes: return
        image_length = int.from_bytes(length_bytes, 'little')
        received_bytes = b''
        while len(received_bytes) < image_length and is_running:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet: break
            received_bytes += packet
        if len(received_bytes) == image_length:
            np_arr = np.frombuffer(received_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                with data_lock: shared_data[data_key] = frame
                
        while is_running:
            readable, _, _ = select.select([sock], [], [], 0.0)
            if not readable: break
            sock.settimeout(1.0)
            length_bytes = sock.recv(4)
            if not length_bytes: return
            image_length = int.from_bytes(length_bytes, 'little')
            received_bytes = b''
            while len(received_bytes) < image_length and is_running:
                packet = sock.recv(image_length - len(received_bytes))
                if not packet: break
                received_bytes += packet
            if len(received_bytes) == image_length:
                np_arr = np.frombuffer(received_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    with data_lock: shared_data[data_key] = frame
    except: pass

def read_front_camera_task(): read_single_camera(front_camera_sock, 'latest_front_frame')
def read_back_camera_task(): read_single_camera(back_camera_sock, 'latest_back_frame')

# ---------------------------------------------------------
# Modular Pipeline: Perception & Decision
# ---------------------------------------------------------
ROI_START_Y = 100

def get_occupied_lanes(x, y, w, h):
    actual_y = y + h/2 + ROI_START_Y
    dist_to_horizon = actual_y - 80 
    
    if dist_to_horizon <= 0: return []
    
    margin_width = dist_to_horizon * 0.857
    margin_left = 160 - margin_width
    margin_right = 160 + margin_width
    
    cx = x + w/2
    if cx < margin_left or cx > margin_right:
        return []
    
    lane_half_width = dist_to_horizon * 0.22
    left_bound = 160 - lane_half_width
    right_bound = 160 + lane_half_width
    
    token_l = x
    token_r = x + w
    
    lanes = []
    if token_l <= left_bound and token_r >= margin_left: lanes.append(-1)
    if token_l <= right_bound and token_r >= left_bound: lanes.append(0)
    if token_l <= margin_right and token_r >= right_bound: lanes.append(1)
    
    return lanes

def detect_environment(front_frame):
    small_frame = cv2.resize(front_frame, (320, 240))
    roi_front = small_frame[ROI_START_Y:220, 0:320]
    roi_hsv = cv2.cvtColor(roi_front, cv2.COLOR_BGR2HSV)
    
    mask_green = cv2.inRange(roi_hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
    mask_red1 = cv2.inRange(roi_hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(roi_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_yellow = cv2.inRange(roi_hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    mask_danger = mask_red1 | mask_red2 | mask_yellow
    
    contours_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_d, _ = cv2.findContours(mask_danger, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    debug_tokens = []

    for c in contours_d:
        area = cv2.contourArea(c)
        if area > 10:  # Increased minimum area to ignore distant noise
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    # Added 'y' for depth sorting
                    detected_objects.append({'type': 'DANGER', 'lanes': lanes, 'area': area, 'y': y})
                    debug_tokens.append(('DANGER', x*2, (y+ROI_START_Y)*2, w*2, h*2))

    for c in contours_g:
        area = cv2.contourArea(c)
        if area > 10:
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    detected_objects.append({'type': 'GREEN', 'lanes': lanes, 'area': area, 'y': y})
                    debug_tokens.append(('GREEN', x*2, (y+ROI_START_Y)*2, w*2, h*2))
    
    # CRITICAL FIX: Sort objects by Y (descending) so the closest tokens are evaluated first
    detected_objects.sort(key=lambda obj: obj['y'], reverse=True)
                    
    return detected_objects, debug_tokens

def evaluate_decision(detected_objects):
    dangers_in_lane = {-1: False, 0: False, 1: False}
    greens_in_lane = {-1: False, 0: False, 1: False}

    # Populate lane states based on the closest objects
    for obj in detected_objects:
        if obj['type'] == 'DANGER':
            for lane in obj['lanes']:
                if not dangers_in_lane[lane]: dangers_in_lane[lane] = True
        elif obj['type'] == 'GREEN':
            for lane in obj['lanes']:
                if not greens_in_lane[lane]: greens_in_lane[lane] = True

    target_steer = 0.0
    debug_text = "CRUISING"

    # 1. Immediate Evasion (Survival)
    if dangers_in_lane[0]:
        if not dangers_in_lane[1] and greens_in_lane[1]:
            target_steer = 1.0  
            debug_text = "EVADE RIGHT TO GREEN >>"
        elif not dangers_in_lane[-1] and greens_in_lane[-1]:
            target_steer = -1.0 
            debug_text = "<< EVADE LEFT TO GREEN"
        elif not dangers_in_lane[1]: 
            target_steer = 1.0  
            debug_text = "EVADE RIGHT >>"
        elif not dangers_in_lane[-1]: 
            target_steer = -1.0 
            debug_text = "<< EVADE LEFT"
        else:
            target_steer = 1.0  
            debug_text = "TRAPPED!"
            
    # 2. Aggressive Reward Seeking (No auto-centering)
    else:
        if greens_in_lane[0]:
            target_steer = 0.0
            debug_text = "HOLD CENTER FOR GREEN"
        elif greens_in_lane[-1] and not dangers_in_lane[-1]:
            target_steer = -1.0
            debug_text = "SEEK GREEN LEFT <<"
        elif greens_in_lane[1] and not dangers_in_lane[1]:
            target_steer = 1.0
            debug_text = "SEEK GREEN RIGHT >>"
        else:
            target_steer = 0.0
            debug_text = "CRUISING"

    return target_steer, debug_text

def processing_task():
    with data_lock: 
        front_frame = shared_data['latest_front_frame']
        
    if front_frame is not None:
        detected_objects, debug_tokens = detect_environment(front_frame)
        target_steer, debug_text = evaluate_decision(detected_objects)

        with data_lock:
            shared_data['steering_input'] = target_steer
            shared_data['debug_tokens'] = debug_tokens
            shared_data['debug_info'] = debug_text
            
def send_controls_task():
    global control_conn, tap_state, tap_timer, active_steering_value
    if control_conn is None: return
    
    with data_lock:
        desired_steer = shared_data['steering_input']
        accel_input = shared_data['acceleration_input']

    if tap_state == 'IDLE':
        if desired_steer != 0.0:
            active_steering_value = desired_steer
            tap_state = 'TAPPING'
            tap_timer = TAP_HOLD_FRAMES
        else: 
            active_steering_value = 0.0
    elif tap_state == 'TAPPING':
        if tap_timer > 0: tap_timer -= 1
        else:
            active_steering_value = 0.0
            tap_state = 'COOLDOWN'
            tap_timer = COOLDOWN_FRAMES
    elif tap_state == 'COOLDOWN':
        active_steering_value = 0.0
        if tap_timer > 0: tap_timer -= 1
        else: tap_state = 'IDLE'

    try:
        data = struct.pack('ff', active_steering_value, accel_input)
        control_conn.sendall(data)
    except: control_conn = None

if __name__ == '__main__':
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    t_ctrl = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    t_proc = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_front = RTTask("ReadFrontCamera", period=0.01, priority=TaskPriority.LOW, execute_func=read_front_camera_task)
    t_back = RTTask("ReadBackCamera", period=0.01, priority=TaskPriority.LOW, execute_func=read_back_camera_task)
    
    t_front.start(); t_back.start(); t_ctrl.start(); t_proc.start()
    
    display_paused = False
    last_display_frame = None

    try:
        while is_running:
            with data_lock:
                front_frame = shared_data['latest_front_frame']
                back_frame = shared_data.get('latest_back_frame', None)
                debug_info = shared_data['debug_info']
                debug_tokens = shared_data['debug_tokens'].copy()
                steer_input = shared_data['steering_input']

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p') or key == ord(' '):
                display_paused = not display_paused
            elif key == ord('q'):
                is_running = False

            if front_frame is not None and not display_paused:
                display_front = cv2.resize(front_frame, (640, 480))
                cv2.putText(display_front, debug_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.line(display_front, (0, 200), (640, 200), (255, 0, 0), 2)
                cv2.line(display_front, (0, 440), (640, 440), (255, 0, 0), 2)
                cv2.putText(display_front, "ROI BOUNDARY", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                cv2.line(display_front, (320 - int(20*0.22*2), 200), (320 - int(160*0.22*2), 480), (255, 255, 255), 2)
                cv2.line(display_front, (320 + int(20*0.22*2), 200), (320 + int(160*0.22*2), 480), (255, 255, 255), 2)
                cv2.putText(display_front, "SAFE CORRIDOR", (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                for (ttype, x, y, w, h) in debug_tokens:
                    color = (0, 0, 255) if ttype == 'DANGER' else (0, 255, 0)
                    cv2.rectangle(display_front, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(display_front, ttype, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                action_text = "STRAIGHT"
                if steer_input < -0.1: action_text = "<< STEER LEFT <<"
                elif steer_input > 0.1: action_text = ">> STEER RIGHT >>"
                cv2.putText(display_front, f"ACTION: {action_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                last_display_frame = display_front
                
            if back_frame is not None and not display_paused:
                display_back = cv2.resize(back_frame, (320, 240))
                cv2.imshow("Back Camera", display_back)

            if display_paused and last_display_frame is not None:
                pause_frame = last_display_frame.copy()
                cv2.putText(pause_frame, "PAUSED (Press 'p' to resume)", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("Fast OpenCV Drive", pause_frame)
            elif last_display_frame is not None:
                cv2.imshow("Fast OpenCV Drive", last_display_frame)
                
    except KeyboardInterrupt: is_running = False

    t_front.join(); t_back.join(); t_proc.join(); t_ctrl.join()
    cv2.destroyAllWindows()