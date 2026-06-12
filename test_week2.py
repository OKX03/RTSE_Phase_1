import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
BACK_CAMERA_PORT = 8082
CONTROL_HOST = '127.0.0.1'
CONTROL_PORT = 8081


# Shared Resources with Mutex Lock for Concurrency
shared_data = {
    'latest_front_frame': None,
    'latest_back_frame': None,
    'steering_input' : 0.0,
    'acceleration_input' : 1.0,
    'tap_state': 'IDLE',      
    'debug_info': "WAITING",  
    'debug_tokens': [],        
    'net_lane_position': 0,
    'low_light': False
}
data_lock = threading.Lock()
is_running = True


# Persistent Global States
baseline_brightness = None


# Tapping control variables
tap_state = 'IDLE'            
tap_timer = 0
active_steering_value = 0.0
TAP_HOLD_FRAMES = 10        
COOLDOWN_FRAMES = 20        


# ---------------------------------------------------------
# Real-Time Scheduling Framework (Do not change this in your code)
# ---------------------------------------------------------
class TaskPriority:
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class RTTask(threading.Thread):
    """
    Real-Time Task implementing:
    - Concurrency (inherits threading.Thread)
    - Task Period (enforced in run loop)
    - Task Priority (logical priority assigned)
    """
    def __init__(self, name, period, priority, execute_func):
        super().__init__()
        self.name = name
        self.period = period
        self.priority = priority
        self.execute_func = execute_func
        self.daemon = True


    def run(self):
        print(f"[{self.name}] Started | Period: {self.period}s | Priority: {self.priority}")
        try:
            handle = ctypes.windll.kernel32.GetCurrentThread()
            if self.priority == TaskPriority.HIGH:
                ctypes.windll.kernel32.SetThreadPriority(handle, 2)
            elif self.priority == TaskPriority.MEDIUM:
                ctypes.windll.kernel32.SetThreadPriority(handle, 0)
            elif self.priority == TaskPriority.LOW:
                ctypes.windll.kernel32.SetThreadPriority(handle, -2)
        except Exception as e:
            pass


        while is_running:
            start_time = time.time()
            self.execute_func()
            exec_time = time.time() - start_time
            sleep_time = self.period - exec_time
           
            if sleep_time > 0:
                time.sleep(sleep_time)


# ---------------------------------------------------------
# Network Connection Setup (Do not change this in your code)
# ---------------------------------------------------------
front_camera_sock = None
back_camera_sock = None
control_conn = None


def setup_cameras():
    global front_camera_sock, back_camera_sock
   
    print("Connecting to Cameras...")
    front_connected = False
    back_connected = False
   
    while is_running and not (front_connected and back_connected):
        if not front_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
                front_camera_sock = s
                print("Connected to Front Camera successfully.")
                front_connected = True
            except Exception:
                pass
               
        if not back_connected:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect((CAMERA_HOST, BACK_CAMERA_PORT))
                back_camera_sock = s
                print("Connected to Back Camera successfully.")
                back_connected = True
            except Exception:
                pass
               
        if not (front_connected and back_connected):
            time.sleep(1)


def setup_control_server():
    global control_conn
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((CONTROL_HOST, CONTROL_PORT))
    server_sock.listen()
    server_sock.settimeout(1.0)
    print(f"Control server listening on {CONTROL_HOST}:{CONTROL_PORT}")
   
    while is_running:
        try:
            conn, addr = server_sock.accept()
            print(f"Control client connected from {addr}")
            control_conn = conn
            break
        except socket.timeout:
            continue


# ---------------------------------------------------------
# Task Implementations (This is where you write your tasks)
# ---------------------------------------------------------


def read_single_camera(sock, data_key):
    #This function reads the latest frame from the camera socket and stores it in the shared data
    if sock is None:
        return
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
               
        # Clear backlog to ensure real-time performance
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
    except Exception: pass


def read_front_camera_task():
    read_single_camera(front_camera_sock, 'latest_front_frame')


def read_back_camera_task():
    read_single_camera(back_camera_sock, 'latest_back_frame')


# ---------------------------------------------------------
# Simplified Perception & Decision
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
    if cx < margin_left or cx > margin_right: return []
   
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
    global baseline_brightness


    small_frame = cv2.resize(front_frame, (320, 240))
    roi_front = small_frame[ROI_START_Y:190, 0:320]
   
    # Gaussian Blur for stability
    blurred_roi = cv2.GaussianBlur(roi_front, (5, 5), 0)
    roi_hsv = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
   
    # Low Brightness Detection Logic
    gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    left_periph = gray_frame[120:180, 0:40]
    right_periph = gray_frame[120:180, 280:320]
    current_periph_brightness = (np.mean(left_periph) + np.mean(right_periph)) / 2.0
   
    if baseline_brightness is None:
        if current_periph_brightness > 40:
            baseline_brightness = current_periph_brightness
           
    low_light_mode = False
    if baseline_brightness is not None:
        if current_periph_brightness < (baseline_brightness * 0.3):
            low_light_mode = True
        elif current_periph_brightness > (baseline_brightness * 0.5):
            baseline_brightness = (baseline_brightness * 0.99) + (current_periph_brightness * 0.01)


    mask_green = cv2.inRange(roi_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    mask_red1 = cv2.inRange(roi_hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(roi_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_red = mask_red1 | mask_red2
    mask_yellow = cv2.inRange(roi_hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))


    contours_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    detected_objects = []
    debug_tokens = []


    for c in contours_red:
        area = cv2.contourArea(c)
        if area > 5:
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    dist = (y + h/2 + ROI_START_Y) - 80
                    detected_objects.append({'type': 'DANGER', 'lanes': lanes, 'dist': dist})
                    debug_tokens.append(('DANGER_RED', x*2, (y+ROI_START_Y)*2, w*2, h*2))


    for c in contours_yellow:
        area = cv2.contourArea(c)
        if area > 5:
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    dist = (y + h/2 + ROI_START_Y) - 80
                    detected_objects.append({'type': 'DANGER', 'lanes': lanes, 'dist': dist})
                    debug_tokens.append(('DANGER_YELLOW', x*2, (y+ROI_START_Y)*2, w*2, h*2))
   
    for c in contours_g:
        area = cv2.contourArea(c)
        if area > 5:
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    dist = (y + h/2 + ROI_START_Y) - 80
                    detected_objects.append({'type': 'GREEN', 'lanes': lanes, 'dist': dist})
                    debug_tokens.append(('GREEN', x*2, (y+ROI_START_Y)*2, w*2, h*2))
                   
    return detected_objects, debug_tokens, low_light_mode


def evaluate_decision(detected_objects, current_lane, low_light_mode):
    target_steer = 0.0
    target_accel = 1.0
    debug_text = "CRUISING"


    # Low brightness event reaction
    if low_light_mode:
        return 0.0, -1.0, "LOW LIGHT: BRAKING TO RECOVER"


    # Simplified tracking: Is there danger in left(-1), center(0), right(1)?
    danger_lanes = set()
    green_lanes = set()


    for obj in detected_objects:
        if obj['type'] == 'DANGER':
            for lane in obj['lanes']: danger_lanes.add(lane)
        elif obj['type'] == 'GREEN':
            for lane in obj['lanes']: green_lanes.add(lane)


    # Highest Priority: Evade Danger directly ahead
    if 0 in danger_lanes:
        if -1 not in danger_lanes:
            target_steer = -1.0
            debug_text = "<< EVADE LEFT"
        elif 1 not in danger_lanes:
            target_steer = 1.0
            debug_text = "EVADE RIGHT >>"
        else:
            # Trapped! Just pick right as default fallback
            target_steer = 1.0
            debug_text = "TRAPPED! PUSH RIGHT >>"
        return target_steer, target_accel, debug_text


    # Medium Priority: Seek Green if safe
    if green_lanes:
        if 0 in green_lanes:
            target_steer = 0.0
            debug_text = "SEEK GREEN AHEAD"
        elif -1 in green_lanes and -1 not in danger_lanes:
            target_steer = -1.0
            debug_text = "<< SEEK GREEN LEFT"
        elif 1 in green_lanes and 1 not in danger_lanes:
            target_steer = 1.0
            debug_text = "SEEK GREEN RIGHT >>"
        return target_steer, target_accel, debug_text


    # Lowest Priority: Auto-Center
    if current_lane < 0:
        target_steer = 1.0
        debug_text = "AUTO CENTER >>"
    elif current_lane > 0:
        target_steer = -1.0
        debug_text = "<< AUTO CENTER"
       
    return target_steer, target_accel, debug_text


def processing_task():
    #This is where you write your image processing code to decide how to control the car
    #You can use libraries like OpenCV to process the image
    #There is no limtation to the complexity of the processing task, you can use any libraries you want
    #Remember to use the shared_data to get the latest frame
    with data_lock:
        front_frame = shared_data['latest_front_frame']
        current_lane = shared_data.get('net_lane_position', 0)
       
    if front_frame is not None:
        # write your processing here
        detected_objects, debug_tokens, low_light_mode = detect_environment(front_frame)
        target_steer, target_accel, debug_text = evaluate_decision(detected_objects, current_lane, low_light_mode)


        with data_lock:
            shared_data['steering_input'] = target_steer
            shared_data['acceleration_input'] = target_accel
            shared_data['debug_tokens'] = debug_tokens
            shared_data['debug_info'] = f"AUTO: {debug_text}"
            shared_data['low_light'] = low_light_mode


def send_controls_task():
    #This is where you send the control commands to the car using the control_conn
    global control_conn, tap_state, tap_timer, active_steering_value
    if control_conn is None:
        return
   
    #these are the variables used to control the car
    #steering_input: -1.0 to 1.0 (left to right)
    #acceleration_input: -1.0 to 1.0 (reverse to forward)
    #this example always accelerate forward
    with data_lock:
        auto_steer = shared_data['steering_input']
        accel_input = shared_data['acceleration_input']


    # --- Autonomous Tapping Logic ---
    if tap_state == 'IDLE':
        if auto_steer != 0.0:
            active_steering_value = auto_steer
            tap_state = 'TAPPING'
            tap_timer = TAP_HOLD_FRAMES
           
            with data_lock:
                if auto_steer < -0.1: shared_data['net_lane_position'] = max(-1, shared_data.get('net_lane_position', 0) - 1)
                elif auto_steer > 0.1: shared_data['net_lane_position'] = min(1, shared_data.get('net_lane_position', 0) + 1)
        else: active_steering_value = 0.0
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
        # Pack and send the control command
        data = struct.pack('ff', active_steering_value, accel_input)
        control_conn.sendall(data)
    except Exception as e:
        print(f"Control send error: {e}")
        control_conn = None




# ---------------------------------------------------------
# Main (Scheduler Initialization)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("Initializing Phase 1 RTSE Drive...")
   
    # Initialize network connections
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
   
    print("\n--- Starting Real-Time Tasks (awaiting connections dynamically) ---\n")
   
    # This is where you define tasks with explicit Scheduling parameters (Concurrency, Priority, Period)
    # Period refers to the period of execution of the task in seconds
    # Priority refers to the priority of the task, higher priority means higher priority
    # Concurrency refers to the number of instances of the task that can run at the same time
    t_front_camera = RTTask("ReadFrontCamera", period=0.01, priority=TaskPriority.LOW, execute_func=read_front_camera_task)
    t_back_camera = RTTask("ReadBackCamera", period=0.01, priority=TaskPriority.LOW, execute_func=read_back_camera_task)
    t_processing = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    t_controls = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
   
    # Start tasks to run concurrently
    t_front_camera.start()
    t_back_camera.start()
    t_processing.start()
    t_controls.start()
   
    display_paused = False
    last_display_frame = None


    print("\n=============================================")
    print(" PRESS 'p' TO PAUSE VIDEO FEED")
    print(" PRESS 'q' TO QUIT")
    print("=============================================\n")


    try:
        # You need this to keep the main thread alive, otherwise the program will exit immediately
        while is_running:
            with data_lock:
                front_frame = shared_data['latest_front_frame']
                back_frame = shared_data.get('latest_back_frame', None)
                debug_info = shared_data['debug_info']
                debug_tokens = shared_data['debug_tokens'].copy()
                steer_input = shared_data['steering_input']
                low_light = shared_data.get('low_light', False)


            key = cv2.waitKey(1) & 0xFF
            if key == ord('p') or key == ord(' '):
                display_paused = not display_paused
            elif key == ord('q'):
                is_running = False


            if front_frame is not None and not display_paused:
                display_front = cv2.resize(front_frame, (640, 480))
               
                cv2.putText(display_front, debug_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
               
                if low_light:
                    cv2.putText(display_front, "LOW LIGHT DETECTED", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)


                cv2.line(display_front, (0, 200), (640, 200), (255, 0, 0), 2)
                cv2.line(display_front, (0, 440), (640, 440), (255, 0, 0), 2)
                cv2.putText(display_front, "ROI BOUNDARY", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


                cv2.line(display_front, (320 - int(20*0.22*2), 200), (320 - int(160*0.22*2), 480), (255, 255, 255), 2)
                cv2.line(display_front, (320 + int(20*0.22*2), 200), (320 + int(160*0.22*2), 480), (255, 255, 255), 2)
               
                for token_data in debug_tokens:
                    if len(token_data) >= 5:
                        ttype, x, y, w, h = token_data[:5]
                        color = (0, 0, 255) if 'RED' in ttype else (0, 255, 0)
                        cv2.rectangle(display_front, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(display_front, ttype, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


                # Draw Action text from sample_drive1
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
                cv2.imshow("Front Camera", pause_frame)
            elif last_display_frame is not None:
                cv2.imshow("Front Camera", last_display_frame)
               
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected. Stopping system...")
        is_running = False


    # This is to make sure that the tasks are terminated cleanly
    t_front_camera.join()
    t_back_camera.join()
    t_processing.join()
    t_controls.join()
   
    # This is to close all the connections
    if front_camera_sock:
        front_camera_sock.close()
    if back_camera_sock:
        back_camera_sock.close()
    if control_conn:
        control_conn.close()
    cv2.destroyAllWindows()
    print("System terminated cleanly.")
