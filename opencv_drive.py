import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes
import random

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
    'light_on': False,
    'police_mode': False,
    'net_lane_position': 0,
    'yellow_effect_timer': 0,    # 黄色效果计时器
    'yellow_effect_type': 0      # 黄色效果类型 (1-5)
}
data_lock = threading.Lock()
is_running = True

# Tapping control variables
tap_state = 'IDLE'            
tap_timer = 0
active_steering_value = 0.0
TAP_HOLD_FRAMES = 10         # Reduced for faster tapping response
COOLDOWN_FRAMES = 20         # Reduced for faster cooldown

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
                
        # Consume backlog to prevent delay build-up
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
    """
    PERCEPTION MODULE: 
    Outputs a structured list of detected objects. 
    """
    small_frame = cv2.resize(front_frame, (320, 240))
    # 🚨 核心修复 4 & 5：将底部裁剪到 190，彻底屏蔽玩家自己的红色车尾灯，防止“被自己吓到”而向右乱躲
    roi_front = small_frame[ROI_START_Y:190, 0:320]
    roi_hsv = cv2.cvtColor(roi_front, cv2.COLOR_BGR2HSV)
    
    # 🚨 核心修复 1：降低绿色饱和度和亮度阈值，捕获因为光照显得发白/发淡的绿币
    mask_green = cv2.inRange(roi_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    mask_red1 = cv2.inRange(roi_hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(roi_hsv, np.array([170, 120, 70]), np.array([180, 255, 255]))
    mask_yellow = cv2.inRange(roi_hsv, np.array([15, 100, 100]), np.array([35, 255, 255]))
    
    contours_g, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_red = mask_red1 | mask_red2
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 🚨 速度状态视觉侦测 (Optical UI Recognition)
    # Unity 界面中，当速度跌至 0.30x (或 0.60x 等低速) 时，右上角的文字会变成红色。
    # 我们提取右上角区域，检测是否存在大量红色像素，即可判断是否陷入“低速深渊”。
    speed_roi = small_frame[10:60, 250:320]
    speed_hsv = cv2.cvtColor(speed_roi, cv2.COLOR_BGR2HSV)
    # 稍微放宽红色检测范围，防止 Unity 的 UI 红色不够纯
    mask_red_speed = cv2.inRange(speed_hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) | cv2.inRange(speed_hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
    is_desperate = (cv2.countNonZero(mask_red_speed) > 10)

    detected_objects = []
    debug_tokens = []

    # 扫描红色危险 (优先级最高)
    for c in contours_red:
        area = cv2.contourArea(c)
        if area > 5:
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    dist = (y + h/2 + ROI_START_Y) - 80
                    detected_objects.append({'type': 'DANGER', 'color': 'RED', 'priority': 1, 'lanes': lanes, 'area': area, 'dist': dist})
                    debug_tokens.append(('DANGER_RED', x*2, (y+ROI_START_Y)*2, w*2, h*2))
    
    # 扫描黄色危险
    for c in contours_yellow:
        area = cv2.contourArea(c)
        if area > 5:
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    dist = (y + h/2 + ROI_START_Y) - 80
                    detected_objects.append({'type': 'DANGER', 'color': 'YELLOW', 'priority': 2, 'lanes': lanes, 'area': area, 'dist': dist})
                    debug_tokens.append(('DANGER_YELLOW', x*2, (y+ROI_START_Y)*2, w*2, h*2))
                    
    # 正常情况：检测绿币
    for c in contours_g:
        area = cv2.contourArea(c)
        if area > 5:
            x, y, w, h = cv2.boundingRect(c)
            if 0.3 < float(w)/h < 3.0:
                lanes = get_occupied_lanes(x, y, w, h)
                if lanes:
                    dist = (y + h/2 + ROI_START_Y) - 80
                    detected_objects.append({'type': 'GREEN', 'lanes': lanes, 'area': area, 'dist': dist})
                    debug_tokens.append(('GREEN', x*2, (y+ROI_START_Y)*2, w*2, h*2))
                    
    return detected_objects, debug_tokens, is_desperate

def evaluate_decision(detected_objects, current_lane, is_desperate):
    target_steer = 0.0
    debug_text = "CRUISING"

    # --- 第一步：引入时空预判 (距离映射威胁度) ---
    # 距离我们越近 (dist越大) 威胁越大。-1代表该车道绝对安全
    closest_danger_dist = {-1: -1, 0: -1, 1: -1}
    max_danger_priority = {-1: 0, 0: 0, 1: 0}
    
    best_green_lane = None
    max_green_dist = -1

    for obj in detected_objects:
        if obj['type'] == 'DANGER':
            priority = obj.get('priority', 2)
            for lane in obj['lanes']:
                # 记录该车道最靠近车头的危险物距离 (dist越大越近)
                if obj['dist'] > closest_danger_dist[lane]:
                    closest_danger_dist[lane] = obj['dist']
                    max_danger_priority[lane] = priority
        elif obj['type'] == 'GREEN':
            # 优先吃正前方的，否则吃距离我们最近的
            if 0 in obj['lanes']: 
                best_green_lane = 0
                max_green_dist = obj['dist']
            elif obj['dist'] > max_green_dist and best_green_lane != 0:
                max_green_dist = obj['dist']
                best_green_lane = obj['lanes'][0]

    # --- 🚨 绝境求生模式 (Desperate Mode) ---
    # 当速度跌入 0.3x 深渊时，UI文字变红 (is_desperate = True)。
    # 此时如果视野内有绿币，吃绿币的优先级绝对大于闪避危险！
    # 即使撞上红黄币也无所谓，因为速度已经见底了，唯有吃绿币才能爬出深渊！
    if is_desperate and best_green_lane is not None:
        if best_green_lane == -1:
            target_steer = -1.0
            debug_text = "DESPERATE: SEEK GREEN <<"
        elif best_green_lane == 1:
            target_steer = 1.0
            debug_text = "DESPERATE: SEEK GREEN >>"
        else:
            target_steer = 0.0
            debug_text = "DESPERATE: SEEK GREEN ^"
        return target_steer, debug_text

    # --- 核心保命规避逻辑 (正常情况下的最高优先级) ---
    # 只要正前方有危险 (距离 > 0)
    danger_0 = closest_danger_dist[0]
    
    if danger_0 > 0:
        # 两害相权取其轻：比较左右哪边的危险离我们更远 (dist更小)
        danger_left = closest_danger_dist[-1]
        danger_right = closest_danger_dist[1]
        
        if danger_right < danger_left:
            target_steer = 1.0  
            debug_text = f"EVADE RIGHT (Safest)"
        elif danger_left < danger_right:
            target_steer = -1.0 
            debug_text = f"<< EVADE LEFT (Safest)"
        else:
            # 左右一样危险，或者都没危险
            if danger_left == -1:
                target_steer = 1.0
                debug_text = f"EVADE RIGHT"
            else:
                target_steer = 1.0  
                debug_text = f"TRAPPED! PUSH RIGHT"
        
        return target_steer, debug_text

    # --- 第三步：在绝对安全的情况下，执行正常逻辑 ---
    
    # 正常模式：吃绿币或回中
    if best_green_lane is not None:
        if best_green_lane == -1:
            # 吃左边绿币前提：左边绝对安全，或者左边危险离得很远(至少比绿币远20)
            if closest_danger_dist[-1] == -1 or (max_green_dist > closest_danger_dist[-1] + 20):
                target_steer = -1.0
                debug_text = "SEEK GREEN LEFT"
            else:
                target_steer = 0.0
                debug_text = "ABORT GREEN (DANGER LEFT)"
        elif best_green_lane == 1:
            if closest_danger_dist[1] == -1 or (max_green_dist > closest_danger_dist[1] + 20):
                target_steer = 1.0
                debug_text = "SEEK GREEN RIGHT"
            else:
                target_steer = 0.0
                debug_text = "ABORT GREEN (DANGER RIGHT)"
        else:
            target_steer = 0.0 
            debug_text = "STEADY"
    else:
        # 空闲回中 (Auto-Center)
        if current_lane < 0:
            target_steer = 1.0
            debug_text = "AUTO CENTER >>"
        elif current_lane > 0:
            target_steer = -1.0
            debug_text = "<< AUTO CENTER"
        else:
            target_steer = 0.0
            debug_text = "CRUISING"
            
    return target_steer, debug_text

def processing_task():
    """ 
    ORCHESTRATOR MODULE: 
    Controls the flow of data through the Perception and Decision nodes.
    """
    with data_lock: 
        front_frame = shared_data['latest_front_frame']
        current_lane = shared_data.get('net_lane_position', 0)
        
    if front_frame is not None:
        detected_objects, debug_tokens, is_desperate = detect_environment(front_frame)
        target_steer, debug_text = evaluate_decision(detected_objects, current_lane, is_desperate)

        with data_lock:
            shared_data['steering_input'] = target_steer
            shared_data['debug_tokens'] = debug_tokens
            shared_data['debug_info'] = debug_text
            shared_data['is_desperate'] = is_desperate
            
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
            
            with data_lock:
                if desired_steer < -0.1: shared_data['net_lane_position'] = max(-1, shared_data.get('net_lane_position', 0) - 1)
                elif desired_steer > 0.1: shared_data['net_lane_position'] = min(1, shared_data.get('net_lane_position', 0) + 1)
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

    with data_lock: shared_data['tap_state'] = f"{tap_state} ({tap_timer})"

    try:
        data = struct.pack('ff', active_steering_value, accel_input)
        control_conn.sendall(data)
    except: control_conn = None

if __name__ == '__main__':
    threading.Thread(target=setup_control_server, daemon=True).start()
    threading.Thread(target=setup_cameras, daemon=True).start()
    
    # RTOS Scheduling Adjustments:
    # Priority: Control > Processing > Camera Read (Rate Monotonic Implementation)
    t_ctrl = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    t_proc = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    
    # Camera read periods extended slightly and priority lowered to ensure actuation is never blocked
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
                is_desperate_flag = shared_data.get('is_desperate', False)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p') or key == ord(' '):
                display_paused = not display_paused
            elif key == ord('q'):
                is_running = False

            if front_frame is not None and not display_paused:
                display_front = cv2.resize(front_frame, (640, 480))
                cv2.putText(display_front, debug_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw Visual Debug for Desperate Mode
                cv2.rectangle(display_front, (500, 20), (640, 120), (0, 255, 255), 1)
                cv2.putText(display_front, "SPEED ROI", (510, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                if is_desperate_flag:
                    cv2.putText(display_front, "!!! DESPERATE MODE !!!", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                # Draw ROI bounds
                cv2.line(display_front, (0, 200), (640, 200), (255, 0, 0), 2)
                cv2.line(display_front, (0, 440), (640, 440), (255, 0, 0), 2)
                cv2.putText(display_front, "ROI BOUNDARY", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Draw Safe Corridor (White Lines)
                cv2.line(display_front, (320 - int(20*0.22*2), 200), (320 - int(160*0.22*2), 480), (255, 255, 255), 2)
                cv2.line(display_front, (320 + int(20*0.22*2), 200), (320 + int(160*0.22*2), 480), (255, 255, 255), 2)
                cv2.putText(display_front, "SAFE CORRIDOR", (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw Tokens
                for token_data in debug_tokens:
                    if len(token_data) >= 5:
                        ttype, x, y, w, h = token_data[:5]
                        # 根据token类型选择颜色
                        if 'RED' in ttype:
                            color = (0, 0, 255)  # 红色
                        elif 'YELLOW' in ttype:
                            color = (0, 255, 255)  # 黄色
                        else:
                            color = (0, 255, 0)  # 绿色
                        cv2.rectangle(display_front, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(display_front, ttype, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # Draw Action
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