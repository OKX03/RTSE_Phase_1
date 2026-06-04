import socket
import threading
import struct
import cv2
import numpy as np
import time
import select
import ctypes
import os

try:
    import keyboard
    KEYBOARD_ENABLED = True
except ImportError:
    print("WARNING: keyboard library not installed. Some events may require manual key presses.")
    KEYBOARD_ENABLED = False

try:
    from ultralytics import YOLO
except ImportError:
    print("WARNING: ultralytics not installed. Please pip install ultralytics")


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
    'yolo_tokens': [],
    'light_on': False,
    'police_mode': False,
    'net_lane_position': 0
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
# YOLO Inference Task (Dual-Track Concurrency)
# ---------------------------------------------------------
# ---------------------------------------------------------
# YOLO Inference Task (Dual-Track Concurrency)
# ---------------------------------------------------------
def yolo_inference_task():
    """
    Runs YOLO at ~30 FPS in background, completely immune to color/lighting changes.
    """
    import os
    # ⚠️ 确保这里是你最新的 V2 模型路径
    model_path = r"C:\Users\ohxua\runs\detect\train-2\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found at {model_path}! Falling back to yolov8n.pt")
        model_path = 'yolov8n.pt'
    else:
        print("✅ V2 Model successfully loaded. Unleashing the beast...")
        
    model = YOLO(model_path)
    
    while is_running:
        start_time = time.time()
        
        with data_lock:
            frame = shared_data['latest_front_frame']
            
        if frame is not None:
            # 获取原图的真实分辨率
            h_orig, w_orig = frame.shape[:2]
            
            # 🚨 核心修复：直接把完整的 frame 喂给 YOLO！
            # 这样输入图像就和它训练时的照片完全一致了。
            results = model(frame, verbose=False)[0]
            
            tokens = []
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > 0.4:  # 置信度阈值
                    cls_id = int(box.cls[0])
                    # 拿到全局坐标 (基于原始分辨率 w_orig x h_orig)
                    x1_full, y1_full, x2_full, y2_full = box.xyxy[0].cpu().numpy()
                    
                    # ⚠️ 坐标精确转换：
                    # 无论游戏原图是 800x600 还是 1920x1080，都必须按比例缩放到 320x240！
                    scale_x = 320.0 / w_orig
                    scale_y = 240.0 / h_orig
                    
                    x_small = x1_full * scale_x
                    y_small = y1_full * scale_y
                    w_small = (x2_full - x1_full) * scale_x
                    h_small = (y2_full - y1_full) * scale_y
                    
                    # 2. 减去顶部裁剪掉的 100 像素偏移量
                    ROI_START_Y = 100
                    y_roi = y_small - ROI_START_Y
                    
                    ttype = 'GREEN' if cls_id == 1 else 'DANGER'
                    
                    # 只保留地平线 (y_small > 80) 以下的物体
                    if y_small > 80:
                        tokens.append((ttype, x_small, y_roi, w_small, h_small))
            
            with data_lock:
                shared_data['yolo_tokens'] = tokens
                
        elapsed = time.time() - start_time
        time.sleep(max(0.01, 0.03 - elapsed))
# ---------------------------------------------------------
def processing_task():
    """ 
    High-Speed Decision Engine (200 FPS)
    Consumes YOLO tokens seamlessly.
    """
    with data_lock: 
        front_frame = shared_data['latest_front_frame']
        yolo_tokens = shared_data['yolo_tokens'].copy()
        
    target_steer = 0.0
    debug_text = ""
    detected_tokens = []

    if front_frame is not None:
        ROI_START_Y = 100
        
        # --- 1. OpenCV 极速视觉流 (0 延迟) ---
        small_frame = cv2.resize(front_frame, (320, 240))
        roi_front = small_frame[ROI_START_Y:220, 0:320]
        roi_hsv = cv2.cvtColor(roi_front, cv2.COLOR_BGR2HSV)
        
        # 用极其宽松的阈值把所有红黄绿白的东西抠出来！反正 YOLO 负责做语义鉴别，OpenCV 哪怕抠出白云和路标也无所谓。
        mask_green = cv2.inRange(roi_hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        mask_red1 = cv2.inRange(roi_hsv, np.array([0, 70, 40]), np.array([10, 255, 255]))
        mask_red2 = cv2.inRange(roi_hsv, np.array([170, 70, 40]), np.array([180, 255, 255]))
        mask_yellow = cv2.inRange(roi_hsv, np.array([15, 70, 70]), np.array([35, 255, 255]))
        mask_white = cv2.inRange(roi_hsv, np.array([0, 0, 150]), np.array([180, 50, 255])) # 捕获无色金币
        
        mask_all = mask_green | mask_red1 | mask_red2 | mask_yellow | mask_white
        contours_all, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_blobs = []
        for c in contours_all:
            if cv2.contourArea(c) > 3:
                bx, by, bw, bh = cv2.boundingRect(c)
                current_blobs.append((bx, by, bw, bh))
        
        # 动态透视区域计算 (Overlap 基于宽度的碰撞体积计算)
        def get_occupied_lanes(x, y, w, h):
            actual_y = y + h/2 + ROI_START_Y
            dist_to_horizon = actual_y - 80 
            
            if dist_to_horizon <= 0: return []
            
            # 路肩边界
            margin_width = dist_to_horizon * 0.857
            margin_left = 160 - margin_width
            margin_right = 160 + margin_width
            
            # 完全忽略路肩上的物体
            cx = x + w/2
            if cx < margin_left or cx > margin_right:
                return []
            
            # 安全中心走廊
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

        dangers_in_lane = {-1: False, 0: False, 1: False}
        best_green_lane = None
        max_green_area = 0

        # --- 2. 传感器融合：YOLO 语义 (延迟 100ms) + OpenCV 极速轮廓 (延迟 0ms) ---
        for (ttype, yx, yy, yw, yh) in yolo_tokens:
            best_blob = None
            min_dist = 2500 # 允许的最大漂移距离的平方 (约 50 像素)
            
            yolo_cx = yx + yw/2
            yolo_cy = yy + yh/2
            
            for (bx, by, bw, bh) in current_blobs:
                blob_cx = bx + bw/2
                blob_cy = by + bh/2
                
                dist = (yolo_cx - blob_cx)**2 + (yolo_cy - blob_cy)**2
                if dist < min_dist:
                    min_dist = dist
                    best_blob = (bx, by, bw, bh)
                    
            if best_blob is not None:
                # 融合成功！使用最新的无延迟坐标
                x, y, w, h = best_blob
            else:
                # 融合失败，降级使用带延迟的 YOLO 坐标
                x, y, w, h = yx, yy, yw, yh
                
            area = w * h
            lanes = get_occupied_lanes(x, y, w, h)
            
            if ttype == 'DANGER':
                for lane in lanes:
                    dangers_in_lane[lane] = True
                if lanes:
                    detected_tokens.append(('DANGER', int(x*2), int((y+ROI_START_Y)*2), int(w*2), int(h*2)))
            
            elif ttype == 'GREEN':
                if lanes:
                    detected_tokens.append(('GREEN', int(x*2), int((y+ROI_START_Y)*2), int(w*2), int(h*2)))
                    if 0 in lanes: 
                        best_green_lane = 0
                    elif area > max_green_area and best_green_lane != 0:
                        max_green_area = area
                        best_green_lane = lanes[0]

        # --- 核心决策逻辑 ---
        if dangers_in_lane[0]:
            if not dangers_in_lane[1]: 
                target_steer = 1.0  
                debug_text = "EVADE RIGHT >>"
            elif not dangers_in_lane[-1]: 
                target_steer = -1.0 
                debug_text = "<< EVADE LEFT"
            else:
                target_steer = 1.0  
                debug_text = "TRAPPED!"
        else:
            if best_green_lane is not None:
                if best_green_lane == -1 and not dangers_in_lane[-1]:
                    target_steer = -1.0
                    debug_text = "SEEK GREEN LEFT"
                elif best_green_lane == 1 and not dangers_in_lane[1]:
                    target_steer = 1.0
                    debug_text = "SEEK GREEN RIGHT"
                else:
                    target_steer = 0.0 
                    debug_text = "STEADY"
            else:
                # 检查是否需要回中 (Auto-Center)
                current_lane = shared_data.get('net_lane_position', 0)
                if current_lane < 0:
                    target_steer = 1.0
                    debug_text = "AUTO CENTER >>"
                elif current_lane > 0:
                    target_steer = -1.0
                    debug_text = "<< AUTO CENTER"
                else:
                    target_steer = 0.0
                    debug_text = "CRUISING"

        with data_lock:
            shared_data['steering_input'] = target_steer
            shared_data['debug_tokens'] = detected_tokens
            shared_data['debug_info'] = debug_text

def send_controls_task():
    global control_conn, tap_state, tap_timer, active_steering_value
    if control_conn is None: return
    
    with data_lock:
        desired_steer = shared_data['steering_input']
        accel_input = shared_data['acceleration_input']

    # Implement "Tap" steering as required by game rules
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
    
    # Priority: Camera > Control > Processing (Rate Monotonic)
    t_front = RTTask("ReadFrontCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_front_camera_task)
    t_back = RTTask("ReadBackCamera", period=0.005, priority=TaskPriority.HIGH, execute_func=read_back_camera_task)
    t_ctrl = RTTask("SendControls", period=0.005, priority=TaskPriority.HIGH, execute_func=send_controls_task)
    t_proc = RTTask("Processing", period=0.005, priority=TaskPriority.MEDIUM, execute_func=processing_task)
    
    t_yolo = threading.Thread(target=yolo_inference_task, daemon=True)
    
    t_front.start()
    t_back.start()
    t_ctrl.start()
    t_proc.start()
    t_yolo.start()
    
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
                
                # Draw ROI bounds
                cv2.line(display_front, (0, 200), (640, 200), (255, 0, 0), 2)
                cv2.line(display_front, (0, 440), (640, 440), (255, 0, 0), 2)
                cv2.putText(display_front, "ROI BOUNDARY", (10, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Draw Safe Corridor (White Lines)
                cv2.line(display_front, (320 - int(20*0.22*2), 200), (320 - int(160*0.22*2), 480), (255, 255, 255), 2)
                cv2.line(display_front, (320 + int(20*0.22*2), 200), (320 + int(160*0.22*2), 480), (255, 255, 255), 2)
                cv2.putText(display_front, "SAFE CORRIDOR", (330, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Draw Tokens
                for (ttype, x, y, w, h) in debug_tokens:
                    color = (0, 0, 255) if ttype == 'DANGER' else (0, 255, 0)
                    cv2.rectangle(display_front, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(display_front, ttype, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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
                # Add pause overlay
                pause_frame = last_display_frame.copy()
                cv2.putText(pause_frame, "PAUSED (Press 'p' to resume)", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.imshow("Fast OpenCV Drive", pause_frame)
            elif last_display_frame is not None:
                cv2.imshow("Fast OpenCV Drive", last_display_frame)
                
    except KeyboardInterrupt: is_running = False

    t_front.join(); t_back.join(); t_proc.join(); t_ctrl.join()
    cv2.destroyAllWindows()