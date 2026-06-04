import socket
import cv2
import numpy as np
import time
import os
import select

# Configuration
CAMERA_HOST = '127.0.0.1'
FRONT_CAMERA_PORT = 8080
SAVE_DIR = 'dataset/images'
SAVE_INTERVAL = 1.0 # Save a frame every 1 second

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

print(f"Connecting to Front Camera at {CAMERA_HOST}:{FRONT_CAMERA_PORT}...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

while True:
    try:
        sock.connect((CAMERA_HOST, FRONT_CAMERA_PORT))
        print("Connected successfully!")
        break
    except Exception:
        time.sleep(1)

saved_count = 0
last_save_time = time.time()

try:
    while True:
        # Check if data is available
        readable, _, _ = select.select([sock], [], [], 0.1)
        if not readable:
            continue
            
        # Read frame length
        length_bytes = sock.recv(4)
        if not length_bytes:
            break
            
        image_length = int.from_bytes(length_bytes, 'little')
        
        # Read image bytes
        received_bytes = b''
        while len(received_bytes) < image_length:
            packet = sock.recv(image_length - len(received_bytes))
            if not packet:
                break
            received_bytes += packet
            
        if len(received_bytes) == image_length:
            # Decode image
            np_arr = np.frombuffer(received_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Display the frame to know what's happening
                cv2.imshow("Data Collection (Press 'q' to stop)", frame)
                
                # Save frame periodically
                current_time = time.time()
                if current_time - last_save_time >= SAVE_INTERVAL:
                    filename = os.path.join(SAVE_DIR, f"frame_{int(current_time)}.jpg")
                    cv2.imwrite(filename, frame)
                    saved_count += 1
                    print(f"Saved {filename} (Total: {saved_count})")
                    last_save_time = current_time
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
except KeyboardInterrupt:
    pass
finally:
    sock.close()
    cv2.destroyAllWindows()
    print(f"Data collection stopped. Total images saved: {saved_count}")
