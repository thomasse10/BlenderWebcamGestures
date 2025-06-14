import cv2
import mediapipe as mp
import socket
import struct
import math

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
BLENDER_IP = '127.0.0.1'
BLENDER_PORT = 5006

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

def get_hand_rotation(landmarks):
    wrist = landmarks.landmark[0]
    mcp = landmarks.landmark[9]  # middle finger MCP

    dx = mcp.x - wrist.x
    dy = mcp.y - wrist.y
    dz = mcp.z - wrist.z

    yaw = math.atan2(dy, dx)       # horizontal rotation
    pitch = math.atan2(dz, dy)     # vertical rotation
    roll = 0                       # no roll for now

    return pitch, yaw, roll

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        pitch, yaw, roll = get_hand_rotation(hand)
        data = struct.pack('fff', pitch, yaw, roll)
        sock.sendto(data, (BLENDER_IP, BLENDER_PORT))

    cv2.imshow("Hand to Blender", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()