import cv2
import mediapipe as mp
import math
import threading
import time
from ursina import *
from direct.actor.Actor import Actor
from panda3d.core import Texture as PandaTexture

# ==============================================================================
# KONFIGURASI (UBAH DI SINI)
# ==============================================================================
SMOOTH = 0.25

# Offset Sudut (sesuaikan ini)
L_UPPER_OFFSET = 90
L_LOWER_OFFSET = 0
R_UPPER_OFFSET = 90
R_LOWER_OFFSET = 0

HEAD_OFFSET = 0
SPINE_OFFSET = 0

# =====================
# THREAD MEDIA PIPE
# =====================
class MocapThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        
        self.pose_detected = False
        self.running = True
        self.frame_lock = threading.Lock()
        
        self.l_upper = 0.0
        self.l_lower = 0.0
        self.r_upper = 0.0
        self.r_lower = 0.0
        
        self.head_roll = 0.0
        self.spine_roll = 0.0
        
        self.current_frame = None

    def run(self):
        mp_drawing = mp.solutions.drawing_utils

        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                if all(lm[i].visibility > 0.7 for i in [11,13,15,12,14,16,7,8,]): #17,18,21,22,23,24,
                    self.pose_detected = True
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    l_upper_vec = (lm[13].x - lm[11].x, lm[13].y - lm[11].y)
                    l_lower_vec = (lm[15].x - lm[13].x, lm[15].y - lm[13].y)
                    
                    r_upper_vec = (lm[14].x - lm[12].x, lm[14].y - lm[12].y)
                    r_lower_vec = (lm[16].x - lm[14].x, lm[16].y - lm[14].y)
                    
                    # Lengan Kiri
                    """l_upper_angle = math.degrees(math.atan2(lm[13].y - lm[11].y, lm[13].x - lm[11].x))
                    self.l_upper = normalize_angle(-l_upper_angle - 90)
                    self.l_lower = math.degrees(math.atan2(lm[15].y - lm[13].y, lm[15].x - lm[13].x)) - 90"""
                    
                    l_upper_angle = math.degrees(math.atan2(l_upper_vec[1], l_upper_vec[0]))
                    self.l_upper = normalize_angle(-l_upper_angle - 90)
                    elbow_angle = angle_between(l_upper_vec, l_lower_vec)
                    self.l_lower = -(elbow_angle)
                    
                    # Lengan Kanan
                    """r_upper_raw = math.degrees(math.atan2(lm[14].y - lm[12].y, lm[14].x - lm[12].x))
                    self.r_upper = normalize_angle(-r_upper_raw - 90)
                    self.r_lower = math.degrees(math.atan2(lm[16].y - lm[14].y, lm[16].x - lm[14].x)) - 90"""

                    r_upper_angle = math.degrees(math.atan2(r_upper_vec[1], r_upper_vec[0]))
                    self.r_upper = normalize_angle(-r_upper_angle - 90)
                    elbow_angle = angle_between(r_upper_vec, r_lower_vec)
                    self.r_lower = -(elbow_angle)

                    # Tangan Terbalik
                    self.r_hand_flipped = lm[22].x > lm[18].x
                    self.l_hand_flipped = lm[21].x < lm[17].x
                    
                    # Kepala
                    self.head_roll = normalize_angle(math.degrees(math.atan2(lm[7].y - lm[8].y, lm[7].x - lm[8].x)))
                    
                    # Spine
                    shoulder_angle = math.degrees(math.atan2(
                        lm[12].y - lm[11].y,
                        lm[12].x - lm[11].x
                    ))

                    target_spine = normalize_angle(shoulder_angle)
                    self.spine_roll = lerp(self.spine_roll,target_spine,0.1)

                else:
                    self.pose_detected = False

            with self.frame_lock:
                self.current_frame = frame.copy()

            time.sleep(0.016)

        self.cap.release()


# ================
# URSINA SETUP
# ================
app = Ursina()
window.title = "Simple Motion Capture"
window.borderless = False
window.exit_button.visible = False

# Karakter
avatar_container = Entity(position=(0.4, 1, 0))
karakter = None
try:
    karakter = Actor("karakter.glb")
    karakter.reparentTo(avatar_container)
    karakter.setScale(1.0)
except Exception as e:
    print(f"[ERROR] Model: {e}")

# Webcam
panda_tex = PandaTexture()
panda_tex.setup2dTexture(640, 480, PandaTexture.T_unsigned_byte, PandaTexture.F_rgb)
cam_texture = Texture(panda_tex)
Entity(parent=camera.ui, 
       model='quad',
       texture=cam_texture,
       position=(-0.65, 0.3),
       scale=(0.55, 0.4)
       )

# Joints
l_upper = l_lower = l_hand = r_upper = r_lower = r_hand = c_head = c_spine = None

def init_joints():
    global l_upper, l_lower, r_upper, r_lower, c_head, c_spine
    if not karakter: return
    l_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_L_UpperArm")
    l_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_L_LowerArm")
    l_hand = karakter.controlJoint(None, "modelRoot", "J_Bip_L_Hand")
    r_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_R_UpperArm")
    r_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_R_LowerArm")
    r_hand = karakter.controlJoint(None, "modelRoot", "J_Bip_R_Hand")
    c_head = karakter.controlJoint(None, "modelRoot", "J_Bip_Head")
    c_spine = karakter.controlJoint(None, "modelRoot", "J_Bip_Spine")
    print("[OK] Joints terhubung")

invoke(init_joints, delay=1.0)

camera.position = (0.4, 1.7, -5)
camera.lookAt(avatar_container)
EditorCamera()

# ==============================================================================
# UPDATE
# ==============================================================================

# ==============================================================================
# UI SLIDER UNTUK TUNING REAL-TIME
# ==============================================================================
# Buat text instruksi di layar
Text(text="Gunakan Slider untuk Tuning Offset secara Real-Time", position=(-0.2, 0.45), scale=1.2)

slider_smooth  = Slider(min=0.01, max=1.0, default=SMOOTH, text='Smooth Speed', position=(0.1, 0.1), dynamic=True)

# Kontrol Rotasi Spesifik Lengan & Pergelangan Tangan
slider_hand_target = Slider(min=-180, max=180, default=90, text='Total Rotation Angle', position=(0.3, -0.3), dynamic=True)
slider_hand_axis   = Slider(min=1, max=3, default=3, text='Twist Axis (1:H, 2:P, 3:R)', position=(0.3, -0.35), dynamic=True)
slider_arm_ratio   = Slider(min=0.0, max=1.0, default=0.4, text='LowerArm Weight Ratio', position=(0.3, -0.4), dynamic=True)
slider_invert      = Slider(min=-1, max=1, default=1, text='Invert Direction (-1 atau 1)', position=(0.3, -0.45), dynamic=True)

frame_count = 0
tracking_was_active = False

frame_count = 0
tracking_was_active = False

def angle_between(v1, v2):
    dot = v1[0]*v2[0] + v1[1]*v2[1]

    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if mag1 == 0 or mag2 == 0:
        return 0

    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))

    return math.degrees(math.acos(cos_angle))

def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def update():
    global frame_count, tracking_was_active, l_upper, l_lower, r_upper, r_lower, l_hand, r_hand, c_head, c_spine
    frame_count += 1

    # Update Webcam Visual
    if ai_mocap.current_frame is not None:
        with ai_mocap.frame_lock:
            frame_copy = ai_mocap.current_frame.copy()
            img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            img_correct = cv2.flip(img_rgb, 0) 
            panda_tex.setRamImage(img_correct.tobytes())

    if not karakter:
        return

    # Ambil nilai kelembutan gerakan langsung dari slider
    current_smooth = slider_smooth.value

    # --- LOGIKA SINKRONISASI ---
    if ai_mocap.pose_detected:
        if not tracking_was_active:
            print("Tracking Regained -> Mengambil alih kendali joint kembali...")
            init_joints()
            tracking_was_active = True
            
        if c_head:
            target = ai_mocap.head_roll
            c_head.setR(lerp(c_head.getR(), target, current_smooth))
            
        if l_upper:
            target = ai_mocap.l_upper + 90
            l_upper.setP(lerp(l_upper.getP(), target, current_smooth))
        
        if l_lower:
            target = -ai_mocap.l_lower + 0
            l_lower.setP(lerp(l_lower.getP(), target, current_smooth))
            
        if r_upper:
            target = -ai_mocap.r_upper + 90
            r_upper.setP(lerp(r_upper.getP(), target, current_smooth))
            
        if r_lower:
            target = -ai_mocap.r_lower + 0
            r_lower.setP(lerp(r_lower.getP(), target, current_smooth))
            
        siku_kiri_menekuk = abs(ai_mocap.l_lower) > 60
        siku_kanan_menekuk = abs(ai_mocap.r_lower) > 60
        
        hand_target_angle = slider_hand_target.value
        axis_pilihan = int(slider_hand_axis.value)
        
        rotasi_hand      = hand_target_angle * 0.60
        
        if l_hand:
            # Telapak tangan merespon murni sumbu pilihan dari slider
            target_hand_l = rotasi_hand if siku_kiri_menekuk else 0
            if axis_pilihan == 1:
                l_hand.setH(lerp(l_hand.getH(), target_hand_l, current_smooth))
            elif axis_pilihan == 2:
                l_hand.setP(lerp(l_hand.getP(), target_hand_l, current_smooth))
            elif axis_pilihan == 3:
                l_hand.setR(lerp(l_hand.getR(), target_hand_l, current_smooth))

        if r_hand:
            target_hand_r = rotasi_hand if siku_kanan_menekuk else 0
            if axis_pilihan == 1:
                r_hand.setH(lerp(r_hand.getH(), target_hand_r, current_smooth))
            elif axis_pilihan == 2:
                r_hand.setP(lerp(r_hand.getP(), target_hand_r, current_smooth))
            elif axis_pilihan == 3:
                r_hand.setR(lerp(r_hand.getR(), target_hand_r, current_smooth))
                
        if c_spine:
            target = ai_mocap.spine_roll
            c_spine.setR(lerp(c_spine.getR(), target, current_smooth))
    else:
        if tracking_was_active:
            print("Tracking Lost -> Melepaskan kendali joint...")
            karakter.releaseJoint("modelRoot", "J_Bip_L_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_L_LowerArm")
            karakter.releaseJoint("modelRoot", "J_Bip_L_Hand")
            karakter.releaseJoint("modelRoot", "J_Bip_R_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_LowerArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_Hand")
            l_upper = l_lower = l_hand = r_upper = r_lower = r_hand = None
            tracking_was_active = False

def on_destroy():
    ai_mocap.running = False
    ai_mocap.join(timeout=1)


# ==============================================================================
ai_mocap = MocapThread()
ai_mocap.start()

print("="*60)
print("SIMPLE MOTION CAPTURE")
print("Gerakkan tangan kamu di depan webcam")
print("Ubah OFFSET di atas jika gerakan tidak cocok")
print("="*60)

app.run()