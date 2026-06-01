import cv2
import mediapipe as mp
import math
import threading
import time
from ursina import *
from direct.actor.Actor import Actor
from panda3d.core import Texture as PandaTexture

# ==============================================================================
# 1. KONFIGURASI GLOBAL & UTILITAS
# ==============================================================================
SMOOTH = 0.25
DEADZONE = 0.5
TORSO_GAIN = 1.0
# Nilai alpha untuk filter (0.0 - 1.0). Semakin kecil = semakin halus, tapi sedikit lambat.
SMOOTH_FILTER_ALPHA = 0.15 

def angle_between(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
        
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))
    return math.degrees(math.acos(cos_angle))

def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

# Fungsi pembantu untuk meredam noise (Low-pass Filter / EMA)
def low_pass_filter(current_val, previous_val, alpha):
    if previous_val is None:
        return current_val
    return alpha * current_val + (1 - alpha) * previous_val

# ==============================================================================
# 2. THREAD MEDIA PIPE (MOCAP)
# ==============================================================================
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
        
        # Data Animasi Joints (Hasil Akhir Terfilter)
        self.head_roll = 0.0
        self.l_upper = 0.0
        self.l_lower = 0.0
        self.r_upper = 0.0
        self.r_lower = 0.0
        self.upperchest_roll = 0.0
        
        self.upperchest_center = None
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
                joint_indices = [11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 7, 8]
                if all(lm[i].visibility > 0.7 for i in joint_indices):
                    self.pose_detected = True
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    # Head Vector
                    head_vec = (lm[7].x - lm[8].x, lm[7].y - lm[8].y)

                    # Left Arm Vectors
                    l_upper_vec = (lm[13].x - lm[11].x, lm[13].y - lm[11].y)
                    l_lower_vec = (lm[15].x - lm[13].x, lm[15].y - lm[13].y)

                    # Right Arm Vectors
                    r_upper_vec = (lm[14].x - lm[12].x, lm[14].y - lm[12].y)
                    r_lower_vec = (lm[16].x - lm[14].x, lm[16].y - lm[14].y)

                    # Shoulder-Hip Vectors
                    shoulder_center = ((lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2)
                    hip_center = ((lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2)
                    upperchest_vec = (shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
                    
                    # --- 1. KALKULASI DATA MENTAH (RAW) UNTUK HEAD & TORSO ---
                    raw_head_roll = normalize_angle(math.degrees(math.atan2(head_vec[1], head_vec[0])))
                    upperchest_angle = math.degrees(math.atan2(upperchest_vec[0], -upperchest_vec[1]))
                    
                    if self.upperchest_center is None:
                        self.upperchest_center = upperchest_angle
                        print(f"[CALIBRATION] Pusat tegap torso terkunci di: {self.upperchest_center:.2f}°")
                    
                    relative_angle = normalize_angle(upperchest_angle - self.upperchest_center)
                    raw_upperchest_roll = max(-30, min(30, relative_angle))
                    
                    # ==============================================================================
                    # SOLUSI BARU: ROTASI VEKTOR SEJAJAR BAHU (ANTI-FLIP / ANTI-PUTAR BALIK)
                    # ==============================================================================
                    # Kita putar balik arah vektor tangan menggunakan minus dari relative_angle bahu.
                    # Hal ini membuat koordinat tangan selalu tegak lurus secara virtual terhadap torso.
                    rad = math.radians(-relative_angle)
                    cos_r = math.cos(rad)
                    sin_r = math.sin(rad)

                    # Rotasi Vektor Lengan Kiri
                    l_u_x = l_upper_vec[0] * cos_r - l_upper_vec[1] * sin_r
                    l_u_y = l_upper_vec[0] * sin_r + l_upper_vec[1] * cos_r

                    # Rotasi Vektor Lengan Kanan
                    r_u_x = r_upper_vec[0] * cos_r - r_upper_vec[1] * sin_r
                    r_u_y = r_upper_vec[0] * sin_r + r_upper_vec[1] * cos_r
                    
                    l_upper_angle = math.degrees(math.atan2(l_u_y, l_u_x))
                    raw_l_upper = normalize_angle(-l_upper_angle - 90)
                    raw_l_upper = max(-180.0, min(45.0, raw_l_upper))
                    raw_l_lower = -angle_between(l_upper_vec, l_lower_vec)

                    
                    r_upper_angle = math.degrees(math.atan2(r_u_y, r_u_x))
                    raw_r_upper = normalize_angle(-r_upper_angle - 90)
                    raw_r_upper = max(-45, min(180.0, raw_r_upper))
                    raw_r_lower = -angle_between(r_upper_vec, r_lower_vec)

                    # --- 2. JALANKAN LOW PASS FILTER (EMA) ---
                    alpha = SMOOTH_FILTER_ALPHA
                    self.head_roll       = low_pass_filter(raw_head_roll, self.head_roll, alpha)
                    self.l_upper         = low_pass_filter(raw_l_upper, self.l_upper, alpha)
                    self.l_lower         = low_pass_filter(raw_l_lower, self.l_lower, alpha)
                    self.r_upper         = low_pass_filter(raw_r_upper, self.r_upper, alpha)
                    self.r_lower         = low_pass_filter(raw_r_lower, self.r_lower, alpha)
                    self.upperchest_roll = low_pass_filter(raw_upperchest_roll, self.upperchest_roll, alpha)
                    
                else:
                    self.pose_detected = False

            with self.frame_lock:
                self.current_frame = frame.copy()

            time.sleep(0.016)

        self.cap.release()

# INISIALISASI THREAD
ai_mocap = MocapThread()
ai_mocap.start()

# ==============================================================================
# 3. URSINA ENGINE SETUP
# ==============================================================================
app = Ursina()
window.title = "Simple Motion Capture"
window.borderless = False
window.exit_button.visible = False

# Setup Avatar
avatar_container = Entity(position=(0.4, 1, 0))
karakter = None
try:
    karakter = Actor("karakter.glb")
    karakter.reparentTo(avatar_container)
    karakter.setScale(1.0)
except Exception as e:
    print(f"[ERROR] Gagal memuat Model: {e}")

# Camera Setup
camera.position = (0.4, 1.7, -5)
camera.lookAt(avatar_container)
EditorCamera()

# Setup Visual Webcam Window
panda_tex = PandaTexture()
panda_tex.setup2dTexture(640, 480, PandaTexture.T_unsigned_byte, PandaTexture.F_rgb)
cam_texture = Texture(panda_tex)
Entity(parent=camera.ui, model='quad', texture=cam_texture, position=(-0.65, 0.3), scale=(0.55, 0.4))

l_upper = l_lower = l_hand = r_upper = r_lower = r_hand = c_head = c_upperchest = None

def init_joints():
    global l_upper, l_lower, l_hand, r_upper, r_lower, r_hand, c_head, c_upperchest
    if not karakter: return
    l_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_L_UpperArm")
    l_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_L_LowerArm")
    l_hand = karakter.controlJoint(None, "modelRoot", "J_Bip_L_Hand")
    r_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_R_UpperArm")
    r_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_R_LowerArm")
    r_hand = karakter.controlJoint(None, "modelRoot", "J_Bip_R_Hand")
    c_head = karakter.controlJoint(None, "modelRoot", "J_Bip_Head")
    c_upperchest = karakter.controlJoint(None, "modelRoot", "J_Bip_C_UpperChest")
    print("[OK] Joints berhasil terhubung.")

invoke(init_joints, delay=1.0)

# ==============================================================================
# 4. INTERFACE / UI SLIDER
# ==============================================================================
Text(text="Gunakan Slider untuk Tuning Offset secara Real-Time", position=(-0.2, 0.45), scale=1.2)
slider_smooth      = Slider(min=0.01, max=1.0, default=SMOOTH, text='Smooth Speed', position=(0.3, -0.25), dynamic=True)
slider_hand_target = Slider(min=-180, max=180, default=90, text='Total Rotation Angle', position=(0.3, -0.3), dynamic=True)
slider_hand_axis   = Slider(min=1, max=3, default=3, text='Twist Axis (1:H, 2:P, 3:R)', position=(0.3, -0.35), dynamic=True)

tracking_was_active = False

# ==============================================================================
# 5. LOOP UPDATE UTAMA
# ==============================================================================
def update():
    global tracking_was_active, l_upper, l_lower, l_hand, r_upper, r_lower, r_hand, c_head, c_upperchest
    
    if ai_mocap.current_frame is not None:
        with ai_mocap.frame_lock:
            frame_copy = ai_mocap.current_frame.copy()
            img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            img_correct = cv2.flip(img_rgb, 0) 
            panda_tex.setRamImage(img_correct.tobytes())

    if not karakter:
        return

    current_smooth = slider_smooth.value

    # Logika Sinkronisasi Gerakan Mocap
    if ai_mocap.pose_detected:
        if not tracking_was_active:
            print("Tracking Regained -> Mengambil alih kendali joint...")
            init_joints()
            tracking_was_active = True
            
        if c_head:
            c_head.setP(lerp(c_head.getP(), ai_mocap.head_roll, current_smooth))
            
        if l_upper:
            l_upper.setP(lerp(l_upper.getP(), ai_mocap.l_upper + 90, current_smooth))
        
        if l_lower:
            l_lower.setP(lerp(l_lower.getP(), -ai_mocap.l_lower, current_smooth))
            
        if r_upper:
            # Sumbu model 3D kanan Ursina biasanya berkebalikan arah rotasi p-nya terhadap kiri.
            # Jika tangan kanan bergerak ke arah sebaliknya dari kiri, Anda bisa ubah tanda minus (-) di depan ai_mocap.r_upper
            r_upper.setP(lerp(r_upper.getP(), -ai_mocap.r_upper + 90, current_smooth))
            
        if r_lower:
            r_lower.setP(lerp(r_lower.getP(), -ai_mocap.r_lower, current_smooth))
            
        # Logika Pergelangan Tangan (Hand)
        siku_kiri_menekuk = abs(ai_mocap.l_lower) > 60
        siku_kanan_menekuk = abs(ai_mocap.r_lower) > 60
        axis_pilihan = int(slider_hand_axis.value)
        rotasi_hand = slider_hand_target.value * 0.60
        
        if l_hand:
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
                
        if c_upperchest:
            raw = ai_mocap.upperchest_roll
            if abs(raw) < DEADZONE:
                filtered_angle = 0.0
            else:
                filtered_angle = math.copysign(abs(raw) - DEADZONE, raw)
                filtered_angle *= TORSO_GAIN
            
            c_upperchest.setR(lerp(c_upperchest.getR(), filtered_angle, current_smooth))
    else:
        if tracking_was_active:
            print("Tracking Lost -> Melepaskan kendali joint...")
            karakter.releaseJoint("modelRoot", "J_Bip_L_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_L_LowerArm")
            karakter.releaseJoint("modelRoot", "J_Bip_L_Hand")
            karakter.releaseJoint("modelRoot", "J_Bip_R_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_LowerArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_Hand")
            karakter.releaseJoint("modelRoot", "J_Bip_Head")
            karakter.releaseJoint("modelRoot", "J_Bip_C_UpperChest")
            l_upper = l_lower = l_hand = r_upper = r_lower = r_hand = c_upperchest = None
            tracking_was_active = False

def on_destroy():
    ai_mocap.running = False
    ai_mocap.join(timeout=1)

# Run Aplikasi
print("="*60)
print("SIMPLE MOTION CAPTURE WITH CENTER CALIBRATION RUNNING...")
print("="*60)
app.run()