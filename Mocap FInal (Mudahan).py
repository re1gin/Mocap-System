import cv2
import mediapipe as mp
import math
import threading
import time
from ursina import *
from direct.actor.Actor import Actor
from panda3d.core import Texture as PandaTexture

SMOOTH = 0.25
DEADZONE = 0.2
TORSO_GAIN = 3.0
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

def signed_angle_between(v1, v2):
    raw_ang = angle_between(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_product < 0:
        return -raw_ang
    return raw_ang

def normalize_angle(angle):
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def low_pass_filter(current_val, previous_val, alpha):
    if previous_val is None:
        return current_val
    return alpha * current_val + (1 - alpha) * previous_val

class MocapThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        self.pose_detected = False
        self.running = True
        self.frame_lock = threading.Lock()
        
        self.head_roll = 0.0
        self.l_upper = 0.0
        self.l_lower = 0.0
        self.r_upper = 0.0
        self.r_lower = 0.0
        self.spine_roll = 0.0
        
        self.spine_center = None
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
                    spine_vec = (shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
                    
                    # --- 1. KALKULASI DATA MENTAH (RAW) UNTUK HEAD & TORSO ---
                    raw_head_roll = normalize_angle(math.degrees(math.atan2(head_vec[1], head_vec[0])))
                    spine_angle = math.degrees(math.atan2(spine_vec[0], -spine_vec[1]))
                    
                    if self.spine_center is None:
                        self.spine_center = spine_angle
                        print(f"[CALIBRATION] Pusat tegap torso terkunci di: {self.spine_center:.2f}°")
                    
                    relative_angle = normalize_angle(spine_angle - self.spine_center)
                    raw_spine_roll = max(-45, min(45, relative_angle))
                    
                    # --- 2. ROTASI VEKTOR SEJAJAR BAHU ---
                    rad = math.radians(-relative_angle)
                    cos_r = math.cos(rad)
                    sin_r = math.sin(rad)

                    # Rotasi Vektor Lengan Kiri
                    l_u_x = l_upper_vec[0] * cos_r - l_upper_vec[1] * sin_r
                    l_u_y = l_upper_vec[0] * sin_r + l_upper_vec[1] * cos_r

                    # Rotasi Vektor Lengan Kanan
                    r_u_x = r_upper_vec[0] * cos_r - r_upper_vec[1] * sin_r
                    r_u_y = r_upper_vec[0] * sin_r + r_upper_vec[1] * cos_r
                    
                    # --- PERUBAHAN: HAPUS LIMITASI CLAMP AGAR BISA BEBAS BERPUTAR (360°) ---
                    l_upper_angle = math.degrees(math.atan2(l_u_y, l_u_x))
                    raw_l_upper = normalize_angle(-l_upper_angle - 90)
                    
                    # Gunakan signed_angle_between agar siku tahu kapan harus melipat ke dalam/luar
                    raw_l_lower = signed_angle_between(l_upper_vec, l_lower_vec)

                    r_upper_angle = math.degrees(math.atan2(r_u_y, r_u_x))
                    raw_r_upper = normalize_angle(-r_upper_angle - 90)
                    
                    # Menggunakan signed_angle_between untuk tangan kanan
                    raw_r_lower = signed_angle_between(r_upper_vec, r_lower_vec)

                    alpha = SMOOTH_FILTER_ALPHA
                    self.head_roll       = low_pass_filter(raw_head_roll, self.head_roll, alpha)
                    self.l_upper         = low_pass_filter(raw_l_upper, self.l_upper, alpha)
                    self.l_lower         = low_pass_filter(raw_l_lower, self.l_lower, alpha)
                    self.r_upper         = low_pass_filter(raw_r_upper, self.r_upper, alpha)
                    self.r_lower         = low_pass_filter(raw_r_lower, self.r_lower, alpha)
                    self.spine_roll = low_pass_filter(raw_spine_roll, self.spine_roll, alpha)
                    
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
window.title = "Mocap Viewport Studio"
window.borderless = False
window.exit_button.visible = False

# PERUBAHAN: Set background gelap & tambahkan atmosfer kabut software 3D
window.color = color.dark_gray
scene.fog_color = color.dark_gray
scene.fog_density = (4, 18)

Entity(model='line', scale=(6,1,1), color=color.red, position=(0.4, 1, 0))
Entity(model='line', scale=(6,1,1), color=color.blue, rotation_y=90, position=(0.4, 1, 0))

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
camera.position = (0.4, 1.7, -6)
camera.lookAt(avatar_container)
EditorCamera()

# Setup Visual Webcam Window
panda_tex = PandaTexture()
panda_tex.setup2dTexture(640, 480, PandaTexture.T_unsigned_byte, PandaTexture.F_rgb)
cam_texture = Texture(panda_tex)
Entity(parent=camera.ui, model='quad', texture=cam_texture, position=(-0.65, 0.3), scale=(0.55, 0.4))

l_upper = l_lower = l_hand = r_upper = r_lower = r_hand = c_head = c_spine = None

def init_joints():
    global l_upper, l_lower, l_hand, r_upper, r_lower, r_hand, c_head, c_spine
    if not karakter: return
    l_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_L_UpperArm")
    l_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_L_LowerArm")
    l_hand = karakter.controlJoint(None, "modelRoot", "J_Bip_L_Hand")
    r_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_R_UpperArm")
    r_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_R_LowerArm")
    r_hand = karakter.controlJoint(None, "modelRoot", "J_Bip_R_Hand")
    c_head = karakter.controlJoint(None, "modelRoot", "J_Bip_C_Head")
    c_spine = karakter.controlJoint(None, "modelRoot", "J_Bip_C_Spine")
    print("[OK] Joints berhasil terhubung.")

invoke(init_joints, delay=1.0)

# ==============================================================================
# 4. INTERFACE / UI SLIDER
# ==============================================================================

# PERUBAHAN: Hanya menyimpan Slider Smooth Speed tepat di bawah tampilan kamera webcam
Text(text="Gunakan Slider untuk Tuning Offset secara Real-Time", position=(-0.85, -0.15), scale=1.0)
slider_smooth = Slider(min=0.01, max=1.0, default=SMOOTH, text='Smooth Speed', position=(-0.65, -0.25), dynamic=True)
tracking_was_active = False

# ==============================================================================
# 5. LOOP UPDATE UTAMA
# ==============================================================================
def update():
    global tracking_was_active, l_upper, l_lower, l_hand, r_upper, r_lower, r_hand, c_head, c_spine
    
    if ai_mocap.current_frame is not None:
        with ai_mocap.frame_lock:
            frame_copy = ai_mocap.current_frame.copy()
            
            img_correct = cv2.flip(frame_copy, 0) 
            panda_tex.setRamImage(img_correct.tobytes())

    if not karakter:
        return

    current_smooth = slider_smooth.value
    s = current_smooth

    if ai_mocap.pose_detected:
        if not tracking_was_active:
            print("Tracking Regained -> Mengambil alih kendali joint...")
            init_joints()
            tracking_was_active = True
            
        if c_head:
            target_head = ai_mocap.head_roll * 0.85          
            c_head.setR(lerp(c_head.getR(), target_head, s)) 
            c_head.setP(lerp(c_head.getP(), ai_mocap.head_roll * 0.3, s*0.6))
            
        if c_spine:
            raw = ai_mocap.spine_roll
            if abs(raw) < DEADZONE:
                filtered = 0
            else:
                filtered = math.copysign(abs(raw) - DEADZONE, raw) * TORSO_GAIN * 0.7  
                
            c_spine.setR(lerp(c_spine.getR(), filtered, s))           
            c_spine.setP(lerp(c_spine.getP(), filtered * 0.25, s*0.5))
                
        # ================== Lengan Kiri ==================
        if l_upper:
            l_upper.setP(lerp(l_upper.getP(), ai_mocap.l_upper + 90, s))   
            l_upper.setR(lerp(l_upper.getR(), ai_mocap.l_upper * -0.25, s*0.4))  

        if l_lower:
            # Mengikuti arah tekukan riil (bisa positif/negatif)
            l_lower.setP(lerp(l_lower.getP(), -ai_mocap.l_lower, s))
            
        # ================== Lengan Kanan ==================
        if r_upper:
            r_upper.setP(lerp(r_upper.getP(), -ai_mocap.r_upper + 90, s))
            r_upper.setR(lerp(r_upper.getR(), ai_mocap.r_upper * -0.25, s*0.4))

        if r_lower:
            # Mengikuti arah tekukan riil (bisa positif/negatif)
            r_lower.setP(lerp(r_lower.getP(), ai_mocap.r_lower, s))
            
        # ================== Tangan (Twist) ==================
        if l_hand or r_hand:
            siku_kiri_menekuk = abs(ai_mocap.l_lower) > 35
            siku_kanan_menekuk = abs(ai_mocap.r_lower) > 35 
            
            rotasi = 90 * 0.65 * 0.65
            
            if l_hand:
                target = rotasi if siku_kiri_menekuk else 15
                l_hand.setR(lerp(l_hand.getR(), target, s))

            if r_hand:
                target = -rotasi if siku_kanan_menekuk else 15
                r_hand.setR(lerp(r_hand.getR(), target, s))
                
    else:
        if tracking_was_active:
            print("Tracking Lost -> Melepaskan kendali joint...")
            karakter.releaseJoint("modelRoot", "J_Bip_L_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_L_LowerArm")
            karakter.releaseJoint("modelRoot", "J_Bip_L_Hand")
            karakter.releaseJoint("modelRoot", "J_Bip_R_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_LowerArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_Hand")
            karakter.releaseJoint("modelRoot", "J_Bip_C_Head")
            karakter.releaseJoint("modelRoot", "J_Bip_C_Spine")
            l_upper = l_lower = l_hand = r_upper = r_lower = r_hand = c_head = c_spine = None
            tracking_was_active = False

def on_destroy():
    ai_mocap.running = False
    ai_mocap.join(timeout=1)

print("="*60)
print("SIMPLE MOTION CAPTURE WITH UNLOCKED ARMS RUNNING...")
print("="*60)
app.run()