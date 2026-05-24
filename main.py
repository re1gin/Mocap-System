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

# ==============================================================================
# THREAD MEDIA PIPE
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
        
        self.l_upper = 0.0
        self.l_lower = 0.0
        self.r_upper = 0.0
        self.r_lower = 0.0
        
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
                if all(lm[i].visibility > 0.7 for i in [11,13,15,12,14,16]):
                    self.pose_detected = True
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    # Perhitungan Sudut Sederhana
                    
                    # Lengan Kiri
                    l_upper_angle = math.degrees(math.atan2(lm[13].y - lm[11].y, lm[13].x - lm[11].x))
                    self.l_upper = normalize_angle(-l_upper_angle - 90)
                    
                    self.l_lower = math.degrees(math.atan2(lm[15].y - lm[13].y, lm[15].x - lm[13].x)) - 90

                    # Lengan Kanan

                    r_upper_raw = math.degrees(math.atan2(lm[14].y - lm[12].y, lm[14].x - lm[12].x))
                    self.r_upper = normalize_angle(-r_upper_raw - 90)
                    self.r_lower = math.degrees(math.atan2(lm[16].y - lm[14].y, lm[16].x - lm[14].x)) - 90
                    
                    print(self.r_upper)
                    
                    # Simpan data untuk debug
                    self.debug_data = {
                        'r_shoulder': (lm[12].x, lm[12].y),
                        'r_elbow':    (lm[14].x, lm[14].y),
                        'r_wrist':    (lm[16].x, lm[16].y),
                        'r_upper_raw': r_upper_raw,
                        'r_upper_final': self.r_upper
                    }

                else:
                    self.pose_detected = False

            with self.frame_lock:
                self.current_frame = frame.copy()

            time.sleep(0.001)

        self.cap.release()


# ==============================================================================
# URSINA SETUP
# ==============================================================================
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
Entity(parent=camera.ui, model='quad', texture=cam_texture, position=(-0.65, 0.3), scale=(0.55, 0.4))

# Joints
l_upper = l_lower = r_upper = r_lower = None

def init_joints():
    global l_upper, l_lower, r_upper, r_lower
    if not karakter: return
    l_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_L_UpperArm")
    l_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_L_LowerArm")
    r_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_R_UpperArm")
    r_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_R_LowerArm")
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

slider_l_upper = Slider(min=-180, max=180, default=L_UPPER_OFFSET, text='L Upper Offset', position=(0.2, -0.2), dynamic=True)
slider_l_lower = Slider(min=-180, max=180, default=L_LOWER_OFFSET, text='L Lower Offset', position=(0.2, -0.25), dynamic=True)
slider_r_upper = Slider(min=-180, max=180, default=R_UPPER_OFFSET, text='R Upper Offset', position=(0.2, -0.3), dynamic=True)
slider_r_lower = Slider(min=-180, max=180, default=R_LOWER_OFFSET, text='R Lower Offset', position=(0.2, -0.35), dynamic=True)
slider_smooth  = Slider(min=0.01, max=1.0, default=SMOOTH, text='Smooth Speed', position=(0.2, -0.4), dynamic=True)

frame_count = 0
tracking_was_active = False

frame_count = 0
tracking_was_active = False

def normalize_angle(angle):
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

def update():
    global frame_count, tracking_was_active, l_upper, l_lower, r_upper, r_lower
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

        # Ambil nilai target secara dinamis dari posisi SLIDER saat ini
        if l_upper:
            target = ai_mocap.l_upper + slider_l_upper.value
            l_upper.setP(lerp(l_upper.getP(), target, current_smooth))

        if l_lower:
            target = -ai_mocap.l_lower + slider_l_lower.value
            l_lower.setR(lerp(l_lower.getR(), target, current_smooth))

        if r_upper:
            target = -ai_mocap.r_upper + slider_r_upper.value
            r_upper.setP(lerp(r_upper.getP(), target, current_smooth))

        if r_lower:
            target = -ai_mocap.r_lower + slider_r_lower.value
            r_lower.setR(lerp(r_lower.getR(), target, current_smooth))

    else:
        if tracking_was_active:
            print("Tracking Lost -> Melepaskan kendali joint...")
            karakter.releaseJoint("modelRoot", "J_Bip_L_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_L_LowerArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_UpperArm")
            karakter.releaseJoint("modelRoot", "J_Bip_R_LowerArm")
            l_upper = l_lower = r_upper = r_lower = None
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