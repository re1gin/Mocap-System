import cv2
import mediapipe as mp
import math
import threading
import time
from ursina import *
from direct.actor.Actor import Actor
from panda3d.core import Texture as PandaTexture

# ==============================================================================
# 1. THREAD AI TRACKING
# ==============================================================================
class MocapAIThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            model_complexity=1
        )
        
        self.pose_detected = False
        self.running = True
        self.frame_lock = threading.Lock()
        
        # Sudut
        self.angle_l_upper = 0.0
        self.angle_l_lower = 0.0
        self.angle_r_upper = 0.0
        self.angle_r_lower = 0.0
        
        self.current_frame = None

    def run(self):
        print("[INFO] Mocap Thread Started")
        mp_drawing = mp.solutions.drawing_utils

        while self.running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                self.pose_detected = True
                lm = results.pose_landmarks.landmark
                
                required_landmarks = [11, 13, 15, 12, 14, 16]
                
                all_joints_visible = all(lm[idx].visibility > 0.8 for idx in required_landmarks)
                
                if all_joints_visible:
                    self.pose_detected = True
                    
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                    # ==================== LENGAN KIRI ====================
                    bahu_l = lm[11]
                    siku_l  = lm[13]
                    pergelangan_l = lm[15]

                    # Upper Arm (Bahu - Siku)
                    self.angle_l_upper = math.degrees(math.atan2(siku_l.y - bahu_l.y, siku_l.x - bahu_l.x)) - 90

                    # Lower Arm / Forearm (Siku - Pergelangan)
                    vec1 = (siku_l.x - bahu_l.x, siku_l.y - bahu_l.y)
                    vec2 = (pergelangan_l.x - siku_l.x, pergelangan_l.y - siku_l.y)
                    angle_lower = math.degrees(math.atan2(vec2[1], vec2[0]) - math.atan2(vec1[1], vec1[0]))
                    self.angle_l_lower = angle_lower

                    # ==================== LENGAN KANAN ====================
                    bahu_r = lm[12]
                    siku_r  = lm[14]
                    pergelangan_r = lm[16]

                    self.angle_r_upper = math.degrees(math.atan2(siku_r.y - bahu_r.y, siku_r.x - bahu_r.x)) - 90
                    self.angle_r_upper = -self.angle_r_upper - 180

                    vec1r = (siku_r.x - bahu_r.x, siku_r.y - bahu_r.y)
                    vec2r = (pergelangan_r.x - siku_r.x, pergelangan_r.y - siku_r.y)
                    angle_lower_r = math.degrees(math.atan2(vec2r[1], vec2r[0]) - math.atan2(vec1r[1], vec1r[0]))
                    self.angle_r_lower = angle_lower_r
                else:
                    self.pose_detected = False
            else:
                self.pose_detected = False

            with self.frame_lock:
                self.current_frame = frame.copy()

            time.sleep(0.001)

        self.cap.release()


# ==============================================================================
# 2. URSINA SETUP
# ==============================================================================
app = Ursina()
window.title = "Motion Capture - Upper & Lower Arm"
window.borderless = False
window.exit_button.visible = False

avatar_container = Entity(position=(0.4, 0, 0))

try:
    karakter = Actor("karakter.glb")
    karakter.reparentTo(avatar_container)
    karakter.setScale(1.0)
    print("[OK] Model loaded")
except Exception as e:
    print(f"[ERROR] Model gagal dimuat: {e}")
    karakter = None

# Webcam Display
panda_tex = PandaTexture()
panda_tex.setup2dTexture(640, 480, PandaTexture.T_unsigned_byte, PandaTexture.F_rgb)
cam_texture = Texture(panda_tex)

Entity(
    parent=camera.ui,
    model='quad',
    texture=cam_texture,
    position=(-0.65, 0.3),
    scale=(0.55, 0.4)
)

# Joints
bone_l_upper = None
bone_l_lower = None
bone_r_upper = None
bone_r_lower = None

# ==============================================================================
# 3. INIT JOINTS
# ==============================================================================
def init_joints():
    global bone_l_upper, bone_l_lower, bone_r_upper, bone_r_lower
    if not karakter: return
    try:
        bone_l_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_L_UpperArm")
        bone_l_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_L_LowerArm")
        bone_r_upper = karakter.controlJoint(None, "modelRoot", "J_Bip_R_UpperArm")
        bone_r_lower = karakter.controlJoint(None, "modelRoot", "J_Bip_R_LowerArm")
        print("[BERHASIL] Semua joint lengan berhasil dikontrol")
        print(karakter.listJoints())
    except:
        print("[WARNING] Beberapa joint tidak ditemukan")

invoke(init_joints, delay=1.0)

camera.position = (0.4, 1.5, -4)
camera.lookAt(avatar_container)
EditorCamera()

# ==============================================================================
# 4. UPDATE
# ==============================================================================
SMOOTH = 0.10

def update():
    if ai_mocap.current_frame is not None:
        with ai_mocap.frame_lock:
            frame_copy = ai_mocap.current_frame.copy()
        img_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        img_correct = cv2.flip(img_rgb, 0) 
        panda_tex.setRamImage(img_correct.tobytes())

    if not ai_mocap.pose_detected or not karakter:
        return
    
    if ai_mocap.pose_detected:
        # Jika terdeteksi, gunakan sudut hasil tracking dari MediaPipe
        target_l_upper = ai_mocap.angle_l_upper
        target_l_lower = ai_mocap.angle_l_lower
        target_r_upper = ai_mocap.angle_r_upper
        target_r_lower = ai_mocap.angle_r_lower
    else:
        target_l_upper = 0.0
        target_l_lower = 0.0
        target_r_upper = 0.0
        target_r_lower = 0.0
    
    # Lengan Kiri
    if bone_l_upper and not bone_l_upper.isEmpty():
        bone_l_upper.setP(lerp(bone_l_upper.getP(), target_l_upper, SMOOTH))

    if bone_l_lower and not bone_l_lower.isEmpty():
        bone_l_lower.setP(lerp(bone_l_lower.getP(), target_l_lower, SMOOTH))

    # Lengan Kanan
    if bone_r_upper and not bone_r_upper.isEmpty():
        bone_r_upper.setP(lerp(bone_r_upper.getP(), target_r_upper, SMOOTH))

    if bone_r_lower and not bone_r_lower.isEmpty():
        bone_r_lower.setP(lerp(bone_r_lower.getP(), target_r_lower, SMOOTH))


def on_destroy():
    ai_mocap.running = False
    ai_mocap.join(timeout=1)

# ==============================================================================
# RUN
# ==============================================================================
ai_mocap = MocapAIThread()
ai_mocap.start()

print("="*70)
print("Motion Capture dengan Upper + Lower Arm AKTIF")
print("Coba tekuk siku kamu sekarang")
print("="*70)

app.run()