import cv2
import math
from ursina import *
from direct.actor.Actor import Actor
from panda3d.core import Texture as PandaTexture
from mocap_thread import MocapThread

# Parameter konfigurasi interpolasi (smoothing) dan sensitivitas torso
SMOOTH = 0.25
DEADZONE = 0.2
TORSO_GAIN = 3.0

# Inisialisasi dan mulai thread background untuk pemrosesan MediaPipe
ai_mocap = MocapThread()
ai_mocap.start()

# Inisialisasi aplikasi Ursina Engine dan konfigurasi window viewport
app = Ursina()
window.title = "Mocap Viewport Studio"
window.borderless = False
window.exit_button.visible = False

# Konfigurasi visual latar belakang dan sistem kabut (fog) studio
window.color = color.dark_gray
scene.fog_color = color.dark_gray
scene.fog_density = (4, 18)

# Indikator garis sumbu (grid guideline) untuk referensi ruang 3D
Entity(model='line', scale=(6,1,1), color=color.red, position=(0.4, 1, 0))
Entity(model='line', scale=(6,1,1), color=color.blue, rotation_y=90, position=(0.4, 1, 0))

# Inisialisasi container objek dan pemuatan model avatar 3D (.glb)
avatar_container = Entity(position=(0.4, 1, 0))
karakter = None
try:
    karakter = Actor("karakter.glb")
    karakter.reparentTo(avatar_container)
    karakter.setScale(1.0)
except Exception as e:
    print(f"[ERROR] Gagal memuat Model: {e}")

# Konfigurasi posisi kamera utama dan kamera editor (orbit cam)
camera.position = (0.4, 1.7, -6)
camera.lookAt(avatar_container)
EditorCamera()

# Setup tekstur Panda3D untuk menampilkan feed webcam real-time pada UI
panda_tex = PandaTexture()
panda_tex.setup2dTexture(640, 480, PandaTexture.T_unsigned_byte, PandaTexture.F_rgb)
cam_texture = Texture(panda_tex)
Entity(parent=camera.ui, model='quad', texture=cam_texture, position=(-0.65, 0.3), scale=(0.55, 0.4))

# Penampung global untuk objek kontrol sendi (bones/joints) avatar
l_upper = l_lower = l_hand = r_upper = r_lower = r_hand = c_head = c_spine = None

def init_joints():
    """
    Mengambil alih kendali transformasi sendi spesifik dari model 3D
    berdasarkan hierarki bone armature (Biped).
    """
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

# Menunda inisialisasi sendi selama 1 detik agar model termuat sempurna
invoke(init_joints, delay=1.0)

# Pembuatan elemen antarmuka (UI) slider untuk mengatur kehalusan gerakan
Text(text="Gunakan Slider untuk Tuning Offset secara Real-Time", position=(-0.85, -0.15), scale=1.0)
slider_smooth = Slider(min=0.01, max=1.0, default=SMOOTH, text='Smooth Speed', position=(-0.65, -0.25), dynamic=True)
tracking_was_active = False

def update():
    """
    Fungsi update loop utama Ursina (berjalan setiap frame).
    Menangani sinkronisasi frame webcam dan pemetaan rotasi sendi avatar.
    """
    global tracking_was_active, l_upper, l_lower, l_hand, r_upper, r_lower, r_hand, c_head, c_spine
    
    # Update tampilan tekstur kamera pada UI (jika ada frame baru)
    if ai_mocap.current_frame is not None:
        with ai_mocap.frame_lock:
            frame_copy = ai_mocap.current_frame.copy()
            img_correct = cv2.flip(frame_copy, 0) 
            panda_tex.setRamImage(img_correct.tobytes())

    if not karakter:
        return

    # Membaca nilai smoothing secara dinamis dari slider UI
    current_smooth = slider_smooth.value
    s = current_smooth

    # Proses pemetaan animasi jika pose tubuh terdeteksi oleh kamera
    if ai_mocap.pose_detected:
        if not tracking_was_active:
            print("Tracking Regained -> Mengambil alih kendali joint...")
            init_joints()
            tracking_was_active = True
            
        # Pemetaan rotasi Pitch dan Roll pada sendi kepala
        if c_head:
            target_head = ai_mocap.head_roll * 0.85          
            c_head.setR(lerp(c_head.getR(), target_head, s)) 
            c_head.setP(lerp(c_head.getP(), ai_mocap.head_roll * 0.3, s*0.6))
            
        # Pemetaan rotasi tulang belakang dengan kalkulasi deadzone
        if c_spine:
            raw = ai_mocap.spine_roll
            if abs(raw) < DEADZONE:
                filtered = 0
            else:
                filtered = math.copysign(abs(raw) - DEADZONE, raw) * TORSO_GAIN * 0.7  
                
            c_spine.setR(lerp(c_spine.getR(), filtered, s))           
            c_spine.setP(lerp(c_spine.getP(), filtered * 0.25, s*0.5))
                
        # Pemetaan rotasi penuh 3D untuk struktur lengan kiri
        if l_upper:
            l_upper.setP(lerp(l_upper.getP(), ai_mocap.l_upper + 90, s))   
            l_upper.setR(lerp(l_upper.getR(), ai_mocap.l_upper * -0.25, s*0.4))  

        if l_lower:
            l_lower.setP(lerp(l_lower.getP(), -ai_mocap.l_lower, s))
            
        # Pemetaan rotasi penuh 3D untuk struktur lengan kanan
        if r_upper:
            r_upper.setP(lerp(r_upper.getP(), -ai_mocap.r_upper + 90, s))
            r_upper.setR(lerp(r_upper.getR(), ai_mocap.r_upper * -0.25, s*0.4))

        if r_lower:
            r_lower.setP(lerp(r_lower.getP(), ai_mocap.r_lower, s))
            
        # Logika otomatisasi rotasi pergelangan tangan berdasarkan tekukan siku
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
                
    # Melepaskan kontrol sendi kembali ke default (ragdoll/pose asal) jika pelacakan hilang
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
    """
    Callback otomatis saat aplikasi ditutup untuk menghentikan
    dan membersihkan background thread secara aman (anti-stuck).
    """
    ai_mocap.running = False
    ai_mocap.join(timeout=1)

# Log status inisialisasi aplikasi ke konsol
print("="*60)
print("SIMPLE MOTION CAPTURE WITH UNLOCKED ARMS RUNNING...")
print("="*60)

# Memulai loop utama engine grafis Ursina
app.run()