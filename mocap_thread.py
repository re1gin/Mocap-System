import cv2
import mediapipe as mp
import math
import threading
import time
from mocap_utils import normalize_angle, signed_angle_between, low_pass_filter

SMOOTH_FILTER_ALPHA = 0.15

class MocapThread(threading.Thread):
    """
    Thread khusus untuk menangani penangkapan video (OpenCV) dan 
    pemrosesan pelacakan rangka tubuh secara real-time menggunakan MediaPipe.
    """
    def __init__(self):
        super().__init__(daemon=True)
        # Inisialisasi dan konfigurasi resolusi kamera webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Konfigurasi detektor pose MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        # Status pelacakan dan sinkronisasi thread
        self.pose_detected = False
        self.running = True
        self.frame_lock = threading.Lock()
        
        # Variabel penyimpanan hasil rotasi sendi tubuh
        self.head_roll = 0.0
        self.l_upper = 0.0
        self.l_lower = 0.0
        self.r_upper = 0.0
        self.r_lower = 0.0
        self.spine_roll = 0.0
        
        # Kalibrasi posisi tegap awal pengguna
        self.spine_center = None
        self.current_frame = None

    def run(self):
        """
        Fungsi utama loop thread untuk membaca frame, memproses data landmark,
        dan menghitung sudut orientasi sendi secara berkelanjutan.
        """
        mp_drawing = mp.solutions.drawing_utils

        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue

            # Membalik gambar (mirror) agar gerakan sesuai dengan pengguna
            frame = cv2.flip(frame, 1)
            results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                # Indeks landmark penting (Bahu, Siku, Pergelangan, Pinggul, Telinga)
                joint_indices = [11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 7, 8]
                
                # Memastikan seluruh bagian tubuh atas terdeteksi dengan akurasi tinggi
                if all(lm[i].visibility > 0.7 for i in joint_indices):
                    self.pose_detected = True
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    # Ekstraksi vektor posisi kepala
                    head_vec = (lm[7].x - lm[8].x, lm[7].y - lm[8].y)

                    # Ekstraksi vektor posisi lengan kiri (lengan atas dan lengan bawah)
                    l_upper_vec = (lm[13].x - lm[11].x, lm[13].y - lm[11].y)
                    l_lower_vec = (lm[15].x - lm[13].x, lm[15].y - lm[13].y)

                    # Ekstraksi vektor posisi lengan kanan (lengan atas dan lengan bawah)
                    r_upper_vec = (lm[14].x - lm[12].x, lm[14].y - lm[12].y)
                    r_lower_vec = (lm[16].x - lm[14].x, lm[16].y - lm[14].y)

                    # Menghitung titik tengah bahu dan pinggul untuk kalkulasi tulang belakang
                    shoulder_center = ((lm[11].x + lm[12].x) / 2, (lm[11].y + lm[12].y) / 2)
                    hip_center = ((lm[23].x + lm[24].x) / 2, (lm[23].y + lm[24].y) / 2)
                    spine_vec = (shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
                    
                    # Menghitung data rotasi mentah untuk kepala dan sudut tulang belakang
                    raw_head_roll = normalize_angle(math.degrees(math.atan2(head_vec[1], head_vec[0])))
                    spine_angle = math.degrees(math.atan2(spine_vec[0], -spine_vec[1]))
                    
                    # Mengunci posisi awal badan sebagai standar kalibrasi posisi tegap
                    if self.spine_center is None:
                        self.spine_center = spine_angle
                        print(f"[CALIBRATION] Pusat tegap torso terkunci di: {self.spine_center:.2f}°")
                    
                    # Menghitung kemiringan badan relatif terhadap posisi kalibrasi awal
                    relative_angle = normalize_angle(spine_angle - self.spine_center)
                    raw_spine_roll = max(-45, min(45, relative_angle))
                    
                    # Melakukan transformasi rotasi vektor agar sejajar dengan orientasi bahu
                    rad = math.radians(-relative_angle)
                    cos_r = math.cos(rad)
                    sin_r = math.sin(rad)

                    # Transformasi vektor lengan kiri
                    l_u_x = l_upper_vec[0] * cos_r - l_upper_vec[1] * sin_r
                    l_u_y = l_upper_vec[0] * sin_r + l_upper_vec[1] * cos_r

                    # Transformasi vektor lengan kanan
                    r_u_x = r_upper_vec[0] * cos_r - r_upper_vec[1] * sin_r
                    r_u_y = r_upper_vec[0] * sin_r + r_upper_vec[1] * cos_r
                    
                    # Menghitung sudut rotasi penuh bebas (360 derajat) untuk lengan atas kiri
                    l_upper_angle = math.degrees(math.atan2(l_u_y, l_u_x))
                    raw_l_upper = normalize_angle(-l_upper_angle - 90)
                    
                    # Menghitung sudut tekukan siku kiri (bisa bernilai positif/negatif)
                    raw_l_lower = signed_angle_between(l_upper_vec, l_lower_vec)

                    # Menghitung sudut rotasi penuh bebas (360 derajat) untuk lengan atas kanan
                    r_upper_angle = math.degrees(math.atan2(r_u_y, r_u_x))
                    raw_r_upper = normalize_angle(-r_upper_angle - 90)
                    
                    # Menghitung sudut tekukan siku kanan (bisa bernilai positif/negatif)
                    raw_r_lower = signed_angle_between(r_upper_vec, r_lower_vec)

                    # Menghaluskan seluruh data pergerakan menggunakan filter Low-Pass
                    alpha = SMOOTH_FILTER_ALPHA
                    self.head_roll       = low_pass_filter(raw_head_roll, self.head_roll, alpha)
                    self.l_upper         = low_pass_filter(raw_l_upper, self.l_upper, alpha)
                    self.l_lower         = low_pass_filter(raw_l_lower, self.l_lower, alpha)
                    self.r_upper         = low_pass_filter(raw_r_upper, self.r_upper, alpha)
                    self.r_lower         = low_pass_filter(raw_r_lower, self.r_lower, alpha)
                    self.spine_roll      = low_pass_filter(raw_spine_roll, self.spine_roll, alpha)
                    
                else:
                    self.pose_detected = False

            # Menyimpan salinan frame webcam secara aman menggunakan Lock untuk rendering Ursina
            with self.frame_lock:
                self.current_frame = frame.copy()

            # Mengatur interval jeda thread agar stabil di kisaran 60 FPS
            time.sleep(0.016)

        self.cap.release()