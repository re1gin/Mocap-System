import cv2
import mediapipe as mp
import threading

class TrackingEngine(threading.Thread):
    def __init__(self):
        super().__init__()
        # Inisialisasi Kamera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Inisialisasi MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.75, 
            min_tracking_confidence=0.75
        )
        
        # Variabel State untuk Komunikasi dengan main.py
        self.running = True
        self.pose_detected = False
        self.landmarks = None
        self.current_frame = None 

    def run(self):
        mp_draw = mp.solutions.drawing_utils

        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue
            
            # 1. Pra-pemrosesan (Mirroring)
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 2. Proses Deteksi AI
            results = self.pose.process(rgb_frame)

            # 3. Logika Gambar Rangka & Teks Status
            if results.pose_landmarks:
                self.pose_detected = True
                self.landmarks = results.pose_landmarks.landmark

                # Gambar skeleton MediaPipe
                mp_draw.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                cv2.putText(
                    frame, "TRACKING ACTIVE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )
            else:
                self.pose_detected = False
                cv2.putText(
                    frame, "TRACKING LOST", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                )

            # 4. Simpan frame yang sudah digambar agar Jendela Ursina juga bisa melihatnya (jika diperlukan)
            self.current_frame = frame

            # 5. Tampilkan Jendela Terpisah OpenCV
            cv2.imshow("AI Motion Capture - Tracking Feed", frame)

            # 6. KRITIKAL: Gunakan waitKey(1) tanpa memutus loop thread
            # Ini wajib agar jendela OpenCV merespons refresh rate layar komputer
            cv2.waitKey(1)

        # 7. Cleanup resource saat aplikasi distop dari main.py
        print("[THREAD] Menutup semua jendela OpenCV dan melepaskan kamera...")
        self.cap.release()
        cv2.destroyAllWindows()