import math

class AvatarController:
    def __init__(self, actor_model):
        self.char = actor_model
        self.bones = {}
        self.setup_bones()

    def setup_bones(self):
        joint_names = [
            "J_Bip_C_Hips", "J_Bip_C_Spine", "J_Bip_C_Chest",
            "J_Bip_L_UpperArm", "J_Bip_L_Forearm", "J_Bip_L_Hand",
            "J_Bip_R_UpperArm", "J_Bip_R_Forearm", "J_Bip_R_Hand"
        ]
        for name in joint_names:
            try:
                bone = self.char.controlJoint(None, "modelRoot", name)
                if bone and not bone.is_empty():
                    self.bones[name] = bone
                    print(f"[SUCCESS] Berhasil mengunci tulang: {name}")
            except Exception as e:
                print(f"[ERROR] Gagal mengontrol tulang {name}: {e}")

    # FUNGSI BARU: Menghitung sudut tekuk murni di sendi B menggunakan 3 titik (A, B, C)
    def calculate_joint_angle(self, pA, pB, pC):
        # Ambil vektor BA dan BC
        ba = (pA.x - pB.x, pA.y - pB.y)
        bc = (pC.x - pB.x, pC.y - pB.y)

        # Hitung Dot Product dan Magnitudo Vektor
        dot_product = ba[0]*bc[0] + ba[1]*bc[1]
        mod_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mod_bc = math.sqrt(bc[0]**2 + bc[1]**2)

        if mod_ba == 0 or mod_bc == 0:
            return 0

        # Hitung nilai Cosine
        cosine_angle = dot_product / (mod_ba * mod_bc)
        # Amankan nilai antrian agar tidak crash di luar batas -1 dan 1
        cosine_angle = max(-1.0, min(1.0, cosine_angle))

        # Ubah ke derajat
        angle = math.degrees(math.acos(cosine_angle))
        return angle

    def update_pose(self, lm, pose_detected):
        if not pose_detected or not lm:
            return

        # -------------------------------------------------------------
        # 1. SIKU KIRI (Bahu=11, Siku=13, Pergelangan=15)
        # -------------------------------------------------------------
        if "J_Bip_L_Forearm" in self.bones:
            # Hitung sudut tekuk siku bagian dalam
            sudut_siku_l = self.calculate_joint_angle(lm[11], lm[13], lm[15])
            
            # Konversi sudut: MediaPipe mendeteksi lurus = 180°, menekuk = mendekati 0°
            # Kita balik agar Ursina membaca: lurus = 0°, menekuk = hingga 145°
            sudut_tekuk_l = 180 - sudut_siku_l
            
            # BATASAN SENDI (CONSTRAINTS) 180 derajat manusia normal
            sudut_aman_l = max(0, min(sudut_tekuk_l, 145))
            
            # Coba set ke sumbu Pitch (P) atau Roll (R) tergantung orientasi glb Anda
            self.bones["J_Bip_L_Forearm"].setR(sudut_aman_l) 

        # -------------------------------------------------------------
        # 2. SIKU KANAN (Bahu=12, Siku=14, Pergelangan=16)
        # -------------------------------------------------------------
        if "J_Bip_R_Forearm" in self.bones:
            sudut_siku_r = self.calculate_joint_angle(lm[12], lm[14], lm[16])
            sudut_tekuk_r = 180 - sudut_siku_r
            sudut_aman_r = max(0, min(sudut_tekuk_r, 145))
            
            self.bones["J_Bip_R_Forearm"].setR(-sudut_aman_r) # Nilai minus untuk tangan kanan berlawanan arah