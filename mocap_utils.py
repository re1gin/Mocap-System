import math

def angle_between(v1, v2):
    """
    Menghitung sudut absolut dalam derajat antara dua vektor 2D.
    Mengembalikan nilai antara 0 hingga 180 derajat.
    """
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
        
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(-1, min(1, cos_angle))
    return math.degrees(math.acos(cos_angle))

def signed_angle_between(v1, v2):
    """
    Menghitung sudut berarah (signed) antara dua vektor 2D menggunakan cross product.
    Menghasilkan nilai positif jika berlawanan arah jarum jam, dan negatif jika searah.
    """
    raw_ang = angle_between(v1, v2)
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    if cross_product < 0:
        return -raw_ang
    return raw_ang

def normalize_angle(angle):
    """
    Menormalisasi nilai sudut agar selalu berada dalam rentang rentang -180 hingga 180 derajat.
    Mencegah terjadinya lonjakan rotasi yang ekstrem pada sendi avatar.
    """
    while angle > 180: angle -= 360
    while angle < -180: angle += 360
    return angle

def low_pass_filter(current_val, previous_val, alpha):
    """
    Menerapkan filter Low-Pass untuk menghaluskan pergerakan data mentah.
    Nilai alpha yang lebih kecil akan menghasilkan gerakan yang lebih stabil tetapi sedikit delay.
    """
    if previous_val is None:
        return current_val
    return alpha * current_val + (1 - alpha) * previous_val