from ursina import *
from panda3d.core import Texture
from direct.actor.Actor import Actor
from tracking_engine import TrackingEngine
from avatar_controller import AvatarController
import cv2

# 1. INIT APP
app = Ursina()
window.title = "AI Motion Capture Studio"
window.borderless = False
window.color = color.dark_gray

# 2. Start tracking engine
tracker = TrackingEngine()
tracker.start()

# 3. Load karakter dan setup controller
karakter = Actor("assets/karakter.glb")

karakter.reparentTo(render)
karakter.setPos(0, 0, 0)
karakter.setScale(1)

# OPTIONAL DEBUG
# karakter.listJoints()

controller = AvatarController(karakter)

# 4. Main camera setup
camera.position = (0, 1.2, -3.5)

# NONAKTIFKAN DULU SAAT DEBUGGING UI
# EditorCamera()

# 5. UI PANEL

stats_panel = Panel(
    parent=camera.ui,
    scale=(0.38, 0.9),
    position=(-0.62, 0),
    color=color.black66,
    origin=(-0.5, 0)
)

title_text = Text(
    parent=camera.ui,
    text="AI MOCAP SYSTEM",
    position=(-0.55, 0.42),
    scale=1.5,
    color=color.cyan
)

fps_text = Text(
    parent=camera.ui,
    text="FPS : 0",
    position=(-0.55, 0.32),
    scale=1.2,
    color=color.white
)

status_text = Text(
    parent=camera.ui,
    text="Tracking : SEARCHING",
    position=(-0.55, 0.26),
    scale=1.2,
    color=color.yellow
)

# 6. Camera feed panel

camera_feed_panel = Entity(
    parent=camera.ui,
    model='quad',
    scale=(0.45, 0.34),
    position=(0.50, 0.28),
    color=color.black33
)

# Pastikan UI selalu di depan
camera_feed_panel.z = -1

feed_title = Text(
    parent=camera.ui,
    text="CAMERA TRACKING FEED",
    position=(0.30, 0.47),
    scale=1.0,
    color=color.white
)

# 7. Create texture
webcam_texture = Texture()

webcam_texture.setup2dTexture(
    320,
    240,
    Texture.T_unsigned_byte,
    Texture.F_rgb
)

camera_feed_panel.model.setTexture(webcam_texture)

# 8. Update loop
def update():

    # FPS
    if time.dt > 0:
        fps_text.text = f"FPS : {int(1 / time.dt)}"

    # TRACKING STATUS
    if tracker.pose_detected:

        status_text.text = "Tracking : ACTIVE"
        status_text.color = color.green

    else:

        status_text.text = "Tracking : LOST"
        status_text.color = color.red

    # UPDATE CAMERA FEED
    if tracker.current_frame is not None:

        try:
            # Convert OpenCV BGR -> RGB
            rgb_frame = cv2.cvtColor(
                tracker.current_frame,
                cv2.COLOR_BGR2RGB
            )

            # Resize agar stabil dan ringan
            rgb_frame = cv2.resize(rgb_frame, (320, 240))

            # Flip vertical agar cocok dengan Panda3D texture
            rgb_frame = cv2.flip(rgb_frame, 0)

            # Update texture TANPA membuat texture baru
            webcam_texture.setRamImage(rgb_frame)

        except Exception as e:
            print("TEXTURE ERROR :", e)


    # UPDATE AVATAR POSE
    controller.update_pose(
        tracker.landmarks,
        tracker.pose_detected
    )

# 9. SAFE EXIT
def exit_program():

    print("Menutup aplikasi...")

    tracker.running = False

    if tracker.is_alive():
        tracker.join()

    application.quit()


# Event close window
base.accept('window-event', lambda window: exit_program() if window is None else None)

# 10. RUN APP
app.run()