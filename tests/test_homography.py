"""Camera / homography math and registration refinement."""

import numpy as np

from pitchlens.schema import apply_h
from pitchlens.vision import pitch_model as pm
from pitchlens.vision.camera import CameraOperator, PinholeCamera
from pitchlens.vision.registration import Calibration, blend_h, reprojection_error, smooth_homographies


def test_ground_homography_matches_full_projection():
    cam = PinholeCamera.look_at([52.5, -30, 21], [40, 34, 0], 2100)
    pts = np.array([[0, 0], [16.5, 13.84], [52.5, 34], [105, 68], [30, 50]], float)
    uv, _ = cam.project(np.c_[pts, np.zeros(len(pts))])
    assert np.allclose(apply_h(cam.ground_homography(), pts), uv, atol=1e-6)


def test_calibration_recovers_homography_from_4_points():
    cam = PinholeCamera.look_at([52.5, -30, 21], [70, 34, 0], 2300)
    H = cam.ground_homography()
    names = ["right_box_bl", "right_box_tl", "right_box_br", "right_box_tr", "right_spot"]
    pitch = np.array([pm.KEYPOINTS[n] for n in names])
    k = Calibration(0, pitch, apply_h(H, pitch))
    assert reprojection_error(k.homography(), H, 1920, 1080) < 1e-3


def test_blend_and_smoothing_are_identity_on_static_camera():
    H = PinholeCamera.look_at([52.5, -30, 21], [50, 34, 0], 2100).ground_homography()
    assert reprojection_error(blend_h(H, H, 0.5), H, 1920, 1080) < 1e-3
    sm = smooth_homographies([H] * 9, 1920, 1080, 2.0)
    assert max(reprojection_error(s, H, 1920, 1080) for s in sm) < 1e-3


def test_operator_follows_ball():
    ball = np.c_[np.linspace(30, 80, 300), np.full(300, 34)]
    cams = CameraOperator().path(ball, 30)
    centre_x = [apply_h(np.linalg.inv(c.ground_homography()), np.array([[960, 600]]))[0, 0] for c in cams]
    assert centre_x[-1] > centre_x[0] + 20  # panned right with the play


def test_chamfer_refinement_reduces_error():
    """Render one synthetic frame, perturb the true homography, refine it back."""
    import cv2

    from pitchlens.footage.broadcast import draw_lines, grass_texture, warp_texture
    from pitchlens.vision.registration import PitchRegistrar, RegistrationConfig

    cam = PinholeCamera.look_at([52.5, -30, 21], [70, 34, 0], 2300)
    frame = np.zeros((1080, 1920, 3), np.uint8)
    warp_texture(frame, cam, grass_texture())
    draw_lines(frame, cam)
    H = cam.ground_homography()
    from pitchlens.schema import CameraTrack
    from pitchlens.vision.calibrate import from_ground_truth

    track = CameraTrack(fps=30, width=1920, height=1080, homographies=[H.ravel().tolist()])
    clicks = from_ground_truth(track, [0], noise_px=4.0, seed=3)["keyframes"][0]["points"]
    noisy = Calibration(0, np.array([pm.KEYPOINTS[p["name"]] for p in clicks]), np.array([p["px"] for p in clicks]))
    reg = PitchRegistrar([noisy], RegistrationConfig())
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    Hw = reg.S @ noisy.homography()
    before = reprojection_error(reg.Si @ Hw, H, 1920, 1080)
    for _ in range(3):
        Hw, q = reg.refine(small, Hw, reg._static_mask(small.shape[:2]))
    after = reprojection_error(reg.Si @ Hw, H, 1920, 1080)
    assert after < 0.35 * before and after < 3.0, (before, after)  # 10 px -> ~2 px; small cross-platform variance
