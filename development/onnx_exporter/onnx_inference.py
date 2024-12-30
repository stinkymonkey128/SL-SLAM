import onnxruntime as ort
import cv2
import numpy as np
import matplotlib.cm as cm

def draw_matches(kpts0, kpts1, img0, img1, color):
    if img0 is None or img1 is None:
        raise FileNotFoundError('img invalid')

    scale_factor = 0.9
    img0 = cv2.resize(img0, (0, 0), fx=scale_factor, fy=scale_factor)
    img1 = cv2.resize(img1, (0, 0), fx=scale_factor, fy=scale_factor)

    kpts0 = (kpts0 * scale_factor).astype(int)
    kpts1 = (kpts1 * scale_factor).astype(int)

    height = max(img0.shape[0], img1.shape[0])
    width = img0.shape[1] + img1.shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    canvas[:img0.shape[0], :img0.shape[1], :] = img0
    canvas[:img1.shape[0], img0.shape[1]:, :] = img1

    kpts1[:, 0] += img0.shape[1]
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]

    for pt0, pt1, c in zip(kpts0, kpts1, color):
        c = c.tolist()

        thickness = 1
        radius = 2
        cv2.circle(canvas, tuple(pt0), radius, c, -1)
        cv2.circle(canvas, tuple(pt1), radius, c, -1)
        cv2.line(canvas, tuple(pt0), tuple(pt1), c, thickness)

    cv2.imshow('lightglue', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_sp(img, kpts):
    display = img

    for key in kpts:
        cv2.circle(display, (int(key[0]), int(key[1])), 1, (0, 255, 0), -1)

    cv2.imshow('sp', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize_keypoints(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    size = np.array([w, h])
    shift = size / 2
    scale = size.max() / 2
    kpts = (kpts - shift) / scale
    return kpts.astype(np.float32)

og_img0 = cv2.imread('../tensorrt/build/00000.jpg')
img0 = cv2.cvtColor(og_img0, cv2.COLOR_BGR2GRAY) / 255.0
img0 = img0[None][None].astype(np.float32)

og_img1 = cv2.imread('../tensorrt/build/00001.jpg')
img1 = cv2.cvtColor(og_img1, cv2.COLOR_BGR2GRAY) / 255.0
img1 = img1[None][None].astype(np.float32)

sp_session = ort.InferenceSession('SuperPoint.onnx')

out0 = sp_session.run(
    None, {'input': img0}
)

out1 = sp_session.run(
    None, {'input': img1}
)

h, w = og_img0.shape[:2]

key0 = out0[0][0]
norm_key0 = normalize_keypoints(key0, h, w)
desc0 = out0[2][0]

key1 = out1[0][0]
norm_key1 = normalize_keypoints(key1, h, w)
desc1 = out1[2][0]

keypoints = np.stack((norm_key0, norm_key1))
descriptors = np.stack((desc0, desc1))

lg_session = ort.InferenceSession('LightGlue.onnx')

out = lg_session.run(
    None, {'keypoints': keypoints, 'descriptors': descriptors}
)

match_indexes = out[0].astype(np.int64)
match_scores = out[1][0]

kpts0 = key0[match_indexes[:, 1]]
kpts1 = key1[match_indexes[:, 2]]

print(match_scores.shape)

draw_matches(kpts0, kpts1, og_img0, og_img1, cm.jet(match_scores))