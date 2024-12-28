from development.toonnx.superpoint import SuperPoint
import torch
import onnxruntime as ort
import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import cv2
import numpy as np

sp = SuperPoint().eval()

dummy_input = torch.randn(1,1,480,752)

dynamic_axes = {
    "input":       {2: "height", 3: "width"},
    "keypoints":   {1: "num_keypoints"},
    "scores":      {1: "num_keypoints"},
    "descriptors": {1: "num_keypoints"}
}

torch.onnx.export(
    sp,
    dummy_input,
    'SuperPointAlt.onnx',
    input_names=["input"],
    output_names=["keypoints", "scores", "descriptors"],
    dynamic_axes=dynamic_axes,
    opset_version=17
)

onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model('SuperPointAlt.onnx'), auto_merge=True), 'SuperPointAlt.onnx')
'''

'''
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)
sp_onnx = torch.onnx.dynamo_export(sp, dummy_input, export_options=export_options)
sp_onnx.save('SuperPointAlt.onnx')

onnx_model = onnx.load_model('SuperPointAlt.onnx')
onnx.checker.check_model(onnx_model)

ogimg = cv2.imread('../tensorrt/build/00000.jpg')
img = cv2.cvtColor(ogimg, cv2.COLOR_BGR2GRAY) / 255.0
img = img[None][None].astype(np.float32)

ort_session = ort.InferenceSession('SuperPointAlt.onnx')
outputs = ort_session.run(
    None, {'input': img}
)

keypoints = outputs[0][0]
h, w = ogimg.shape[:2]
size = np.array([w, h], dtype=np.float32)
normalized_keypoints = 2.0 * keypoints / size - 1.0

for keypoint in keypoints:
    print(keypoint)
    cv2.circle(ogimg, (int(keypoint[0]), int(keypoint[1])), 1, (0, 255, 0), -1)

cv2.imshow('sp', ogimg)
cv2.waitKey(0)
cv2.destroyAllWindows()