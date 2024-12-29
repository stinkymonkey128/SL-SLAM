from lightglue import LightGlue
import torch
import onnxruntime as ort
import onnx
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
import cv2
import numpy as np

lg = LightGlue().eval()

dummy_keypoints = torch.rand(2, 1024, 2)
dummy_descriptors = torch.rand(2, 1024, 256)

dynamic_axes = {
    'matches': {0: 'num_matches'},
    'scores': {0: 'num_matches'}
}

torch.onnx.export(
    lg,
    (dummy_keypoints, dummy_descriptors),
    'LightGlue.onnx',
    input_names=['keypoints', 'descriptors'],
    output_names=['matches', 'scores'],
    dynamic_axes=dynamic_axes,
    opset_version=17
)

onnx.save_model(SymbolicShapeInference.infer_shapes(onnx.load_model('LightGlue.onnx'), auto_merge=True), 'LightGlue.onnx')