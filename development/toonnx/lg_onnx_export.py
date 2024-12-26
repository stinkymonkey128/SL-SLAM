from lightglue import LightGlue
import torch
import onnxruntime as ort
import onnx
import numpy as np

lglue = LightGlue('https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth')
lglue.eval()
keypoints = torch.randn(2, 1024, 2)
descriptors = torch.randn(2, 1024, 256)

export_options = torch.onnx.ExportOptions(dynamic_shapes=None)
lglue_onnx = torch.onnx.dynamo_export(lglue, *(keypoints, descriptors), export_options=export_options)
lglue_onnx.save('LightGlue.onnx')

ort_session = ort.InferenceSession("LightGlue.onnx")
outputs = ort_session.run(
    None, {"l_keypoints_": keypoints.numpy(), "l_descriptors_": descriptors.numpy()}
)
print(outputs)