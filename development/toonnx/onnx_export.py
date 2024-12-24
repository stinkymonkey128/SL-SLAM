from superpoint import SuperPoint
import torch
import onnx
import onnxruntime as ort

spoint = SuperPoint()
spoint.load_state_dict(torch.hub.load_state_dict_from_url('https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth'))
spoint.eval()

dummy_input = torch.randn(1,1,480,752)

spoint_onnx = torch.onnx.dynamo_export(spoint, dummy_input)
spoint_onnx.save('SuperPoint.onnx')

ort_session = ort.InferenceSession("SuperPoint.onnx")
outputs = ort_session.run(
    None, {"l_data_": dummy_input.numpy()}
)
print(outputs)