import onnxruntime as ort
import  numpy as np

ort_session = ort.InferenceSession("facenet.onnx")

outputs = ort_session.run(
    None,
    {"actual_input_1": np.random.randn(1, 3, 160, 160).astype(np.float32)},
)
print("type: ", type())
#print(outputs[0].shape)
