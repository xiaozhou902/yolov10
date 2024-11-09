import openvino as ov

# 定义导出的 ONNX 模型路径
onnx_path = '/root/yolov10-main/runs/detect/train64/weights/best.onnx'

# 使用 OpenVINO API 转换 ONNX 模型为 IR 格式
ov_model = ov.convert_model(onnx_path)

# 定义保存的路径
ir_path = '/root/openvino_model/best_model.xml'

# 保存 OpenVINO 模型（IR 格式）
ov.save_model(ov_model, ir_path)

print(f"OpenVINO IR model saved to: {ir_path}")
