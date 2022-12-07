import onnx
from onnx.tools import update_model_dims

import furiosa.quantizer.frontend.onnx


# Extract pre/post-processing sub-graph
input_path = "object_detection/efficientdet_d0_coco17_tpu-32/eval/efficientdet_d0_orig.onnx"
output_path = "object_detection/efficientdet_d0_coco17_tpu-32/eval/efficientdet_d0.onnx"
input_names = ["StatefulPartitionedCall/EfficientDet-D0/model/stem_conv2d/Conv2D__264:0"]
output_names = ["StatefulPartitionedCall/concat:0", "raw_detection_scores"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# Give spatial dimensions
onnx_model = onnx.load_model(output_path)
optimized_onnx_model = furiosa.quantizer.frontend.onnx.optimize_model(onnx_model, 
                                                                      input_shapes={"StatefulPartitionedCall/EfficientDet-D0/model/stem_conv2d/Conv2D__264:0": [1, 3, 512, 512]})
onnx.save_model(optimized_onnx_model, output_path)
