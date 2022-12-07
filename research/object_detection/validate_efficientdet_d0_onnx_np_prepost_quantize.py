import functools
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import furiosa.quantizer.frontend.onnx
import furiosa.quantizer_experimental  # type: ignore[import]
import furiosa.runtime.session
import numpy as np
import onnx
import onnxruntime as ort
import tensorflow.compat.v1 as tf
import tqdm
from furiosa.quantizer_experimental import CalibrationMethod, Calibrator, Graph
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
tf.get_logger().setLevel("ERROR")

# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/g3doc/tf2_training_and_evaluation.md#evaluation
#
# > # From the tensorflow/models/research/ directory
# > PIPELINE_CONFIG_PATH={path to pipeline config file}
# > MODEL_DIR={path to model directory}
# > CHECKPOINT_DIR=${MODEL_DIR}
# > MODEL_DIR={path to model directory}
# > python object_detection/model_main_tf2.py \
# >     --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
# >     --model_dir=${MODEL_DIR} \
# >     --checkpoint_dir=${CHECKPOINT_DIR} \
# >     --alsologtostderr
#


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L1022-L1169
def eval_continuously():
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L40
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    eval_input = eval_input_np()
    onnx_model = onnx.load_model(ONNX_PATH)
    onnx.shape_inference.infer_shapes(onnx_model, check_type=True, strict_mode=True)
    onnx.checker.check_model(onnx_model)


    optimized_onnx_model = furiosa.quantizer.frontend.onnx.optimize_model(onnx_model)
    onnx.checker.check_model(optimized_onnx_model, full_check=True)
    optimized_onnx_model = optimized_onnx_model.SerializeToString()

    calibrator = Calibrator(optimized_onnx_model, CalibrationMethod.MIN_MAX)
    transform_data_fn = functools.partial(
        transform_input_data,
        model_preprocess_fn=_preprocess,
        image_resizer_fn=_image_resizer_fn,
    )
    cal_image_files = [str(f) for f in cal_image_dir.glob("*jpg")]

    for image_file in tqdm.tqdm(
        cal_image_files, desc="Calibrating", unit="image", mininterval=0.5
    ):
        with tf.gfile.GFile(image_file, "rb") as fid:
            encoded_jpg = fid.read()
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/official/legacy/detection/dataloader/tf_example_decoder.py#L61
        img = tf.io.decode_image(encoded_jpg, channels=3).numpy()
        
        image = transform_data_fn({"image": img})["image"].transpose(2, 0, 1)
        calibrator.collect_data([[image[np.newaxis, ...]]])

    ranges = calibrator.compute_range()

    graph = furiosa.quantizer_experimental.quantize(optimized_onnx_model, ranges)
    session = furiosa.runtime.session.create(bytes(graph))

    eager_eval_loop(
        session,
        eval_input,
    )

    return


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L833-L1019
def eager_eval_loop(
    session,
    eval_dataset,
):
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L124
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    results = []
    for i, features in tqdm.tqdm(
        enumerate(eval_dataset.values()),
        desc="Evaluating",
        unit="image",
        mininterval=0.5,
    ):
        prediction_dict, eval_features = compute_eval_dict(session, features)

        local_prediction_dict = concat_replica_results(prediction_dict)
        local_eval_features = concat_replica_results(eval_features)

        eval_dict = prepare_eval_dict(local_prediction_dict, local_eval_features)

        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L1000
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_evaluation.py#L356
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_evaluation.py#L230
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_tools.py#L652-L661
        for bbox, cls, score in zip(
            eval_dict["detection_boxes"][0],
            eval_dict["detection_classes"][0],
            eval_dict["detection_scores"][0],
        ):
            results.append(
                {
                    "image_id": int(features["image_id"]),
                    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_tools.py#L359
                    # convert a box in [ymin, xmin, ymax, xmax] format to COCO format([xmin, ymin, width, height])
                    "bbox": [
                        float(bbox[1]),
                        float(bbox[0]),
                        float(bbox[3] - bbox[1]),
                        float(bbox[2] - bbox[0]),
                    ],
                    "category_id": int(cls),
                    "score": float(score),
                }
            )

        # if i == 100:
        #     break

    coco_detections = coco.loadRes(results)
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_tools.py#L207
    coco_eval = COCOeval(coco, coco_detections, "bbox")
    coco_eval.params.imgIds = list(set([elem["image_id"] for elem in results]))
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_tools.py#L285-L287
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    eval_result = coco_eval.stats
    print(f"DetectionBoxes_Precision/mAP {eval_result[0]}")
    # assert eval_result[0] == 0.4133381090793865, f"{eval_result[0]}"
    print("test passed.")
    return coco_eval


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L896-L926
def compute_eval_dict(
    session: ort.InferenceSession,
    features: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L209
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    preprocessed_images = features["image"]
    prediction_dict = predict(preprocessed_images, session)
    prediction_dict = postprocess(
        prediction_dict,
        features["true_image_shape"],
    )
    eval_features = {
        "image": features["image"],
        "original_image": features["original_image"],
        "original_image_spatial_shape": features["original_image_spatial_shape"],
        "true_image_shape": features["true_image_shape"],
        "image_id": features["image_id"],
    }
    return prediction_dict, eval_features


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L525
def predict(
    preprocessed_inputs: np.ndarray, session: ort.InferenceSession
) -> Dict[str, np.ndarray]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/meta_architectures/ssd_meta_arch.py#L526
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L585-L588
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L38-L45
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L30-L152
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/anchor_generator.py#L81-L112
    def _anchor_generate(
        feature_map_shape_list: List[Tuple[int]], im_height: int, im_width: int
    ) -> List[np.ndarray]:
        # TODO: find the source of anchor grid info
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L117
        anchor_grid_info = [
            {
                "level": 3,
                "info": [
                    [1.0, 1.2599210498948732, 1.5874010519681994],
                    [1.0, 2.0, 0.5],
                    [32.0, 32.0],
                    [8, 8],
                ],
            },
            {
                "level": 4,
                "info": [
                    [1.0, 1.2599210498948732, 1.5874010519681994],
                    [1.0, 2.0, 0.5],
                    [64.0, 64.0],
                    [16, 16],
                ],
            },
            {
                "level": 5,
                "info": [
                    [1.0, 1.2599210498948732, 1.5874010519681994],
                    [1.0, 2.0, 0.5],
                    [128.0, 128.0],
                    [32, 32],
                ],
            },
            {
                "level": 6,
                "info": [
                    [1.0, 1.2599210498948732, 1.5874010519681994],
                    [1.0, 2.0, 0.5],
                    [256.0, 256.0],
                    [64, 64],
                ],
            },
            {
                "level": 7,
                "info": [
                    [1.0, 1.2599210498948732, 1.5874010519681994],
                    [1.0, 2.0, 0.5],
                    [512.0, 512.0],
                    [128, 128],
                ],
            },
        ]
        anchor_grid_list = []
        for feat_shape, grid_info in zip(feature_map_shape_list, anchor_grid_info):
            level = grid_info["level"]
            stride = 2**level
            scales, aspect_ratios, base_anchor_size, anchor_stride = grid_info["info"]
            feat_h = feat_shape[0]
            feat_w = feat_shape[1]
            anchor_offset = [0, 0]

            if im_height % 2.0**level == 0 or im_height == 1:
                anchor_offset[0] = stride / 2.0
            if im_width % 2.0**level == 0 or im_width == 1:
                anchor_offset[1] = stride / 2.0

            (anchor_grid,) = _grid_anchor_generator(
                feature_map_shape_list=[(feat_h, feat_w)],
                scales=scales,
                aspect_ratios=aspect_ratios,
                base_anchor_size=base_anchor_size,
                anchor_stride=anchor_stride,
                anchor_offset=anchor_offset,
            )

            # TODO: find the source of normalize_coordinates
            # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L142
            # normalize_coordinates = True
            # check_range = False # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L845
            anchor_grid = _to_normalized_coordinates(anchor_grid, im_height, im_width)
            anchor_grid_list.append(anchor_grid)
        return anchor_grid_list

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L134
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L30-L137
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L82-L137
    def _grid_anchor_generator(
        feature_map_shape_list: List[Tuple[int, int]],
        scales: List[float],
        aspect_ratios: List[float],
        base_anchor_size: List[float],
        anchor_stride: List[int],
        anchor_offset: List[int],
    ) -> List[np.ndarray]:
        grid_height, grid_width = feature_map_shape_list[0]
        scales_grid, aspect_ratios_grid = _meshgrid(scales, aspect_ratios)
        scales_grid = tf.reshape(scales_grid, [-1])
        aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
        anchors = _tile_anchors(
            grid_height,
            grid_width,
            scales_grid,
            aspect_ratios_grid,
            base_anchor_size,
            anchor_stride,
            anchor_offset,
        )

        return [anchors]

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L120-L121
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/ops.py#L99-L135
    def _meshgrid(x: List[float], y: List[float]) -> Tuple[np.array]:
        x = np.array(x)
        y = np.array(y)

        x_exp_shape = _expanded_shape(x.shape, 0, y.ndim)
        y_exp_shape = _expanded_shape(y.shape, y.ndim, x.ndim)

        xgrid = np.tile(np.reshape(x, x_exp_shape), y_exp_shape).astype(np.float32)
        ygrid = np.tile(np.reshape(y, y_exp_shape), x_exp_shape).astype(np.float32)
        return xgrid, ygrid

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/ops.py#L40-L59
    def _expanded_shape(
        orig_shape: List[int], start_dim: int, num_dims: int
    ) -> List[int]:
        before = orig_shape[:start_dim]
        add_shape = np.ones(np.reshape(num_dims, [1]))
        after = orig_shape[start_dim:]
        new_shape = np.concatenate([before, add_shape, after], 0).astype(np.int64)
        return new_shape

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L124-L130
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L140-L199
    def _tile_anchors(
        grid_height: int,
        grid_width: int,
        scales: List[float],
        aspect_ratios: List[float],
        base_anchor_size: List[float],
        anchor_stride: List[int],
        anchor_offset: List[int],
    ) -> np.ndarray:
        ratio_sqrts = np.sqrt(aspect_ratios)
        heights = scales / ratio_sqrts * base_anchor_size[0]
        widths = scales * ratio_sqrts * base_anchor_size[1]

        y_centers = np.arange(grid_height)
        y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
        x_centers = np.arange(grid_width)
        x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
        x_centers, y_centers = _meshgrid(x_centers, y_centers)

        widths_grid, x_centers_grid = _meshgrid(widths, x_centers)
        heights_grid, y_centers_grid = _meshgrid(heights, y_centers)
        bbox_centers = np.stack([y_centers_grid, x_centers_grid], axis=3)
        bbox_sizes = np.stack([heights_grid, widths_grid], axis=3)
        bbox_centers = np.reshape(bbox_centers, [-1, 2])
        bbox_sizes = np.reshape(bbox_sizes, [-1, 2])
        return _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L202-L213
    def _center_size_bbox_to_corners_bbox(
        centers: np.ndarray, sizes: np.ndarray
    ) -> np.array:
        return np.concatenate([centers - 0.5 * sizes, centers + 0.5 * sizes], 1)

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L148-L149
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L844-L878
    def _to_normalized_coordinates(
        bbox: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        return _scale(bbox, 1 / height, 1 / width)

    feature_map_spatial_dims = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
    image_shape = preprocessed_inputs.shape

    boxlist_list = _anchor_generate(
        feature_map_spatial_dims, im_height=image_shape[1], im_width=image_shape[2]
    )
    _anchors = np.concatenate(boxlist_list, 0)

    raw_bboxes, sigmoided_scores = session.run(preprocessed_inputs.transpose(0, 3, 1, 2)).numpy()
    predictions_dict = {
        "box_encodings": raw_bboxes,
        "detection_scores_with_background": sigmoided_scores,
        "preprocessed_inputs": preprocessed_inputs,
        "anchors": _anchors,
    }
    return predictions_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L1197
def _batch_decode(
    box_encodings: np.ndarray,
    anchors: np.ndarray,
    code_size: int,
) -> np.ndarray:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/meta_architectures/ssd_meta_arch.py#L1673
    # arguments and variables which does not acffect eval mAP might be removed or modified.
    combined_shape = box_encodings.shape
    batch_size = combined_shape[0]
    tiled_anchor_boxes = np.tile(np.expand_dims(anchors, 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = np.reshape(tiled_anchor_boxes, [-1, 4])

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L15-L22
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L1197-L1231
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_coder.py#L80-L92
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/box_coders/faster_rcnn_box_coder.py#L92-L118
    def _box_decode(rel_codes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
        # TODO make anchors np array?
        # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/core/box_list.py#L161-L177
        def _get_center_coordinates_and_sizes(box_corners: np.ndarray) -> np.ndarray:
            ymin, xmin, ymax, xmax = np.moveaxis(np.transpose(box_corners), 0, 0)
            width = xmax - xmin
            height = ymax - ymin
            ycenter = ymin + height / 2.0
            xcenter = xmin + width / 2.0
            return [ycenter, xcenter, height, width]

        ycenter_a, xcenter_a, ha, wa = _get_center_coordinates_and_sizes(anchors)

        # scale_factors=[1.0, 1.0, 1.0, 1.0]
        # so omit https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/box_coders/faster_rcnn_box_coder.py#L105-L109
        # TODO where did the scale_factors come from?
        ty, tx, th, tw = np.moveaxis(np.transpose(rel_codes), 0, 0)
        w = np.exp(tw) * wa
        h = np.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0
        return np.transpose(np.stack([ymin, xmin, ymax, xmax]))

    decoded_boxes = _box_decode(
        np.reshape(box_encodings, [-1, code_size]),
        tiled_anchors_boxlist,
    )

    decoded_boxes = np.reshape(
        decoded_boxes, np.stack([combined_shape[0], combined_shape[1], 4])
    )
    return decoded_boxes


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L655
def postprocess(
    prediction_dict: Dict[str, np.ndarray],
    true_image_shapes: np.ndarray,
) -> Dict[str, np.ndarray]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/meta_architectures/ssd_meta_arch.py#L802
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    preprocessed_images = prediction_dict["preprocessed_inputs"]
    box_encodings = prediction_dict["box_encodings"]
    detection_scores = prediction_dict["detection_scores_with_background"]

    # TODO: check reference
    code_size = 4
    detection_boxes = _batch_decode(
        box_encodings, prediction_dict["anchors"], code_size
    )
    detection_boxes = np.expand_dims(detection_boxes, axis=2)

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L767-L768
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L486-L523
    def _compute_clip_window(
        preprocessed_images: np.ndarray, true_image_shapes: Tuple[int]
    ) -> np.ndarray:
        # always have true_image_shapes
        # so remove https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L508-L509
        # preprocessed_images always have static shape
        # not use shape_utils.combined_static_and_dynamic_shape https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L511-L512

        # NOTE: channel last order in tf.
        resized_inputs_shape = np.array(preprocessed_images.shape)
        true_heights, true_widths, _ = np.moveaxis(true_image_shapes, 0, 1)
        padded_height = resized_inputs_shape[1].astype(np.float32)
        padded_width = resized_inputs_shape[2].astype(np.float32)
        return np.stack(
            [
                np.zeros_like(true_heights),
                np.zeros_like(true_widths),
                true_heights / padded_height,
                true_widths / padded_width,
            ],
            axis=1,
        ).astype(np.float32)

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L878-L1276
    def _batch_multiclass_non_max_suppression(
        boxes: np.ndarray,
        scores: np.ndarray,
        score_thresh: float,
        iou_thresh: float,
        max_size_per_class: int,
        max_total_size: int,
        clip_window: Optional[np.array] = None,
        soft_nms_sigma: float = 0.0,
    ) -> Tuple[np.array, np.array, np.array, np.array, Dict[str, np.array], np.array]:
        boxes_shape = boxes.shape
        batch_size = boxes_shape[0]

        nms_configs = {
            "score_thresh": score_thresh,
            "iou_thresh": iou_thresh,
            "max_size_per_class": max_size_per_class,
            "max_total_size": max_total_size,
            "soft_nms_sigma": soft_nms_sigma,
        }
        # for loop impl. of tf.map_fn
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L1244-L1249
        batch_outputs = [
            _single_image_nms_fn(
                per_image_boxes=boxes[i],
                per_image_scores=scores[i],
                per_image_clip_window=clip_window[i],
                **nms_configs,
            )
            for i in range(batch_size)
        ]
        # convert List[List[np.array]] to List[np.array]
        batch_outputs = list(map(np.stack, np.stack(batch_outputs, axis=1)))
        batch_nmsed_boxes = batch_outputs[0]
        batch_nmsed_scores = batch_outputs[1]
        batch_nmsed_classes = batch_outputs[2]
        batch_num_detections = batch_outputs[-1]

        return (
            batch_nmsed_boxes,
            batch_nmsed_scores,
            batch_nmsed_classes,
            batch_num_detections,
        )

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L1099-L1232
    def _single_image_nms_fn(
        per_image_boxes: np.ndarray,
        per_image_scores: np.ndarray,
        per_image_clip_window: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nmsed_boxes, nmsed_scores, nmsed_classes = _multiclass_non_max_suppression(
            boxes=per_image_boxes,
            scores=per_image_scores,
            clip_window=per_image_clip_window,
            **kwargs,
        )

        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L1069-L1089
        max_total_size = kwargs["max_total_size"]
        nmsed_boxes = nmsed_boxes[:max_total_size]
        nmsed_scores = nmsed_scores[:max_total_size]
        nmsed_classes = nmsed_classes[:max_total_size]
        num_detections = nmsed_boxes.shape[0]

        return [
            nmsed_boxes,
            nmsed_scores,
            nmsed_classes,
        ] + [num_detections]

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L1200-L1215
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L422-L651
    def _multiclass_non_max_suppression(
        boxes: np.ndarray,
        scores: np.ndarray,
        score_thresh: float,
        iou_thresh: float,
        max_size_per_class: int,
        max_total_size: int,
        clip_window: Optional[np.array] = None,
        pad_to_max_output_size: bool = False,
        soft_nms_sigma: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        num_classes = scores.shape[1]

        nms_result = defaultdict(list)
        num_valid_nms_boxes_cumulative = np.array(0, dtype=np.int64)
        per_class_boxes_list = np.moveaxis(boxes, 0, 1)
        per_class_scores_list = np.moveaxis(scores, 0, 1)

        boxes_ids = (
            range(num_classes) if len(per_class_boxes_list) > 1 else [0] * num_classes
        )
        for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
            per_class_boxes = per_class_boxes_list[boxes_idx]
            per_class_scores = per_class_scores_list[class_idx]
            max_selection_size = min(max_size_per_class, per_class_boxes.shape[0])
            # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L583-L589
            # https://github.com/tensorflow/tensorflow/blob/v2.10.1/tensorflow/python/ops/image_ops_impl.py#L3804-L3891](https://github.com/tensorflow/tensorflow/blob/v2.10.1/tensorflow/python/ops/image_ops_impl.py#L3804-L3891)
            # https://github.com/tensorflow/tensorflow/blob/c7adce4cb2293b66a96b811a0dcdcfb7e361c23f/tensorflow/core/kernels/image/non_max_suppression_op.cc#L829-L907
            # https://github.com/tensorflow/tensorflow/blob/c7adce4cb2293b66a96b811a0dcdcfb7e361c23f/tensorflow/core/kernels/image/non_max_suppression_op.cc#L194-L330
            # TODO: CPP impl?
            (
                selected_indices,
                selected_scores,
            ) = tf.image.non_max_suppression_with_scores(
                per_class_boxes,
                per_class_scores,
                max_selection_size,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh,
                soft_nms_sigma=soft_nms_sigma,
            )
            if selected_indices is None:
                continue

            selected_indices = selected_indices.numpy()
            selected_scores = selected_scores.numpy()
            selected_boxes = per_class_boxes[selected_indices]
            num_valid_nms_boxes_cumulative += selected_indices.shape[0]
            selected_classes = np.full(selected_scores.shape, class_idx)

            nms_result["boxes"].append(selected_boxes)
            nms_result["scores"].append(selected_scores)
            nms_result["classes"].append(selected_classes)

        selected_boxes = np.concatenate(nms_result["boxes"])
        selected_scores = np.concatenate(nms_result["scores"])
        selected_classes = np.concatenate(nms_result["classes"])

        # TODO: check argsort
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L662-L700
        sorted_boxes = selected_boxes[np.argsort(selected_scores)]
        sorted_scores = selected_scores[np.argsort(selected_scores)]
        sorted_classes = selected_classes[np.argsort(selected_scores)]

        sorted_boxes, sorted_scores, sorted_classes = _clip_window_prune_boxes(
            sorted_boxes,
            sorted_scores,
            sorted_classes,
            clip_window,
            pad_to_max_output_size,
        )

        return sorted_boxes, sorted_scores, sorted_classes

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L636-L638
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L345-L388
    def _clip_window_prune_boxes(
        sorted_boxes: np.ndarray,
        sorted_scores: np.ndarray,
        sorted_classes: np.ndarray,
        clip_window: np.ndarray,
        pad_to_max_output_size: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L133-L171
        def _clip_to_window(
            boxes: np.ndarray,
            scores: np.ndarray,
            classes: np.ndarray,
            window: np.ndarray,
            filter_nonoverlapping: bool = True,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            y_min, x_min, y_max, x_max = np.split(boxes, 4, axis=1)
            win_y_min = window[0]
            win_x_min = window[1]
            win_y_max = window[2]
            win_x_max = window[3]
            y_min_clipped = np.maximum(np.minimum(y_min, win_y_max), win_y_min)
            y_max_clipped = np.maximum(np.minimum(y_max, win_y_max), win_y_min)
            x_min_clipped = np.maximum(np.minimum(x_min, win_x_max), win_x_min)
            x_max_clipped = np.maximum(np.minimum(x_max, win_x_max), win_x_min)
            clipped = np.concatenate(
                [y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped], 1
            )
            if filter_nonoverlapping:
                areas = _area(clipped)
                nonzero_area_indices = np.reshape(np.where(areas > 0.0), -1)
                clipped = clipped[nonzero_area_indices]
                scores = scores[nonzero_area_indices]
                classes = classes[nonzero_area_indices]
            return clipped, scores, classes

        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L49-L62
        def _area(boxlist: List[np.ndarray]) -> np.ndarray:
            y_min, x_min, y_max, x_max = np.split(boxlist, 4, axis=1)
            return np.squeeze((y_max - y_min) * (x_max - x_min), 1)

        sorted_boxes, sorted_scores, sorted_classes = _clip_to_window(
            sorted_boxes,
            sorted_scores,
            sorted_classes,
            clip_window,
            filter_nonoverlapping=not pad_to_max_output_size,
        )

        sorted_boxes_size = sorted_boxes.shape[0]
        non_zero_box_area = _area(sorted_boxes).astype(np.bool)
        sorted_scores = np.where(
            non_zero_box_area,
            sorted_scores,
            -1 * np.ones(sorted_boxes_size),
        )

        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L662-L700
        sorted_boxes = sorted_boxes[np.argsort(sorted_scores)[::-1]]
        sorted_classes = sorted_classes[np.argsort(sorted_scores)[::-1]]
        sorted_scores = sorted_scores[np.argsort(sorted_scores)[::-1]]

        sorted_boxes = _change_coordinate_frame(sorted_boxes, clip_window)
        return sorted_boxes, sorted_scores, sorted_classes

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L764-L770
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L129-L134
    # TODO: where are the default values not specified in configuration?
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L58-L59
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L70-L109
    _non_max_suppression_fn = functools.partial(
        _batch_multiclass_non_max_suppression,
        score_thresh=9.99999993922529e-09,
        iou_thresh=0.5,
        max_size_per_class=100,
        max_total_size=100,
        soft_nms_sigma=0.0,
    )
    nmsed_boxes, nmsed_scores, nmsed_classes, num_detections = _non_max_suppression_fn(
        detection_boxes,
        detection_scores,
        clip_window=_compute_clip_window(preprocessed_images, tuple(true_image_shapes)),
    )

    detection_dict = {
        "detection_boxes": nmsed_boxes,
        "detection_scores": nmsed_scores,
        "detection_classes": nmsed_classes,
        "num_detections": num_detections,
    }
    return detection_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L878
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L82-L105
def _scale(boxes: np.ndarray, y_scale: float, x_scale: float, scope=None) -> np.ndarray:
    y_min, x_min, y_max, x_max = np.split(boxes, 4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    return np.concatenate((y_min, x_min, y_max, x_max), axis=1)


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/eval_util.py#L548-L551
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L881-L920
def _scale_box_to_absolute(boxes: np.ndarray, image_shape: List[int]) -> np.ndarray:
    height, width = image_shape[0], image_shape[1]

    return _scale(boxes, height, width)


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/eval_util.py#L767-L1065
def result_dict_for_batched_example(
    images,
    keys,
    detections,
    original_image_spatial_shapes=None,
    true_image_shapes=None,
    label_id_offset=1,
):
    output_dict = {
        "original_image": images,
        "key": keys,
        "original_image_spatial_shape": (original_image_spatial_shapes),
        "true_image_shape": true_image_shapes,
    }

    detection_boxes = detections["detection_boxes"]
    detection_scores = detections["detection_scores"]
    num_detections = detections["num_detections"]

    detection_classes = detections["detection_classes"] + label_id_offset

    output_dict["detection_boxes"] = [
        _scale_box_to_absolute(boxes, image_shape)
        for boxes, image_shape in zip(detection_boxes, original_image_spatial_shapes)
    ]

    output_dict["detection_classes"] = detection_classes
    output_dict["detection_scores"] = detection_scores
    output_dict["num_detections"] = num_detections

    return output_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L733-L823
def prepare_eval_dict(
    detections: Dict[str, np.ndarray],
    features: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L267

    eval_images = features["original_image"]
    true_image_shapes = features["true_image_shape"][:, :3]
    original_image_spatial_shapes = features["original_image_spatial_shape"]

    eval_dict = result_dict_for_batched_example(
        eval_images,
        features["image_id"],
        detections,
        original_image_spatial_shapes=original_image_spatial_shapes,
        true_image_shapes=true_image_shapes,
    )

    return eval_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L826-L830
def concat_replica_results(tensor_dict):
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L310
    # arguments and variables which does not acffect eval mAP might be removed or modified.
    new_tensor_dict = {}
    for key, values in tensor_dict.items():
        new_tensor_dict[key] = np.concatenate((values,), axis=0)
    return new_tensor_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/image_resizer_builder.py#L87
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/preprocessor.py#L2984-L3094
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/image_resizer_builder.py#L86-L92
# TODO: CPP impl tf.image.resize_image?
# TODO: numpy transcoding when writting e2e-script
def _resize_to_range(
    image: np.ndarray,
    min_dimension: Optional[int] = None,
    max_dimension: Optional[int] = None,
    method: tf.image.ResizeMethod = tf.image.ResizeMethod.BILINEAR,
    align_corners: bool = False,
    pad_to_max_dimension: bool = False,
    per_channel_pad_value: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/inputs_np.py#L462
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    def _resize_landscape_image(image):
        # resize a landscape image
        return tf.image.resize_images(
            image,
            np.stack([min_dimension, max_dimension]),
            method=method,
            align_corners=align_corners,
            preserve_aspect_ratio=True,
        )

    def _resize_portrait_image(image):
        # resize a portrait image
        return tf.image.resize_images(
            image,
            np.stack([max_dimension, min_dimension]),
            method=method,
            align_corners=align_corners,
            preserve_aspect_ratio=True,
        )

    image = tf.convert_to_tensor(image)
    if image.shape[0] < image.shape[1]:
        new_image = _resize_landscape_image(image)
    else:
        new_image = _resize_portrait_image(image)
    new_image = new_image.numpy()
    new_size = new_image.shape

    if pad_to_max_dimension:
        # TODO: consider channel first case
        channels = np.moveaxis(new_image, 2, 0)
        if len(channels) != len(per_channel_pad_value):
            raise ValueError(
                "Number of channels must be equal to the length of "
                "per-channel pad value."
            )
        new_image = np.stack(
            [
                np.pad(  # pylint: disable=g-complex-comprehension
                    channels[i],
                    [
                        [0, max_dimension - new_size[0]],
                        [0, max_dimension - new_size[1]],
                    ],
                    constant_values=per_channel_pad_value[i],
                )
                for i in range(len(channels))
            ],
            axis=2,
        )

    result = [new_image]
    result.append(new_size)
    return result


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L484
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L47-L53
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/image_resizer_builder.py#L76-L82
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/image_resizer_builder.py#L86-L92
_image_resizer_fn = functools.partial(
    _resize_to_range,
    min_dimension=512,
    max_dimension=512,
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/image_resizer_builder.py#L81
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/image_resizer_builder.py#L37-L38
    method=tf.image.ResizeMethod.BILINEAR,
    pad_to_max_dimension=True,
    per_channel_pad_value=(0, 0, 0),
)
# copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/inputs_np.py#L528
# arguments and variables which does not acffect eval mAP might be removed or modified.

# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L10
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L459-L484
def _preprocess(inputs: np.ndarray) -> np.ndarray:
    normalized_inputs = _feature_extractor_preprocess(inputs)
    return _resize_images_and_return_shapes(normalized_inputs, _image_resizer_fn)


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L483-L484
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/shape_utils.py#L471-L499
def _resize_images_and_return_shapes(
    inputs: np.ndarray, image_resizer_fn: functools.partial
) -> Tuple[np.ndarray, np.ndarray]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/inputs_np.py#L551
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/shape_utils.py#L492-L497
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/shape_utils.py#L186-L256
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/shape_utils.py#L246
    outputs = [image_resizer_fn(arg) for arg in np.moveaxis(inputs, 0, 0)]
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/shape_utils.py#L251-L255
    outputs = [np.stack(output_tuple) for output_tuple in zip(*outputs)]
    resized_inputs = outputs[0]
    true_image_shapes = outputs[1]
    return resized_inputs, true_image_shapes


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L442-L469
def _change_coordinate_frame(boxes: np.ndarray, window):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    boxlist_new = _scale(
        boxes - [window[0], window[1], window[0], window[1]],
        1.0 / win_height,
        1.0 / win_width,
    )
    return boxlist_new


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/inputs.py#L883
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/inputs.py#L151-L395
def transform_input_data(
    tensor_dict: Dict[str, np.ndarray],
    model_preprocess_fn,
    image_resizer_fn: functools.partial,
    retain_original_image: bool = True,
) -> Dict[str, Union[np.ndarray, List]]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/inputs_np.py#L65
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    out_tensor_dict = tensor_dict.copy()

    if retain_original_image:
        out_tensor_dict["original_image"] = image_resizer_fn(out_tensor_dict["image"])[
            0
        ].astype(np.uint8)

    # Apply model preprocessing ops and resize instance masks.
    image = out_tensor_dict["image"]
    preprocessed_resized_image, true_image_shape = model_preprocess_fn(
        np.expand_dims(image.astype(np.float32), axis=0)
    )

    out_tensor_dict["image"] = np.squeeze(preprocessed_resized_image, axis=0)
    out_tensor_dict["true_image_shape"] = np.squeeze(true_image_shape, axis=0)

    return out_tensor_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L482
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L44
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L83-L90
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/models/ssd_efficientnet_bifpn_feature_extractor.py#L198-L217
def _feature_extractor_preprocess(inputs: np.ndarray) -> np.ndarray:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/inputs_np.py#L569
    # arguments and variables which does not acffect eval mAP might be removed or modified.
    channel_offset = [0.485, 0.456, 0.406]
    channel_scale = [0.229, 0.224, 0.225]
    return ((inputs / 255.0) - [[channel_offset]]) / [[channel_scale]]


def eval_input_np():
    transform_data_fn = functools.partial(
        transform_input_data,
        model_preprocess_fn=_preprocess,
        image_resizer_fn=_image_resizer_fn,
    )
    eval_dataset = dict()
    # fmt: off
    # test_image_id_list = [
    #     397133, 475779, 551215, 48153, 21503, 356347, 83113, 312237, 176606, 518770,
    #     488270, 11760, 404923, 101420, 45550, 10977, 253386, 76731, 417779, 414034,
    #     10363, 11122, 424521, 283785, 279927, 104666, 310622, 449661, 206487, 48555,
    #     325527, 46804, 331352, 562121, 434230, 450758, 339442, 442480, 509131, 520264,
    #     375278, 50326, 354829, 114871, 340697, 183716, 143572, 17436, 20059, 349837,
    #     578093, 351823, 449996, 187236, 27696, 213816, 382111, 159112, 469067, 91500,
    #     360325, 329827, 294163, 50165, 226111, 109798, 550426, 8021, 100582, 130599,
    #     475064, 221708, 110211, 120420, 123131, 295316, 103585, 509735, 205105, 42070,
    #     533206, 493286, 130699, 255483, 315450, 217948, 32038, 369675, 567825, 152771,
    #     229601, 418961, 78565, 187990, 78823, 289229, 443303, 474028, 263796, 375015,
    #     284282,
    # ]
    # fmt: on
    for image_id in coco.getImgIds():
        # if image_id not in test_image_id_list:
        #     continue
        image = coco.loadImgs(ids=[image_id])[0]
        tensor_dict = dict()
        features = dict()
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/dataset_tools/create_coco_tf_record.py#L171-L173
        with tf.gfile.GFile(val_image_dir / image["file_name"], "rb") as fid:
            encoded_jpg = fid.read()
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/official/legacy/detection/dataloader/tf_example_decoder.py#L61
        img = tf.io.decode_image(encoded_jpg, channels=3).numpy()
        # img = np.asarray(Image.open(val_image_dir / image["file_name"]).convert("RGB"), dtype=np.uint8)
        tensor_dict["image"] = img
        h, w = image["height"], image["width"]
        tensor_dict["true_image_shape"] = [h, w, 3]
        tensor_dict["original_image_spatial_shape"] = [h, w]
        tensor_dict = transform_data_fn(tensor_dict)

        features["image"] = np.expand_dims(tensor_dict["image"], 0)
        features["image_id"] = np.expand_dims(image_id, 0)
        features["true_image_shape"] = np.expand_dims(
            tensor_dict["true_image_shape"], 0
        )
        features["original_image_spatial_shape"] = np.expand_dims(
            tensor_dict["original_image_spatial_shape"], 0
        )
        features["original_image"] = np.expand_dims(tensor_dict["original_image"], 0)

        eval_dataset[image_id] = features

    return eval_dataset


if __name__ == "__main__":
    val_image_dir = Path("dataset/mscoco/val2017")
    val_annotations_file = Path("dataset/mscoco/annotations/instances_val2017.json")
    cal_image_dir = Path("dataset/mscoco/mscoco_calibration")
    coco = COCO(val_annotations_file)
    ONNX_PATH = "object_detection/efficientdet_d0.onnx"
    eval_continuously()
