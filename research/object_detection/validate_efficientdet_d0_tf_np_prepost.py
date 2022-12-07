import collections
import copy
import functools
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow.compat.v1 as tf
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from object_detection import eval_util, inputs_np, model_lib
from object_detection.builders import image_resizer_builder, model_builder
from object_detection.core import box_list, box_list_ops, model
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util, label_map_util
from object_detection.utils import ops
from object_detection.utils import ops as util_ops
from object_detection.utils import shape_utils

# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/inputs.py#L48
_LABEL_OFFSET = 1

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
def eval_continuously(
    pipeline_config_path,
    config_override=None,
    sample_1_of_n_eval_examples=1,
    sample_1_of_n_eval_on_train_examples=1,
    override_eval_num_epochs=True,
    postprocess_on_cpu=False,
    checkpoint_dir=None,
    wait_interval=180,
    timeout=3600,
    eval_index=0,
    **kwargs,
):
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L40
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    configs = config_util.get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override
    )
    kwargs.update({"sample_1_of_n_eval_examples": sample_1_of_n_eval_examples})
    configs = config_util.merge_external_params_with_configs(
        configs, None, kwargs_dict=kwargs
    )

    model_config = configs["model"]
    train_input_config = configs["train_input_config"]
    eval_on_train_input_config = copy.deepcopy(train_input_config)
    eval_on_train_input_config.sample_1_of_n_examples = (
        sample_1_of_n_eval_on_train_examples
    )
    if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
        tf.logging.warning(
            (
                "Expected number of evaluation epochs is 1, but "
                "instead encountered `eval_on_train_input_config"
                ".num_epochs` = %d. Overwriting `num_epochs` to 1."
            ),
            eval_on_train_input_config.num_epochs,
        )
        eval_on_train_input_config.num_epochs = 1

    strategy = tf.compat.v2.distribute.get_strategy()
    with strategy.scope():
        detection_model = model_builder.build(
            model_config=model_config, is_training=True
        )

    eval_input = eval_input_np(model_config)

    for latest_checkpoint in tf.train.checkpoints_iterator(
        checkpoint_dir, timeout=timeout, min_interval_secs=wait_interval
    ):

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

        ckpt.restore(latest_checkpoint).expect_partial()

        eager_eval_loop(
            detection_model,
            configs,
            eval_input,
            postprocess_on_cpu=postprocess_on_cpu,
        )

        return


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L833-L1019
def eager_eval_loop(
    detection_model,
    configs,
    eval_dataset,
    postprocess_on_cpu=False,
):
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L124
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    del postprocess_on_cpu

    is_training = False
    detection_model._is_training = is_training  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(is_training)

    strategy = tf.compat.v2.distribute.get_strategy()

    results = []
    for i, (features, labels) in enumerate(eval_dataset.values()):

        prediction_dict, groundtruth_dict, eval_features = compute_eval_dict(
            detection_model, features, labels
        )
        (
            local_prediction_dict,
            local_groundtruth_dict,
            local_eval_features,
        ) = tf.nest.map_structure(
            strategy.experimental_local_results,
            [prediction_dict, groundtruth_dict, eval_features],
        )
        local_prediction_dict = concat_replica_results(local_prediction_dict)
        local_groundtruth_dict = concat_replica_results(local_groundtruth_dict)
        local_eval_features = concat_replica_results(local_eval_features)

        eval_dict = prepare_eval_dict(
            local_prediction_dict, local_groundtruth_dict, local_eval_features
        )

        if i % 100 == 0:
            print(f"Finished eval step %{i}")

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
                    "image_id": int(features["hash"]),
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

    coco_detections = coco.loadRes(results)
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_tools.py#L207
    coco_eval = COCOeval(coco, coco_detections, "bbox")
    coco_eval.params.imgIds = list(set([elem["image_id"] for elem in results]))
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/metrics/coco_tools.py#L285-L287
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    eval_result = coco_eval.stats
    print(f"DetectionBoxes_Precision/mAP: {eval_result[0]}")
    return coco_eval


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L896-L926
def compute_eval_dict(
    detection_model: model.DetectionModel,
    features: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L209
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    groundtruth_dict = labels
    preprocessed_images = features[fields.InputDataFields.image]
    prediction_dict = predict(preprocessed_images, detection_model)
    prediction_dict = postprocess(
        prediction_dict,
        features[fields.InputDataFields.true_image_shape],
        detection_model,
    )
    eval_features = {
        fields.InputDataFields.image: features[fields.InputDataFields.image],
        fields.InputDataFields.original_image: features[
            fields.InputDataFields.original_image
        ],
        fields.InputDataFields.original_image_spatial_shape: features[
            fields.InputDataFields.original_image_spatial_shape
        ],
        fields.InputDataFields.true_image_shape: features[
            fields.InputDataFields.true_image_shape
        ],
        inputs_np.HASH_KEY: features[inputs_np.HASH_KEY],
    }
    return prediction_dict, groundtruth_dict, eval_features


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L525
def predict(
    preprocessed_inputs: np.ndarray, model: model.DetectionModel
) -> Dict[str, np.ndarray]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/meta_architectures/ssd_meta_arch.py#L526
    # arguments and variables which does not acffect eval mAP might be removed or modified.
    feature_maps = model._feature_extractor(tf.convert_to_tensor(preprocessed_inputs))

    feature_map_spatial_dims = model._get_feature_map_spatial_dims(feature_maps)
    image_shape = preprocessed_inputs.shape

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

    boxlist_list = _anchor_generate(
        feature_map_spatial_dims, im_height=image_shape[1], im_width=image_shape[2]
    )
    _anchors = np.concatenate(boxlist_list, 0)
    predictor_results_dict = model._box_predictor(feature_maps)
    predictions_dict = {
        "preprocessed_inputs": preprocessed_inputs,
        "feature_maps": [fm.numpy() for fm in feature_maps],
        "anchors": _anchors,
    }
    for prediction_key, prediction_list in iter(predictor_results_dict.items()):
        prediction = np.concatenate(prediction_list, axis=1)
        predictions_dict[prediction_key] = prediction
    return predictions_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L1197
def _batch_decode(
    box_encodings: np.ndarray, anchors: np.ndarray, model: model.DetectionModel
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
        np.reshape(box_encodings, [-1, model._box_coder.code_size]),
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
    model: model.DetectionModel,
) -> Dict[str, np.ndarray]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/meta_architectures/ssd_meta_arch.py#L802
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    preprocessed_images = prediction_dict["preprocessed_inputs"]
    box_encodings = prediction_dict["box_encodings"]
    class_predictions_with_background = prediction_dict[
        "class_predictions_with_background"
    ]

    detection_boxes = _batch_decode(box_encodings, prediction_dict["anchors"], model)
    detection_boxes = np.expand_dims(detection_boxes, axis=2)

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L727-L728](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L727-L728)
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L135](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L135)
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/post_processing_builder.py#L60-L62](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L60-L62)
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/post_processing_builder.py#L140-L141](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L140-L141)
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/post_processing_builder.py#L112-L119](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L112-L119)
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # pylint: disable=invalid-name
        return 1 / (1 + np.exp(-x))

    detection_scores_with_background = _sigmoid(class_predictions_with_background)

    detection_scores = detection_scores_with_background

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
        max_total_size: int = 0,
        clip_window: Optional[np.array] = None,
        change_coordinate_frame: bool = False,
        num_valid_boxes: Optional[int] = None,
        soft_nms_sigma: float = 0.0,
    ) -> Tuple[np.array, np.array, np.array, np.array, Dict[str, np.array], np.array]:
        boxes_shape = boxes.shape
        batch_size = boxes_shape[0]

        nms_configs = {
            "score_thresh": score_thresh,
            "iou_thresh": iou_thresh,
            "max_size_per_class": max_size_per_class,
            "max_total_size": max_total_size,
            "change_coordinate_frame": change_coordinate_frame,
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
        max_total_size: int = 0,
        clip_window: Optional[np.array] = None,
        change_coordinate_frame: bool = False,
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
            change_coordinate_frame,
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
        change_coordinate_frame: bool,
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
        change_coordinate_frame=True,
    )
    nmsed_boxes, nmsed_scores, nmsed_classes, num_detections = _non_max_suppression_fn(
        detection_boxes,
        detection_scores,
        clip_window=_compute_clip_window(preprocessed_images, tuple(true_image_shapes)),
    )

    detection_dict = {
        fields.DetectionResultFields.detection_boxes: nmsed_boxes,
        fields.DetectionResultFields.detection_scores: nmsed_scores,
        fields.DetectionResultFields.detection_classes: nmsed_classes,
        fields.DetectionResultFields.num_detections: num_detections,
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
    groundtruth=None,
    class_agnostic=False,
    scale_to_absolute=False,
    original_image_spatial_shapes=None,
    true_image_shapes=None,
    max_gt_boxes=None,
    label_id_offset=1,
):
    input_data_fields = fields.InputDataFields
    output_dict = {
        input_data_fields.original_image: images,
        input_data_fields.key: keys,
        input_data_fields.original_image_spatial_shape: (original_image_spatial_shapes),
        input_data_fields.true_image_shape: true_image_shapes,
    }

    detection_fields = fields.DetectionResultFields
    detection_boxes = detections[detection_fields.detection_boxes]
    detection_scores = detections[detection_fields.detection_scores]
    num_detections = detections[detection_fields.num_detections]

    detection_classes = detections[detection_fields.detection_classes] + label_id_offset

    output_dict[detection_fields.detection_boxes] = [
        _scale_box_to_absolute(boxes, image_shape)
        for boxes, image_shape in zip(detection_boxes, original_image_spatial_shapes)
    ]

    output_dict[detection_fields.detection_classes] = detection_classes
    output_dict[detection_fields.detection_scores] = detection_scores
    output_dict[detection_fields.num_detections] = num_detections

    return output_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L733-L823
def prepare_eval_dict(
    detections: Dict[str, np.ndarray],
    groundtruth: Dict[str, np.ndarray],
    features: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/validate_efficientdet_d0_tf.py#L267
    # arguments and variables which does not acffect eval mAP might be removed or modified.
    groundtruth_classes_one_hot = groundtruth[
        fields.InputDataFields.groundtruth_classes
    ]
    label_id_offset = 1
    groundtruth_classes = (
        tf.argmax(groundtruth_classes_one_hot, axis=2) + label_id_offset
    )
    groundtruth[fields.InputDataFields.groundtruth_classes] = groundtruth_classes

    eval_images = features[fields.InputDataFields.original_image]
    true_image_shapes = features[fields.InputDataFields.true_image_shape][:, :3]
    original_image_spatial_shapes = features[
        fields.InputDataFields.original_image_spatial_shape
    ]

    eval_dict = result_dict_for_batched_example(
        eval_images,
        features[inputs_np.HASH_KEY],
        detections,
        groundtruth,
        scale_to_absolute=True,
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
        new_tensor_dict[key] = np.concatenate(values, axis=0)
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
    num_classes: int,
    retain_original_image: bool = True,
) -> Dict[str, Union[np.ndarray, List]]:
    # copied from https://github.com/deeplearningfromscratch/tf-models/blob/effdet-d0/research/object_detection/inputs_np.py#L65
    # arguments and variables which does not acffect eval mAP might be removed or modified.

    out_tensor_dict = tensor_dict.copy()
    input_fields = fields.InputDataFields

    if retain_original_image:
        out_tensor_dict[input_fields.original_image] = image_resizer_fn(
            out_tensor_dict[input_fields.image]
        )[0].astype(np.uint8)

    # Apply model preprocessing ops and resize instance masks.
    image = out_tensor_dict[input_fields.image]
    preprocessed_resized_image, true_image_shape = model_preprocess_fn(
        np.expand_dims(image.astype(np.float32), axis=0)
    )

    preprocessed_shape = preprocessed_resized_image.shape
    new_height, new_width = preprocessed_shape[1], preprocessed_shape[2]

    im_box = np.stack(
        [
            0.0,
            0.0,
            float(new_height) / float(true_image_shape[0, 0]),
            float(new_width) / float(true_image_shape[0, 1]),
        ]
    )

    bboxes = out_tensor_dict[input_fields.groundtruth_boxes]
    realigned_bboxes = _change_coordinate_frame(bboxes, im_box)
    out_tensor_dict[input_fields.groundtruth_boxes] = realigned_bboxes

    out_tensor_dict[input_fields.image] = np.squeeze(preprocessed_resized_image, axis=0)
    out_tensor_dict[input_fields.true_image_shape] = np.squeeze(
        true_image_shape, axis=0
    )

    zero_indexed_groundtruth_classes = (
        out_tensor_dict[input_fields.groundtruth_classes] - _LABEL_OFFSET
    )
    out_tensor_dict[input_fields.groundtruth_classes] = np.eye(num_classes)[
        zero_indexed_groundtruth_classes.reshape(-1)
    ]
    out_tensor_dict[input_fields.num_groundtruth_boxes] = np.array(
        out_tensor_dict[input_fields.groundtruth_boxes].shape[0], dtype=np.int32
    )
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


def eval_input_np(model_config):
    num_classes = config_util.get_number_of_classes(model_config)

    transform_data_fn = functools.partial(
        transform_input_data,
        model_preprocess_fn=_preprocess,
        image_resizer_fn=_image_resizer_fn,
        num_classes=num_classes,
    )
    eval_dataset = dict()
    for image_id in coco.getImgIds():
        # if image_id not in test_image_id_list:
        #     continue
        image = coco.loadImgs(ids=[image_id])[0]
        tensor_dict = dict()
        features = dict()
        labels = dict()
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

        anns = coco.imgToAnns[image_id]
        tensor_dict["num_groundtruth_boxes"] = len(anns)
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/dataset_tools/create_coco_tf_record.py#L128-L134
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/dataset_tools/create_coco_tf_record.py#L207-L222
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/data_decoders/tf_example_decoder.py#L293-L295
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        bboxes = []
        for subdict in anns:
            (x, y, width, height) = tuple(subdict["bbox"])
            xmin = float(x) / w
            xmax = float(x + width) / w
            ymin = float(y) / h
            ymax = float(y + height) / h
            bboxes.append([ymin, xmin, ymax, xmax])
        tensor_dict["groundtruth_boxes"] = np.asarray(bboxes, dtype=np.float32)
        tensor_dict["groundtruth_classes"] = np.asarray(
            [subdict["category_id"] for subdict in anns], dtype=np.int64
        )
        tensor_dict["groundtruth_area"] = np.asarray(
            [subdict["area"] for subdict in anns], dtype=np.float32
        )
        tensor_dict["groundtruth_is_crowd"] = np.asarray(
            [bool(subdict["iscrowd"]) for subdict in anns], dtype=bool
        )
        if len(anns) == 0:
            tensor_dict["groundtruth_boxes"] = np.empty((0, 4), dtype=np.float32)
            tensor_dict["groundtruth_classes"] = np.empty(0, dtype=np.int64)
            tensor_dict["groundtruth_area"] = np.empty(0, dtype=np.float32)
            tensor_dict["groundtruth_is_crowd"] = np.empty(0, dtype=bool)

        tensor_dict = transform_data_fn(tensor_dict)

        features["image"] = np.expand_dims(tensor_dict["image"], 0)
        features["hash"] = np.expand_dims(image_id, 0)
        features["true_image_shape"] = np.expand_dims(
            tensor_dict["true_image_shape"], 0
        )
        features["original_image_spatial_shape"] = np.expand_dims(
            tensor_dict["original_image_spatial_shape"], 0
        )
        features["original_image"] = np.expand_dims(tensor_dict["original_image"], 0)
        labels["num_groundtruth_boxes"] = np.expand_dims(
            tensor_dict["num_groundtruth_boxes"], 0
        )
        labels["groundtruth_boxes"] = np.expand_dims(
            tensor_dict["groundtruth_boxes"], 0
        )
        labels["groundtruth_classes"] = np.expand_dims(
            tensor_dict["groundtruth_classes"], 0
        )
        labels["groundtruth_area"] = np.expand_dims(tensor_dict["groundtruth_area"], 0)
        labels["groundtruth_is_crowd"] = np.expand_dims(
            tensor_dict["groundtruth_is_crowd"], 0
        )

        eval_dataset[image_id] = [features, labels]

    return eval_dataset


if __name__ == "__main__":
    root_dir = "object_detection/efficientdet_d0_coco17_tpu-32/eval"
    pipeline_config_path = os.path.join(root_dir, "pipeline.config")
    checkpoint_dir = os.path.join(root_dir, "checkpoint")
    val_image_dir = Path("dataset/mscoco/val2017")
    val_annotations_file = Path("dataset/mscoco/annotations/instances_val2017.json")
    coco = COCO(val_annotations_file)
    eval_continuously(
        pipeline_config_path=pipeline_config_path,
        model_dir=None,  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_main_tf2.py#L44-L46
        trans_steps=None,  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_main_tf2.py#L35
        sample_1_of_n_eval_examples=None,  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_main_tf2.py#L38-L39
        sample_1_of_n_eval_on_train_examples=5,  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_main_tf2.py#L40-L43
        checkpoint_dir=checkpoint_dir,  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_main_tf2.py#L47-L50
        wait_interval=300,  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_main_tf2.py#L89
        timeout=3600,  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_main_tf2.py#L52-L53
    )
