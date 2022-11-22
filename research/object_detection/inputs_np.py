# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model input function for tf-learn object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from object_detection.builders import dataset_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import model_builder
from object_detection.builders import preprocessor_builder
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import densepose_ops
from object_detection.core import keypoint_ops
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.data_decoders import tf_example_decoder
from object_detection.protos import eval_pb2
from object_detection.protos import image_resizer_pb2
from object_detection.protos import input_reader_pb2
from object_detection.protos import model_pb2
from object_detection.protos import train_pb2
from object_detection.utils import config_util
from object_detection.utils import ops as util_ops
from object_detection.utils import shape_utils

HASH_KEY = "hash"
HASH_BINS = 1 << 31
SERVING_FED_EXAMPLE_KEY = "serialized_example"
_LABEL_OFFSET = 1

# A map of names to methods that help build the input pipeline.
INPUT_BUILDER_UTIL_MAP = {
    "dataset_build": dataset_builder.build,
    "model_build": model_builder.build,
}


def convert_labeled_classes_to_k_hot(
    groundtruth_labeled_classes, num_classes, map_empty_to_ones=False
):
    # If the input labeled_classes is empty, it assumes all classes are
    # exhaustively labeled, thus returning an all-one encoding.
    def true_fn():
        return tf.sparse_to_dense(
            groundtruth_labeled_classes - _LABEL_OFFSET,
            [num_classes],
            tf.constant(1, dtype=tf.float32),
            validate_indices=False,
        )

    def false_fn():
        return tf.ones(num_classes, dtype=tf.float32)

    if map_empty_to_ones:
        return tf.cond(tf.size(groundtruth_labeled_classes) > 0, true_fn, false_fn)
    return true_fn()


def _remove_unrecognized_classes(class_ids, unrecognized_label):
    recognized_indices = tf.squeeze(
        tf.where(tf.greater(class_ids, unrecognized_label)), -1
    )
    return tf.gather(class_ids, recognized_indices)


def assert_or_prune_invalid_boxes(boxes):
    ymin, xmin, ymax, xmax = tf.split(boxes, num_or_size_splits=4, axis=1)

    height_check = tf.Assert(tf.reduce_all(ymax >= ymin), [ymin, ymax])
    width_check = tf.Assert(tf.reduce_all(xmax >= xmin), [xmin, xmax])

    with tf.control_dependencies([height_check, width_check]):
        boxes_tensor = tf.concat([ymin, xmin, ymax, xmax], axis=1)
        boxlist = box_list.BoxList(boxes_tensor)
        # TODO(b/149221748) Remove pruning when XLA supports assertions.
        boxlist = box_list_ops.prune_small_boxes(boxlist, 0)

    return boxlist.get()


def transform_input_data(
    tensor_dict,
    model_preprocess_fn,
    image_resizer_fn,
    num_classes,
    data_augmentation_fn=None,
    merge_multiple_boxes=False,
    retain_original_image=False,
    use_multiclass_scores=False,
    use_bfloat16=False,
    retain_original_image_additional_channels=False,
    keypoint_type_weight=None,
    image_classes_field_map_empty_to_ones=True,
):
    out_tensor_dict = tensor_dict.copy()

    input_fields = fields.InputDataFields
    labeled_classes_field = input_fields.groundtruth_labeled_classes
    image_classes_field = input_fields.groundtruth_image_classes
    verified_neg_classes_field = input_fields.groundtruth_verified_neg_classes
    not_exhaustive_field = input_fields.groundtruth_not_exhaustive_classes

    if (
        labeled_classes_field in out_tensor_dict
        and image_classes_field in out_tensor_dict
    ):
        raise KeyError(
            "groundtruth_labeled_classes and groundtruth_image_classes"
            "are provided by the decoder, but only one should be set."
        )

    for field, map_empty_to_ones in [
        (labeled_classes_field, True),
        (image_classes_field, image_classes_field_map_empty_to_ones),
        (verified_neg_classes_field, False),
        (not_exhaustive_field, False),
    ]:
        if field in out_tensor_dict:
            out_tensor_dict[field] = _remove_unrecognized_classes(
                out_tensor_dict[field], unrecognized_label=-1
            )
            out_tensor_dict[field] = convert_labeled_classes_to_k_hot(
                out_tensor_dict[field], num_classes, map_empty_to_ones
            )
    # print(f'{input_fields.multiclass_scores in out_tensor_dict=}')
    # print(f'{input_fields.groundtruth_boxes in out_tensor_dict=}')
    # print(f'{retain_original_image=}')
    # print(f'{input_fields.image_additional_channels in out_tensor_dict=}')
    # print(f'{data_augmentation_fn is not None=}')
    # print(f'{input_fields.groundtruth_boxes in tensor_dict=}')
    # print(f'{input_fields.groundtruth_keypoints in tensor_dict=}')
    # print(f'{input_fields.groundtruth_dp_surface_coords in tensor_dict=}')
    # print(f'{use_bfloat16=}')
    # print(f'{input_fields.groundtruth_instance_masks in out_tensor_dict=}')
    # print(f'{use_multiclass_scores=}')
    # print(f'{input_fields.groundtruth_confidences in out_tensor_dict=}')
    # print(f'{merge_multiple_boxes=}')
    # print(f'{input_fields.groundtruth_boxes in out_tensor_dict=}')
    #
    # input_fields.multiclass_scores in out_tensor_dict=False
    # input_fields.groundtruth_boxes in out_tensor_dict=True
    # retain_original_image=True
    # input_fields.image_additional_channels in out_tensor_dict=False
    # data_augmentation_fn is not None=False
    # input_fields.groundtruth_boxes in tensor_dict=True
    # input_fields.groundtruth_keypoints in tensor_dict=False
    # input_fields.groundtruth_dp_surface_coords in tensor_dict=False
    # use_bfloat16=False
    # input_fields.groundtruth_instance_masks in out_tensor_dict=False
    # use_multiclass_scores=False
    # input_fields.groundtruth_confidences in out_tensor_dict=False
    # merge_multiple_boxes=False
    # input_fields.groundtruth_boxes in out_tensor_dict=True

    if input_fields.groundtruth_boxes in out_tensor_dict:
        out_tensor_dict = util_ops.filter_groundtruth_with_nan_box_coordinates(
            out_tensor_dict
        )
        out_tensor_dict = util_ops.filter_unrecognized_classes(out_tensor_dict)

    if retain_original_image:
        out_tensor_dict[input_fields.original_image] = tf.cast(
            image_resizer_fn(out_tensor_dict[input_fields.image], None)[0], tf.uint8
        )

    # Apply model preprocessing ops and resize instance masks.
    image = out_tensor_dict[input_fields.image]
    preprocessed_resized_image, true_image_shape = model_preprocess_fn(
        tf.expand_dims(tf.cast(image, dtype=tf.float32), axis=0)
    )

    preprocessed_shape = tf.shape(preprocessed_resized_image)
    new_height, new_width = preprocessed_shape[1], preprocessed_shape[2]

    im_box = tf.stack(
        [
            0.0,
            0.0,
            tf.to_float(new_height) / tf.to_float(true_image_shape[0, 0]),
            tf.to_float(new_width) / tf.to_float(true_image_shape[0, 1]),
        ]
    )

    if input_fields.groundtruth_boxes in tensor_dict:
        bboxes = out_tensor_dict[input_fields.groundtruth_boxes]
        boxlist = box_list.BoxList(bboxes)
        realigned_bboxes = box_list_ops.change_coordinate_frame(boxlist, im_box)

        realigned_boxes_tensor = realigned_bboxes.get()
        valid_boxes_tensor = assert_or_prune_invalid_boxes(realigned_boxes_tensor)
        out_tensor_dict[input_fields.groundtruth_boxes] = valid_boxes_tensor

    out_tensor_dict[input_fields.image] = tf.squeeze(preprocessed_resized_image, axis=0)
    out_tensor_dict[input_fields.true_image_shape] = tf.squeeze(
        true_image_shape, axis=0
    )

    zero_indexed_groundtruth_classes = (
        out_tensor_dict[input_fields.groundtruth_classes] - _LABEL_OFFSET
    )

    out_tensor_dict[input_fields.groundtruth_classes] = tf.one_hot(
        zero_indexed_groundtruth_classes, num_classes
    )
    out_tensor_dict.pop(input_fields.multiclass_scores, None)

    groundtruth_confidences = tf.ones_like(
        zero_indexed_groundtruth_classes, dtype=tf.float32
    )
    out_tensor_dict[input_fields.groundtruth_confidences] = out_tensor_dict[
        input_fields.groundtruth_classes
    ]

    if input_fields.groundtruth_boxes in out_tensor_dict:
        out_tensor_dict[input_fields.num_groundtruth_boxes] = tf.shape(
            out_tensor_dict[input_fields.groundtruth_boxes]
        )[0]

    return out_tensor_dict


def pad_input_data_to_static_shapes(
    tensor_dict,
    max_num_boxes,
    num_classes,
    spatial_image_shape=None,
    max_num_context_features=None,
    context_feature_length=None,
    max_dp_points=336,
):
    if not spatial_image_shape or spatial_image_shape == [-1, -1]:
        height, width = None, None
    else:
        height, width = spatial_image_shape  # pylint: disable=unpacking-non-sequence

    input_fields = fields.InputDataFields
    num_additional_channels = 0
    # print(f'{input_fields.image_additional_channels in tensor_dict=}')
    # print(f'{input_fields.image in tensor_dict=}')
    # print(f'{num_additional_channels=}')
    # print(f'{input_fields.context_features in tensor_dict and (max_num_context_features is None)=}')
    # print(f'{input_fields.original_image in tensor_dict=}')
    # print(f'{input_fields.groundtruth_keypoints in tensor_dict=}')
    # print(f'{input_fields.groundtruth_keypoint_visibilities in tensor_dict=}')
    # print(f'{fields.InputDataFields.groundtruth_keypoint_depths in tensor_dict=}')
    # print(f'{input_fields.groundtruth_keypoint_weights in tensor_dict=}')
    # print(f'{input_fields.groundtruth_dp_num_points in tensor_dict=}')
    # print(f'{input_fields.groundtruth_track_ids in tensor_dict=}')
    # print(f'{input_fields.groundtruth_verified_neg_classes in tensor_dict=}')
    # print(f'{input_fields.groundtruth_not_exhaustive_classes in tensor_dict=}')
    # print(f'{input_fields.context_features in tensor_dict=}')
    # print(f'{fields.InputDataFields.context_feature_length in tensor_dict=}')
    # print(f'{fields.InputDataFields.context_features_image_id_list in tensor_dict=}')
    # print(f'{input_fields.is_annotated in tensor_dict=}')
    # print(f'{input_fields.num_groundtruth_boxes in padded_tensor_dict=}')
    # input_fields.image_additional_channels in tensor_dict=False
    # input_fields.image in tensor_dict=True
    # num_additional_channels=0
    # input_fields.context_features in tensor_dict and (max_num_context_features is None)=False
    # input_fields.original_image in tensor_dict=True
    # input_fields.groundtruth_keypoints in tensor_dict=False
    # input_fields.groundtruth_keypoint_visibilities in tensor_dict=False
    # fields.InputDataFields.groundtruth_keypoint_depths in tensor_dict=False
    # input_fields.groundtruth_keypoint_weights in tensor_dict=False
    # input_fields.groundtruth_dp_num_points in tensor_dict=False
    # input_fields.groundtruth_track_ids in tensor_dict=False
    # input_fields.groundtruth_verified_neg_classes in tensor_dict=True
    # input_fields.groundtruth_not_exhaustive_classes in tensor_dict=True
    # input_fields.context_features in tensor_dict=False
    # fields.InputDataFields.context_feature_length in tensor_dict=False
    # fields.InputDataFields.context_features_image_id_list in tensor_dict=False
    # input_fields.is_annotated in tensor_dict=False
    # input_fields.num_groundtruth_boxes in padded_tensor_dict=True
    # We assume that if num_additional_channels > 0, then it has already been
    # concatenated to the base image (but not the ground truth).
    num_channels = 3
    if input_fields.image in tensor_dict:
        num_channels = shape_utils.get_dim_as_int(
            tensor_dict[input_fields.image].shape[2]
        )

    padding_shapes = {
        input_fields.image: [height, width, num_channels],
        input_fields.original_image_spatial_shape: [2],
        input_fields.image_additional_channels: [
            height,
            width,
            num_additional_channels,
        ],
        input_fields.source_id: [],
        input_fields.filename: [],
        input_fields.key: [],
        input_fields.groundtruth_difficult: [max_num_boxes],
        input_fields.groundtruth_boxes: [max_num_boxes, 4],
        input_fields.groundtruth_classes: [max_num_boxes, num_classes],
        input_fields.groundtruth_instance_masks: [max_num_boxes, height, width],
        input_fields.groundtruth_instance_mask_weights: [max_num_boxes],
        input_fields.groundtruth_is_crowd: [max_num_boxes],
        input_fields.groundtruth_group_of: [max_num_boxes],
        input_fields.groundtruth_area: [max_num_boxes],
        input_fields.groundtruth_weights: [max_num_boxes],
        input_fields.groundtruth_confidences: [max_num_boxes, num_classes],
        input_fields.num_groundtruth_boxes: [],
        input_fields.groundtruth_label_types: [max_num_boxes],
        input_fields.groundtruth_label_weights: [max_num_boxes],
        input_fields.true_image_shape: [3],
        input_fields.groundtruth_image_classes: [num_classes],
        input_fields.groundtruth_image_confidences: [num_classes],
        input_fields.groundtruth_labeled_classes: [num_classes],
    }

    if input_fields.original_image in tensor_dict:
        padding_shapes[input_fields.original_image] = [
            height,
            width,
            shape_utils.get_dim_as_int(
                tensor_dict[input_fields.original_image].shape[2]
            ),
        ]
    if input_fields.groundtruth_verified_neg_classes in tensor_dict:
        padding_shapes[input_fields.groundtruth_verified_neg_classes] = [num_classes]
    if input_fields.groundtruth_not_exhaustive_classes in tensor_dict:
        padding_shapes[input_fields.groundtruth_not_exhaustive_classes] = [num_classes]

    padded_tensor_dict = {}
    for tensor_name in tensor_dict:
        padded_tensor_dict[tensor_name] = shape_utils.pad_or_clip_nd(
            tensor_dict[tensor_name], padding_shapes[tensor_name]
        )

    # Make sure that the number of groundtruth boxes now reflects the
    # padded/clipped tensors.
    if input_fields.num_groundtruth_boxes in padded_tensor_dict:
        padded_tensor_dict[input_fields.num_groundtruth_boxes] = tf.minimum(
            padded_tensor_dict[input_fields.num_groundtruth_boxes], max_num_boxes
        )
    return padded_tensor_dict


def _get_labels_dict(input_dict):
    """Extracts labels dict from input dict."""
    required_label_keys = [
        fields.InputDataFields.num_groundtruth_boxes,
        fields.InputDataFields.groundtruth_boxes,
        fields.InputDataFields.groundtruth_classes,
        fields.InputDataFields.groundtruth_weights,
    ]
    labels_dict = {}
    for key in required_label_keys:
        labels_dict[key] = input_dict[key]

    optional_label_keys = [
        fields.InputDataFields.groundtruth_confidences,
        fields.InputDataFields.groundtruth_labeled_classes,
        fields.InputDataFields.groundtruth_keypoints,
        fields.InputDataFields.groundtruth_keypoint_depths,
        fields.InputDataFields.groundtruth_keypoint_depth_weights,
        fields.InputDataFields.groundtruth_instance_masks,
        fields.InputDataFields.groundtruth_instance_mask_weights,
        fields.InputDataFields.groundtruth_area,
        fields.InputDataFields.groundtruth_is_crowd,
        fields.InputDataFields.groundtruth_group_of,
        fields.InputDataFields.groundtruth_difficult,
        fields.InputDataFields.groundtruth_keypoint_visibilities,
        fields.InputDataFields.groundtruth_keypoint_weights,
        fields.InputDataFields.groundtruth_dp_num_points,
        fields.InputDataFields.groundtruth_dp_part_ids,
        fields.InputDataFields.groundtruth_dp_surface_coords,
        fields.InputDataFields.groundtruth_track_ids,
        fields.InputDataFields.groundtruth_verified_neg_classes,
        fields.InputDataFields.groundtruth_not_exhaustive_classes,
        fields.InputDataFields.groundtruth_image_classes,
    ]

    for key in optional_label_keys:
        if key in input_dict:
            labels_dict[key] = input_dict[key]
    if fields.InputDataFields.groundtruth_difficult in labels_dict:
        labels_dict[fields.InputDataFields.groundtruth_difficult] = tf.cast(
            labels_dict[fields.InputDataFields.groundtruth_difficult], tf.int32
        )
    return labels_dict


def _replace_empty_string_with_random_number(string_tensor):
    """Returns string unchanged if non-empty, and random string tensor otherwise.

    The random string is an integer 0 and 2**63 - 1, casted as string.


    Args:
      string_tensor: A tf.tensor of dtype string.

    Returns:
      out_string: A tf.tensor of dtype string. If string_tensor contains the empty
        string, out_string will contain a random integer casted to a string.
        Otherwise string_tensor is returned unchanged.

    """

    empty_string = tf.constant("", dtype=tf.string, name="EmptyString")

    random_source_id = tf.as_string(
        tf.random_uniform(shape=[], maxval=2**63 - 1, dtype=tf.int64)
    )

    out_string = tf.cond(
        tf.equal(string_tensor, empty_string),
        true_fn=lambda: random_source_id,
        false_fn=lambda: string_tensor,
    )

    return out_string


def _get_features_dict(input_dict, include_source_id=False):
    """Extracts features dict from input dict."""

    source_id = _replace_empty_string_with_random_number(
        input_dict[fields.InputDataFields.source_id]
    )

    hash_from_source_id = tf.string_to_hash_bucket_fast(source_id, HASH_BINS)
    features = {
        fields.InputDataFields.image: input_dict[fields.InputDataFields.image],
        HASH_KEY: tf.cast(hash_from_source_id, tf.int32),
        fields.InputDataFields.true_image_shape: input_dict[
            fields.InputDataFields.true_image_shape
        ],
        fields.InputDataFields.original_image_spatial_shape: input_dict[
            fields.InputDataFields.original_image_spatial_shape
        ],
    }
    if include_source_id:
        features[fields.InputDataFields.source_id] = source_id
    if fields.InputDataFields.original_image in input_dict:
        features[fields.InputDataFields.original_image] = input_dict[
            fields.InputDataFields.original_image
        ]
    if fields.InputDataFields.image_additional_channels in input_dict:
        features[fields.InputDataFields.image_additional_channels] = input_dict[
            fields.InputDataFields.image_additional_channels
        ]
    if fields.InputDataFields.context_features in input_dict:
        features[fields.InputDataFields.context_features] = input_dict[
            fields.InputDataFields.context_features
        ]
    if fields.InputDataFields.valid_context_size in input_dict:
        features[fields.InputDataFields.valid_context_size] = input_dict[
            fields.InputDataFields.valid_context_size
        ]
    if fields.InputDataFields.context_features_image_id_list in input_dict:
        features[fields.InputDataFields.context_features_image_id_list] = input_dict[
            fields.InputDataFields.context_features_image_id_list
        ]
    return features


def create_predict_input_fn(model_config, predict_input_config):
    """Creates a predict `input` function for `Estimator`.

    Args:
      model_config: A model_pb2.DetectionModel.
      predict_input_config: An input_reader_pb2.InputReader.

    Returns:
      `input_fn` for `Estimator` in PREDICT mode.
    """

    def _predict_input_fn(params=None):
        """Decodes serialized tf.Examples and returns `ServingInputReceiver`.

        Args:
          params: Parameter dictionary passed from the estimator.

        Returns:
          `ServingInputReceiver`.
        """
        del params
        example = tf.placeholder(dtype=tf.string, shape=[], name="tf_example")

        num_classes = config_util.get_number_of_classes(model_config)
        model_preprocess_fn = INPUT_BUILDER_UTIL_MAP["model_build"](
            model_config, is_training=False
        ).preprocess

        image_resizer_config = config_util.get_image_resizer_config(model_config)
        image_resizer_fn = image_resizer_builder.build(image_resizer_config)

        transform_fn = functools.partial(
            transform_input_data,
            model_preprocess_fn=model_preprocess_fn,
            image_resizer_fn=image_resizer_fn,
            num_classes=num_classes,
            data_augmentation_fn=None,
        )

        decoder = tf_example_decoder.TfExampleDecoder(
            load_instance_masks=False,
            num_additional_channels=predict_input_config.num_additional_channels,
        )
        input_dict = transform_fn(decoder.decode(example))
        images = tf.cast(input_dict[fields.InputDataFields.image], dtype=tf.float32)
        images = tf.expand_dims(images, axis=0)
        true_image_shape = tf.expand_dims(
            input_dict[fields.InputDataFields.true_image_shape], axis=0
        )

        return tf_estimator.export.ServingInputReceiver(
            features={
                fields.InputDataFields.image: images,
                fields.InputDataFields.true_image_shape: true_image_shape,
            },
            receiver_tensors={SERVING_FED_EXAMPLE_KEY: example},
        )

    return _predict_input_fn


def create_eval_input_fn(eval_config, eval_input_config, model_config):
    def _eval_input_fn(params=None):
        return eval_input(eval_config, eval_input_config, model_config, params=params)

    return _eval_input_fn


def eval_input(
    eval_config,
    eval_input_config,
    model_config,
    model=None,
    params=None,
    input_context=None,
):
    params = params or {}
    if not isinstance(eval_config, eval_pb2.EvalConfig):
        raise TypeError(
            "For eval mode, the `eval_config` must be a " "train_pb2.EvalConfig."
        )
    if not isinstance(eval_input_config, input_reader_pb2.InputReader):
        raise TypeError(
            "The `eval_input_config` must be a " "input_reader_pb2.InputReader."
        )
    if not isinstance(model_config, model_pb2.DetectionModel):
        raise TypeError("The `model_config` must be a " "model_pb2.DetectionModel.")

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L10
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L459-L484
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L482
    #
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L84
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/models/ssd_efficientnet_bifpn_feature_extractor.py#L234
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/models/ssd_efficientnet_bifpn_feature_extractor.py#L45
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/models/ssd_efficientnet_bifpn_feature_extractor.py#L190-L209
    #
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/shape_utils.py#L471-L499
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/utils/shape_utils.py#L186-L256
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L47-L53
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/image_resizer_builder.py#L76-L92
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/image_resizer_builder.py#L81
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/image_resizer_builder.py#L87
    # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/core/preprocessor.py#L2895-L3005
    def _np_preprocess():
        # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/models/ssd_efficientnet_bifpn_feature_extractor.py#L205-L207
        channel_offset = [0.485, 0.456, 0.406]
        channel_scale = [0.229, 0.224, 0.225]
        normalized_inputs = ((inputs / 255.0) - [[channel_offset]]) / [[channel_scale]]

    model_preprocess_fn = model.preprocess

    def transform_and_pad_input_data_fn(tensor_dict):
        """Combines transform and pad operation."""
        num_classes = config_util.get_number_of_classes(model_config)

        image_resizer_config = config_util.get_image_resizer_config(model_config)
        image_resizer_fn = image_resizer_builder.build(image_resizer_config)
        keypoint_type_weight = eval_input_config.keypoint_type_weight or None

        transform_data_fn = functools.partial(
            transform_input_data,
            model_preprocess_fn=model_preprocess_fn,
            image_resizer_fn=image_resizer_fn,
            num_classes=num_classes,
            data_augmentation_fn=None,
            retain_original_image=eval_config.retain_original_images,
            retain_original_image_additional_channels=eval_config.retain_original_image_additional_channels,
            keypoint_type_weight=keypoint_type_weight,
            image_classes_field_map_empty_to_ones=eval_config.image_classes_field_map_empty_to_ones,
        )
        tensor_dict = pad_input_data_to_static_shapes(
            tensor_dict=transform_data_fn(tensor_dict),
            max_num_boxes=eval_input_config.max_number_of_boxes,
            num_classes=config_util.get_number_of_classes(model_config),
            spatial_image_shape=config_util.get_spatial_image_size(
                image_resizer_config
            ),
            max_num_context_features=config_util.get_max_num_context_features(
                model_config
            ),
            context_feature_length=config_util.get_context_feature_length(model_config),
        )
        include_source_id = eval_input_config.include_source_id
        return (
            _get_features_dict(tensor_dict, include_source_id),
            _get_labels_dict(tensor_dict),
        )

    reduce_to_frame_fn = get_reduce_to_frame_fn(eval_input_config, False)

    dataset = INPUT_BUILDER_UTIL_MAP["dataset_build"](
        eval_input_config,
        batch_size=params["batch_size"] if params else eval_config.batch_size,
        transform_input_data_fn=transform_and_pad_input_data_fn,
        input_context=input_context,
        reduce_to_frame_fn=reduce_to_frame_fn,
    )
    return dataset


def get_reduce_to_frame_fn(input_reader_config, is_training):
    """Returns a function reducing sequence tensors to single frame tensors.

    If the input type is not TF_SEQUENCE_EXAMPLE, the tensors are passed through
    this function unchanged. Otherwise, when in training mode, a single frame is
    selected at random from the sequence example, and the tensors for that frame
    are converted to single frame tensors, with all associated context features.
    In evaluation mode all frames are converted to single frame tensors with
    copied context tensors. After the sequence example tensors are converted into
    one or many single frame tensors, the images from each frame are decoded.

    Args:
      input_reader_config: An input_reader_pb2.InputReader.
      is_training: Whether we are in training mode.

    Returns:
      `reduce_to_frame_fn` for the dataset builder
    """
    if input_reader_config.input_type != (
        input_reader_pb2.InputType.Value("TF_SEQUENCE_EXAMPLE")
    ):
        return lambda dataset, dataset_map_fn, batch_size, config: dataset
    else:

        def reduce_to_frame(dataset, dataset_map_fn, batch_size, input_reader_config):
            """Returns a function reducing sequence tensors to single frame tensors.

            Args:
              dataset: A tf dataset containing sequence tensors.
              dataset_map_fn: A function that handles whether to
                map_with_legacy_function for this dataset
              batch_size: used if map_with_legacy_function is true to determine
                num_parallel_calls
              input_reader_config: used if map_with_legacy_function is true to
                determine num_parallel_calls

            Returns:
              A tf dataset containing single frame tensors.
            """
            if is_training:

                def get_single_frame(tensor_dict):
                    """Returns a random frame from a sequence.

                    Picks a random frame and returns slices of sequence tensors
                    corresponding to the random frame. Returns non-sequence tensors
                    unchanged.

                    Args:
                      tensor_dict: A dictionary containing sequence tensors.

                    Returns:
                      Tensors for a single random frame within the sequence.
                    """
                    num_frames = tf.cast(
                        tf.shape(tensor_dict[fields.InputDataFields.source_id])[0],
                        dtype=tf.int32,
                    )
                    if input_reader_config.frame_index == -1:
                        frame_index = tf.random.uniform(
                            (), minval=0, maxval=num_frames, dtype=tf.int32
                        )
                    else:
                        frame_index = tf.constant(
                            input_reader_config.frame_index, dtype=tf.int32
                        )
                    out_tensor_dict = {}
                    for key in tensor_dict:
                        if key in fields.SEQUENCE_FIELDS:
                            # Slice random frame from sequence tensors
                            out_tensor_dict[key] = tensor_dict[key][frame_index]
                        else:
                            # Copy all context tensors.
                            out_tensor_dict[key] = tensor_dict[key]
                    return out_tensor_dict

                dataset = dataset_map_fn(
                    dataset, get_single_frame, batch_size, input_reader_config
                )
            else:
                dataset = dataset_map_fn(
                    dataset,
                    util_ops.tile_context_tensors,
                    batch_size,
                    input_reader_config,
                )
                dataset = dataset.unbatch()
            # Decode frame here as SequenceExample tensors contain encoded images.
            dataset = dataset_map_fn(
                dataset, util_ops.decode_image, batch_size, input_reader_config
            )
            return dataset

        return reduce_to_frame
