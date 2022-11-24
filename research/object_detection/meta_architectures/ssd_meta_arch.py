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
"""SSD Meta-architecture definition.

General tensorflow implementation of convolutional Multibox/SSD detection
models.
"""
import abc
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.util.deprecation import deprecated_args
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import matcher
from object_detection.core import model
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import ops
from object_detection.utils import shape_utils
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils


# pylint: disable=g-import-not-at-top
try:
  import tf_slim as slim
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top


class SSDFeatureExtractor(object):
  """SSD Slim Feature Extractor definition."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False):
    """Constructor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    self._is_training = is_training
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._pad_to_multiple = pad_to_multiple
    self._conv_hyperparams_fn = conv_hyperparams_fn
    self._reuse_weights = reuse_weights
    self._use_explicit_padding = use_explicit_padding
    self._use_depthwise = use_depthwise
    self._num_layers = num_layers
    self._override_base_feature_extractor_hyperparams = (
        override_base_feature_extractor_hyperparams)

  @property
  def is_keras_model(self):
    return False

  @abc.abstractmethod
  def preprocess(self, resized_inputs):
    """Preprocesses images for feature extraction (minus image resizing).

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
    """
    pass

  @abc.abstractmethod
  def extract_features(self, preprocessed_inputs):
    """Extracts features from preprocessed inputs.

    This function is responsible for extracting feature maps from preprocessed
    images.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    raise NotImplementedError

  def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
    """Returns a map of variables to load from a foreign checkpoint.

    Args:
      feature_extractor_scope: A scope name for the feature extractor.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    """
    variables_to_restore = {}
    for variable in variables_helper.get_global_variables_safely():
      var_name = variable.op.name
      if var_name.startswith(feature_extractor_scope + '/'):
        var_name = var_name.replace(feature_extractor_scope + '/', '')
        variables_to_restore[var_name] = variable

    return variables_to_restore


class SSDKerasFeatureExtractor(tf.keras.Model):
  """SSD Feature Extractor definition."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               freeze_batchnorm,
               inplace_batchnorm_update,
               use_explicit_padding=False,
               use_depthwise=False,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False,
               name=None):
    """Constructor.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: `hyperparams_builder.KerasLayerHyperparams` object
        containing convolution hyperparameters for the layers added on top of
        the base feature extractor.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      num_layers: Number of SSD layers.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_config`.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(SSDKerasFeatureExtractor, self).__init__(name=name)

    self._is_training = is_training
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._pad_to_multiple = pad_to_multiple
    self._conv_hyperparams = conv_hyperparams
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update
    self._use_explicit_padding = use_explicit_padding
    self._use_depthwise = use_depthwise
    self._num_layers = num_layers
    self._override_base_feature_extractor_hyperparams = (
        override_base_feature_extractor_hyperparams)

  @property
  def is_keras_model(self):
    return True

  @abc.abstractmethod
  def preprocess(self, resized_inputs):
    """Preprocesses images for feature extraction (minus image resizing).

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _extract_features(self, preprocessed_inputs):
    """Extracts features from preprocessed inputs.

    This function is responsible for extracting feature maps from preprocessed
    images.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    raise NotImplementedError

  # This overrides the keras.Model `call` method with the _extract_features
  # method.
  def call(self, inputs, **kwargs):
    return self._extract_features(inputs)


class SSDMetaArch(model.DetectionModel):
  """SSD Meta-architecture definition."""

  @deprecated_args(None,
                   'NMS is always placed on TPU; do not use nms_on_host '
                   'as it has no effect.', 'nms_on_host')
  def __init__(self,
               is_training,
               anchor_generator,
               box_predictor,
               box_coder,
               feature_extractor,
               encode_background_as_zeros,
               image_resizer_fn,
               non_max_suppression_fn,
               score_conversion_fn,
               classification_loss,
               localization_loss,
               classification_loss_weight,
               localization_loss_weight,
               normalize_loss_by_num_matches,
               hard_example_miner,
               target_assigner_instance,
               add_summaries=True,
               normalize_loc_loss_by_codesize=False,
               freeze_batchnorm=False,
               inplace_batchnorm_update=False,
               add_background_class=True,
               explicit_background_class=False,
               random_example_sampler=None,
               expected_loss_weights_fn=None,
               use_confidences_as_targets=False,
               implicit_example_weight=0.5,
               equalization_loss_config=None,
               return_raw_detections_during_predict=False,
               nms_on_host=True):
    """SSDMetaArch Constructor.

    TODO(rathodv,jonathanhuang): group NMS parameters + score converter into
    a class and loss parameters into a class and write config protos for
    postprocessing and losses.

    Args:
      is_training: A boolean indicating whether the training version of the
        computation graph should be constructed.
      anchor_generator: an anchor_generator.AnchorGenerator object.
      box_predictor: a box_predictor.BoxPredictor object.
      box_coder: a box_coder.BoxCoder object.
      feature_extractor: a SSDFeatureExtractor object.
      encode_background_as_zeros: boolean determining whether background
        targets are to be encoded as an all zeros vector or a one-hot
        vector (where background is the 0th class).
      image_resizer_fn: a callable for image resizing.  This callable always
        takes a rank-3 image tensor (corresponding to a single image) and
        returns a rank-3 image tensor, possibly with new spatial dimensions and
        a 1-D tensor of shape [3] indicating shape of true image within
        the resized image tensor as the resized image tensor could be padded.
        See builders/image_resizer_builder.py.
      non_max_suppression_fn: batch_multiclass_non_max_suppression
        callable that takes `boxes`, `scores` and optional `clip_window`
        inputs (with all other inputs already set) and returns a dictionary
        hold tensors with keys: `detection_boxes`, `detection_scores`,
        `detection_classes` and `num_detections`. See `post_processing.
        batch_multiclass_non_max_suppression` for the type and shape of these
        tensors.
      score_conversion_fn: callable elementwise nonlinearity (that takes tensors
        as inputs and returns tensors).  This is usually used to convert logits
        to probabilities.
      classification_loss: an object_detection.core.losses.Loss object.
      localization_loss: a object_detection.core.losses.Loss object.
      classification_loss_weight: float
      localization_loss_weight: float
      normalize_loss_by_num_matches: boolean
      hard_example_miner: a losses.HardExampleMiner object (can be None)
      target_assigner_instance: target_assigner.TargetAssigner instance to use.
      add_summaries: boolean (default: True) controlling whether summary ops
        should be added to tensorflow graph.
      normalize_loc_loss_by_codesize: whether to normalize localization loss
        by code size of the box encoder.
      freeze_batchnorm: Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      inplace_batchnorm_update: Whether to update batch norm moving average
        values inplace. When this is false train op must add a control
        dependency on tf.graphkeys.UPDATE_OPS collection in order to update
        batch norm statistics.
      add_background_class: Whether to add an implicit background class to
        one-hot encodings of groundtruth labels. Set to false if training a
        single class model or using groundtruth labels with an explicit
        background class.
      explicit_background_class: Set to true if using groundtruth labels with an
        explicit background class, as in multiclass scores.
      random_example_sampler: a BalancedPositiveNegativeSampler object that can
        perform random example sampling when computing loss. If None, random
        sampling process is skipped. Note that random example sampler and hard
        example miner can both be applied to the model. In that case, random
        sampler will take effect first and hard example miner can only process
        the random sampled examples.
      expected_loss_weights_fn: If not None, use to calculate
        loss by background/foreground weighting. Should take batch_cls_targets
        as inputs and return foreground_weights, background_weights. See
        expected_classification_loss_by_expected_sampling and
        expected_classification_loss_by_reweighting_unmatched_anchors in
        third_party/tensorflow_models/object_detection/utils/ops.py as examples.
      use_confidences_as_targets: Whether to use groundtruth_condifences field
        to assign the targets.
      implicit_example_weight: a float number that specifies the weight used
        for the implicit negative examples.
      equalization_loss_config: a namedtuple that specifies configs for
        computing equalization loss.
      return_raw_detections_during_predict: Whether to return raw detection
        boxes in the predict() method. These are decoded boxes that have not
        been through postprocessing (i.e. NMS). Default False.
      nms_on_host: boolean (default: True) controlling whether NMS should be
        carried out on the host (outside of TPU).
    """
    super(SSDMetaArch, self).__init__(num_classes=box_predictor.num_classes)
    self._is_training = is_training
    self._freeze_batchnorm = freeze_batchnorm
    self._inplace_batchnorm_update = inplace_batchnorm_update

    self._anchor_generator = anchor_generator
    self._box_predictor = box_predictor

    self._box_coder = box_coder
    self._feature_extractor = feature_extractor
    self._add_background_class = add_background_class
    self._explicit_background_class = explicit_background_class

    if add_background_class and explicit_background_class:
      raise ValueError("Cannot have both 'add_background_class' and"
                       " 'explicit_background_class' true.")

    # Needed for fine-tuning from classification checkpoints whose
    # variables do not have the feature extractor scope.
    if self._feature_extractor.is_keras_model:
      # Keras feature extractors will have a name they implicitly use to scope.
      # So, all contained variables are prefixed by this name.
      # To load from classification checkpoints, need to filter out this name.
      self._extract_features_scope = feature_extractor.name
    else:
      # Slim feature extractors get an explicit naming scope
      self._extract_features_scope = 'FeatureExtractor'

    if encode_background_as_zeros:
      background_class = [0]
    else:
      background_class = [1]

    if self._add_background_class:
      num_foreground_classes = self.num_classes
    else:
      num_foreground_classes = self.num_classes - 1

    self._unmatched_class_label = tf.constant(
        background_class + num_foreground_classes * [0], tf.float32)

    self._target_assigner = target_assigner_instance

    self._classification_loss = classification_loss
    self._localization_loss = localization_loss
    self._classification_loss_weight = classification_loss_weight
    self._localization_loss_weight = localization_loss_weight
    self._normalize_loss_by_num_matches = normalize_loss_by_num_matches
    self._normalize_loc_loss_by_codesize = normalize_loc_loss_by_codesize
    self._hard_example_miner = hard_example_miner
    self._random_example_sampler = random_example_sampler
    self._parallel_iterations = 16

    self._image_resizer_fn = image_resizer_fn
    self._non_max_suppression_fn = non_max_suppression_fn
    self._score_conversion_fn = score_conversion_fn

    self._anchors = None
    self._add_summaries = add_summaries
    self._batched_prediction_tensor_names = []
    self._expected_loss_weights_fn = expected_loss_weights_fn
    self._use_confidences_as_targets = use_confidences_as_targets
    self._implicit_example_weight = implicit_example_weight

    self._equalization_loss_config = equalization_loss_config

    self._return_raw_detections_during_predict = (
        return_raw_detections_during_predict)

  @property
  def feature_extractor(self):
    return self._feature_extractor

  @property
  def anchors(self):
    if not self._anchors:
      raise RuntimeError('anchors have not been constructed yet!')
    if not isinstance(self._anchors, box_list.BoxList):
      raise RuntimeError('anchors should be a BoxList object, but is not.')
    return self._anchors

  @property
  def batched_prediction_tensor_names(self):
    if not self._batched_prediction_tensor_names:
      raise RuntimeError('Must call predict() method to get batched prediction '
                         'tensor names.')
    return self._batched_prediction_tensor_names

  def preprocess(self, inputs):
    """Feature-extractor specific preprocessing.

    SSD meta architecture uses a default clip_window of [0, 0, 1, 1] during
    post-processing. On calling `preprocess` method, clip_window gets updated
    based on `true_image_shapes` returned by `image_resizer_fn`.

    Args:
      inputs: a [batch, height_in, width_in, channels] float tensor representing
        a batch of images with values between 0 and 255.0.

    Returns:
      preprocessed_inputs: a [batch, height_out, width_out, channels] float
        tensor representing a batch of images.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Raises:
      ValueError: if inputs tensor does not have type tf.float32
    """
    with tf.name_scope('Preprocessor'):
      normalized_inputs = self._feature_extractor.preprocess(inputs)
      return shape_utils.resize_images_and_return_shapes(
          normalized_inputs, self._image_resizer_fn)

  def _compute_clip_window(self, preprocessed_images, true_image_shapes):
    """Computes clip window to use during post_processing.

    Computes a new clip window to use during post-processing based on
    `resized_image_shapes` and `true_image_shapes` only if `preprocess` method
    has been called. Otherwise returns a default clip window of [0, 0, 1, 1].

    Args:
      preprocessed_images: the [batch, height, width, channels] image
          tensor.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros. Or None if the clip window should cover the full image.

    Returns:
      a 2-D float32 tensor of the form [batch_size, 4] containing the clip
      window for each image in the batch in normalized coordinates (relative to
      the resized dimensions) where each clip window is of the form [ymin, xmin,
      ymax, xmax] or a default clip window of [0, 0, 1, 1].

    """
    if true_image_shapes is None:
      return tf.constant([0, 0, 1, 1], dtype=tf.float32)

    resized_inputs_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_images)
    true_heights, true_widths, _ = tf.unstack(
        tf.cast(true_image_shapes, dtype=tf.float32), axis=1)
    padded_height = tf.cast(resized_inputs_shape[1], dtype=tf.float32)
    padded_width = tf.cast(resized_inputs_shape[2], dtype=tf.float32)
    return tf.stack(
        [
            tf.zeros_like(true_heights),
            tf.zeros_like(true_widths), true_heights / padded_height,
            true_widths / padded_width
        ],
        axis=1)

  def predict(self, preprocessed_inputs, true_image_shapes):
    """Predicts unpostprocessed tensors from input tensor.

    This function takes an input batch of images and runs it through the forward
    pass of the network to yield unpostprocessesed predictions.

    A side effect of calling the predict method is that self._anchors is
    populated with a box_list.BoxList of anchors.  These anchors must be
    constructed before the postprocess or loss functions can be called.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] image tensor.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.

    Returns:
      prediction_dict: a dictionary holding "raw" prediction tensors:
        1) preprocessed_inputs: the [batch, height, width, channels] image
          tensor.
        2) box_encodings: 4-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        3) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions (at class index 0).
        4) feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].
        5) anchors: 2-D float tensor of shape [num_anchors, 4] containing
          the generated anchors in normalized coordinates.
        6) final_anchors: 3-D float tensor of shape [batch_size, num_anchors, 4]
          containing the generated anchors in normalized coordinates.
        If self._return_raw_detections_during_predict is True, the dictionary
        will also contain:
        7) raw_detection_boxes: a 4-D float32 tensor with shape
          [batch_size, self.max_num_proposals, 4] in normalized coordinates.
        8) raw_detection_feature_map_indices: a 3-D int32 tensor with shape
          [batch_size, self.max_num_proposals].
    """
    if self._inplace_batchnorm_update:
      batchnorm_updates_collections = None
    else:
      batchnorm_updates_collections = tf.GraphKeys.UPDATE_OPS
    if self._feature_extractor.is_keras_model:
      feature_maps = self._feature_extractor(preprocessed_inputs)
    else:
      with slim.arg_scope([slim.batch_norm],
                          is_training=(self._is_training and
                                       not self._freeze_batchnorm),
                          updates_collections=batchnorm_updates_collections):
        with tf.variable_scope(None, self._extract_features_scope,
                               [preprocessed_inputs]):
          feature_maps = self._feature_extractor.extract_features(
              preprocessed_inputs)

    feature_map_spatial_dims = self._get_feature_map_spatial_dims(
        feature_maps)
    image_shape = shape_utils.combined_static_and_dynamic_shape(
        preprocessed_inputs)

    # print(self._anchor_generator.generate)
    # <bound method AnchorGenerator.generate of <object_detection.anchor_generators.multiscale_grid_anchor_generator.MultiscaleGridAnchorGenerator object at 0x7fe46c5fb0a0>>

    from typing import List, Tuple

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L585-L588
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L38-L45
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L30-L152
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/anchor_generator.py#L81-L112
    def _anchor_generate(feature_map_shape_list: List[Tuple[int]], im_height: int, im_width: int) -> box_list.BoxList:
      # TODO: find the source of anchor grid info
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L117
      anchor_grid_info = [
                          {'level': 3, 'info': [[1.0, 1.2599210498948732, 1.5874010519681994], [1.0, 2.0, 0.5], [32.0, 32.0], [8, 8]]}, 
                          {'level': 4, 'info': [[1.0, 1.2599210498948732, 1.5874010519681994], [1.0, 2.0, 0.5], [64.0, 64.0], [16, 16]]}, 
                          {'level': 5, 'info': [[1.0, 1.2599210498948732, 1.5874010519681994], [1.0, 2.0, 0.5], [128.0, 128.0], [32, 32]]}, 
                          {'level': 6, 'info': [[1.0, 1.2599210498948732, 1.5874010519681994], [1.0, 2.0, 0.5], [256.0, 256.0], [64, 64]]}, 
                          {'level': 7, 'info': [[1.0, 1.2599210498948732, 1.5874010519681994], [1.0, 2.0, 0.5], [512.0, 512.0], [128, 128]]}
                          ]
      anchor_grid_list = []
      for feat_shape, grid_info in zip(feature_map_shape_list, anchor_grid_info):
        level = grid_info['level']
        stride = 2**level
        scales, aspect_ratios, base_anchor_size, anchor_stride = grid_info['info']
        feat_h = feat_shape[0]
        feat_w = feat_shape[1]
        anchor_offset = [0, 0]

        if im_height % 2.0**level == 0 or im_height == 1:
          anchor_offset[0] = stride / 2.0
        if im_width % 2.0**level == 0 or im_width == 1:
          anchor_offset[1] = stride / 2.0

        from object_detection.anchor_generators import grid_anchor_generator

        (anchor_grid,) = _grid_anchor_generator(feature_map_shape_list=[(feat_h, feat_w)],
                                                scales=scales, aspect_ratios=aspect_ratios, base_anchor_size=base_anchor_size,
                                                anchor_stride=anchor_stride,anchor_offset=anchor_offset)

        # TODO: find the source of normalize_coordinates
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L142
        # normalize_coordinates = True
        # check_range = False # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L845
        anchor_grid = _to_normalized_coordinates(
            anchor_grid, im_height, im_width)
        anchor_grid_list.append(anchor_grid)
      return anchor_grid_list

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L134
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L30-L137
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L82-L137
    def _grid_anchor_generator(feature_map_shape_list: List[Tuple[int, int]], scales: List[float], aspect_ratios: List[float], 
                               base_anchor_size: List[float] , anchor_stride: List[int], 
                               anchor_offset: List[int]):
      grid_height, grid_width = feature_map_shape_list[0]
      scales_grid, aspect_ratios_grid = _meshgrid(scales, aspect_ratios)
      scales_grid = tf.reshape(scales_grid, [-1])
      aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
      anchors = _tile_anchors(grid_height,
                              grid_width,
                              scales_grid,
                              aspect_ratios_grid,
                              base_anchor_size,
                              anchor_stride,
                              anchor_offset)

      num_anchors = anchors.num_boxes_static()
      anchor_indices = tf.zeros([num_anchors])
      anchors.add_field('feature_map_index', anchor_indices)
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
    def _expanded_shape(orig_shape, start_dim, num_dims):
      start_dim = np.expand_dims(start_dim, 0)
      # TODO: numpy impl. of tf.slice?
      before = tf.slice(orig_shape, [0], start_dim)
      add_shape = np.ones(np.reshape(num_dims, [1])).astype(np.int32)
      # TODO: numpy impl. of tf.slice?
      after = tf.slice(orig_shape, start_dim, [-1])
      new_shape = np.concatenate([before, add_shape, after], 0)
      return new_shape

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L124-L130
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L140-L199
    def _tile_anchors(grid_height: int,
                      grid_width: int,
                      scales: List[float],
                      aspect_ratios: List[float],
                      base_anchor_size: List[float],
                      anchor_stride: List[int],
                      anchor_offset: List[int]) -> box_list.BoxList:
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
      bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
      return box_list.BoxList(tf.convert_to_tensor(bbox_corners))
    
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/grid_anchor_generator.py#L202-L213
    def _center_size_bbox_to_corners_bbox(centers: np.array, sizes: np.array) -> np.array:
      return np.concatenate([centers - .5 * sizes, centers + .5 * sizes], 1)
    
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/anchor_generators/multiscale_grid_anchor_generator.py#L148-L149
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L844-L878
    def _to_normalized_coordinates(boxlist: box_list.BoxList, height: int, width:int) -> box_list.BoxList:
      return _scale(boxlist, 1 / height, 1 / width)
    
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L878
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_list_ops.py#L82-L105
    def _scale(boxlist: box_list.BoxList, y_scale: float, x_scale: float, scope=None) -> box_list.BoxList:
        y_min, x_min, y_max, x_max = np.split(boxlist.get(), 4, axis=1)
        y_min = y_scale * y_min
        y_max = y_scale * y_max
        x_min = x_scale * x_min
        x_max = x_scale * x_max
        scaled_boxlist = box_list.BoxList(tf.convert_to_tensor(np.concatenate((y_min, x_min, y_max, x_max), axis=1)))
        return box_list_ops._copy_extra_fields(scaled_boxlist, boxlist)
    
    # replaced with np anchor generator
    # boxlist_list = self._anchor_generator.generate(
    #     feature_map_spatial_dims,
    #     im_height=image_shape[1],
    #     im_width=image_shape[2])
    boxlist_list = _anchor_generate(feature_map_spatial_dims, im_height=image_shape[1], im_width=image_shape[2])
    self._anchors = box_list_ops.concatenate(boxlist_list)
    if self._box_predictor.is_keras_model:
      predictor_results_dict = self._box_predictor(feature_maps)
    else:
      with slim.arg_scope([slim.batch_norm],
                          is_training=(self._is_training and
                                       not self._freeze_batchnorm),
                          updates_collections=batchnorm_updates_collections):
        predictor_results_dict = self._box_predictor.predict(
            feature_maps, self._anchor_generator.num_anchors_per_location())
    predictions_dict = {
        'preprocessed_inputs':
            preprocessed_inputs,
        'feature_maps':
            feature_maps,
        'anchors':
            self._anchors.get(),
        'final_anchors':
            tf.tile(
                tf.expand_dims(self._anchors.get(), 0), [image_shape[0], 1, 1])
    }
    for prediction_key, prediction_list in iter(predictor_results_dict.items()):
      prediction = tf.concat(prediction_list, axis=1)
      if (prediction_key == 'box_encodings' and prediction.shape.ndims == 4 and
          prediction.shape[2] == 1):
        prediction = tf.squeeze(prediction, axis=2)
      predictions_dict[prediction_key] = prediction
    if self._return_raw_detections_during_predict:
      predictions_dict.update(self._raw_detections_and_feature_map_inds(
          predictions_dict['box_encodings'], boxlist_list))
    self._batched_prediction_tensor_names = [x for x in predictions_dict
                                             if x != 'anchors']
    return predictions_dict

  def _raw_detections_and_feature_map_inds(self, box_encodings, boxlist_list):
    anchors = self._anchors.get()
    raw_detection_boxes, _ = self._batch_decode(box_encodings, anchors)
    batch_size, _, _ = shape_utils.combined_static_and_dynamic_shape(
        raw_detection_boxes)
    feature_map_indices = (
        self._anchor_generator.anchor_index_to_feature_map_index(boxlist_list))
    feature_map_indices_batched = tf.tile(
        tf.expand_dims(feature_map_indices, 0),
        multiples=[batch_size, 1])
    return {
        fields.PredictionFields.raw_detection_boxes: raw_detection_boxes,
        fields.PredictionFields.raw_detection_feature_map_indices:
            feature_map_indices_batched
    }

  def _get_feature_map_spatial_dims(self, feature_maps):
    """Return list of spatial dimensions for each feature map in a list.

    Args:
      feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].

    Returns:
      a list of pairs (height, width) for each feature map in feature_maps
    """
    feature_map_shapes = [
        shape_utils.combined_static_and_dynamic_shape(
            feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]

  def postprocess(self, prediction_dict, true_image_shapes):
    """Converts prediction tensors to final detections.

    This function converts raw predictions tensors to final detection results by
    slicing off the background class, decoding box predictions and applying
    non max suppression and clipping to the image window.

    See base class for output format conventions.  Note also that by default,
    scores are to be interpreted as logits, but if a score_conversion_fn is
    used, then scores are remapped (and may thus have a different
    interpretation).

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) preprocessed_inputs: a [batch, height, width, channels] image
          tensor.
        2) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        3) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions.
        4) mask_predictions: (optional) a 5-D float tensor of shape
          [batch_size, num_anchors, q, mask_height, mask_width]. `q` can be
          either number of classes or 1 depending on whether a separate mask is
          predicted per class.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros. Or None, if the clip window should cover the full image.

    Returns:
      detections: a dictionary containing the following fields
        detection_boxes: [batch, max_detections, 4] tensor with post-processed
          detection boxes.
        detection_scores: [batch, max_detections] tensor with scalar scores for
          post-processed detection boxes.
        detection_multiclass_scores: [batch, max_detections,
          num_classes_with_background] tensor with class score distribution for
          post-processed detection boxes including background class if any.
        detection_classes: [batch, max_detections] tensor with classes for
          post-processed detection classes.
        detection_keypoints: [batch, max_detections, num_keypoints, 2] (if
          encoded in the prediction_dict 'box_encodings')
        detection_masks: [batch_size, max_detections, mask_height, mask_width]
          (optional)
        num_detections: [batch]
        raw_detection_boxes: [batch, total_detections, 4] tensor with decoded
          detection boxes before Non-Max Suppression.
        raw_detection_score: [batch, total_detections,
          num_classes_with_background] tensor of multi-class scores for raw
          detection boxes.
    Raises:
      ValueError: if prediction_dict does not contain `box_encodings` or
        `class_predictions_with_background` fields.
    """
    if ('box_encodings' not in prediction_dict or
        'class_predictions_with_background' not in prediction_dict):
      raise ValueError('prediction_dict does not contain expected entries.')
    if 'anchors' not in prediction_dict:
      prediction_dict['anchors'] = self.anchors.get()
    with tf.name_scope('Postprocessor'):
      preprocessed_images = prediction_dict['preprocessed_inputs']
      box_encodings = prediction_dict['box_encodings']
      box_encodings = tf.identity(box_encodings, 'raw_box_encodings')
      class_predictions_with_background = (
          prediction_dict['class_predictions_with_background'])
      # print(f'{self._batch_decode}')
      # self._batch_decode=<bound method SSDMetaArch._batch_decode of <object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch object at 0x7fa8a82edfd0>>
      detection_boxes, detection_keypoints = self._batch_decode(
          box_encodings, prediction_dict['anchors'])
      detection_boxes = tf.identity(detection_boxes, 'raw_box_locations')
      detection_boxes = tf.expand_dims(detection_boxes, axis=2)

      # replaced with numpy sigmoid
      # detection_scores_with_background = self._score_conversion_fn(
      #     class_predictions_with_background)
      #
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L727-L728](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L727-L728)
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L135](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L135)
      # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/post_processing_builder.py#L60-L62](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L60-L62)
      # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/post_processing_builder.py#L140-L141](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L140-L141)
      # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/builders/post_processing_builder.py#L112-L119](https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L112-L119)
      def _sigmoid(x: np.ndarray) -> np.ndarray:
        # pylint: disable=invalid-name
        return 1 / (1 + np.exp(-x))
      detection_scores_with_background = tf.convert_to_tensor(_sigmoid(class_predictions_with_background.numpy()))

      detection_scores = tf.identity(detection_scores_with_background,
                                     'raw_box_scores')
      if self._add_background_class or self._explicit_background_class:
        detection_scores = tf.slice(detection_scores, [0, 0, 1], [-1, -1, -1])
      additional_fields = None

      batch_size = (
          shape_utils.combined_static_and_dynamic_shape(preprocessed_images)[0])

      if 'feature_maps' in prediction_dict:
        feature_map_list = []
        for feature_map in prediction_dict['feature_maps']:
          feature_map_list.append(tf.reshape(feature_map, [batch_size, -1]))
        box_features = tf.concat(feature_map_list, 1)
        box_features = tf.identity(box_features, 'raw_box_features')
      additional_fields = {
          'multiclass_scores': detection_scores_with_background
      }
      if self._anchors is not None:
        num_boxes = (self._anchors.num_boxes_static() or
                     self._anchors.num_boxes())
        anchor_indices = tf.range(num_boxes)
        batch_anchor_indices = tf.tile(
            tf.expand_dims(anchor_indices, 0), [batch_size, 1])
        # All additional fields need to be float.
        additional_fields.update({
            'anchor_indices': tf.cast(batch_anchor_indices, tf.float32),
        })
      if detection_keypoints is not None:
        detection_keypoints = tf.identity(
            detection_keypoints, 'raw_keypoint_locations')
        additional_fields[fields.BoxListFields.keypoints] = detection_keypoints

      # print(f'{self._compute_clip_window}')
      # <bound method SSDMetaArch._compute_clip_window of <object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch object at 0x7f257016ea60>>
      from typing import Tuple

      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L767-L768
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L486-L523
      def _compute_clip_window(preprocessed_images: np.ndarray, true_image_shapes: Tuple[int]) -> np.ndarray:
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
                np.zeros_like(true_widths), true_heights / padded_height,
                true_widths / padded_width
            ],
            axis=1).astype(np.float32)

      # print(f'{self._non_max_suppression_fn=}')
      # self._non_max_suppression_fn=functools.partial(<function batch_multiclass_non_max_suppression at 0x7fae07ebb9d0>, 
      #  score_thresh=9.99999993922529e-09, iou_thresh=0.5, max_size_per_class=100, max_total_size=100, 
      #  use_static_shapes=False, use_class_agnostic_nms=False, max_classes_per_detection=1, soft_nms_sigma=0.0, 
      #  use_partitioned_nms=False, use_combined_nms=False, change_coordinate_frame=True, use_hard_nms=False, use_cpu_nms=False)

      # replaced with numpy _non_max_supression_fn
      # (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
      #  nmsed_additional_fields,
      #  num_detections) = self._non_max_suppression_fn(
      #      detection_boxes,
      #      detection_scores,
      #      clip_window=tf.convert_to_tensor(_compute_clip_window(
      #          preprocessed_images.numpy(), tuple(true_image_shapes))),
      #      additional_fields=additional_fields,
      #      masks=prediction_dict.get('mask_predictions'))
      
      import collections
      from typing import Optional, Dict
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L878-L1276
      def _batch_multiclass_non_max_suppression(boxes: np.array,
                                    scores: np.array,
                                    score_thresh: float,
                                    iou_thresh: float,
                                    max_size_per_class: int,
                                    max_total_size: int=0,
                                    clip_window: Optional[np.array]=None,
                                    change_coordinate_frame: bool=False,
                                    num_valid_boxes: Optional[int]=None,
                                    masks: Optional[np.array]=None,
                                    additional_fields: Optional[Dict[str, np.array]]=None,
                                    soft_nms_sigma: float=0.0,
                                    scope=None,
                                    use_static_shapes: bool=False,
                                    use_partitioned_nms: bool=False,
                                    parallel_iterations: int=32,
                                    use_class_agnostic_nms: bool=False,
                                    max_classes_per_detection: int=1,
                                    use_dynamic_map_fn: bool=False,
                                    use_combined_nms: bool=False,
                                    use_hard_nms: bool=False,
                                    use_cpu_nms: bool=False) -> Tuple[np.array, np.array, np.array, np.array, Dict[str, np.array], np.array]:
        q = boxes.shape[2]
        ordered_additional_fields = collections.OrderedDict(sorted(additional_fields.items(), key=lambda item: item[0]))
        
        boxes_shape = boxes.shape
        batch_size = boxes_shape[0]
        num_anchors = boxes_shape[1]

        num_valid_boxes = np.ones([batch_size], dtype=np.int32) * num_anchors
        masks_shape = np.stack([batch_size, num_anchors, q, 1, 1])
        masks = np.zeros(masks_shape)

        num_additional_fields = len(ordered_additional_fields)
        nms_configs = {'score_thresh': score_thresh, 
                      'iou_thresh': iou_thresh, 
                      'max_size_per_class': max_size_per_class, 
                      'max_total_size': max_total_size, 
                      'change_coordinate_frame': change_coordinate_frame, 
                      # 'use_static_shapes': use_static_shapes, 
                      'use_partitioned_nms': use_partitioned_nms, 
                      'soft_nms_sigma': soft_nms_sigma, 
                      'use_hard_nms': use_hard_nms, 
                      'use_cpu_nms': use_cpu_nms}
        # for loop impl. of tf.map_fn
        # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L1244-L1249
        batch_outputs = [_single_image_nms_fn(
                          per_image_boxes=boxes[i],
                          per_image_scores=scores[i],
                          per_image_masks=masks[i],
                          per_image_clip_window=clip_window[i],
                          per_image_additional_fields=list(map(dict, zip(*[[(k, v) for v in value] for k, value in ordered_additional_fields.items()])))[i],
                          per_image_num_valid_boxes=num_valid_boxes[i], 
                          **nms_configs,
                          )
                          for i in range(batch_size)]
        # convert List[List[np.array]] to List[np.array]
        batch_outputs = list(map(np.stack, np.stack(batch_outputs, axis=1)))
        batch_nmsed_boxes = batch_outputs[0]
        batch_nmsed_scores = batch_outputs[1]
        batch_nmsed_classes = batch_outputs[2]
        batch_nmsed_masks = batch_outputs[3]
        batch_nmsed_values = batch_outputs[4:-1]

        batch_nmsed_additional_fields = {}
        batch_nmsed_keys = list(ordered_additional_fields.keys())
        for i in range(len(batch_nmsed_keys)):
          batch_nmsed_additional_fields[
              batch_nmsed_keys[i]] = batch_nmsed_values[i]

        batch_num_detections = batch_outputs[-1]
        batch_nmsed_masks = None

        return (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes,
                batch_nmsed_masks, batch_nmsed_additional_fields,
                batch_num_detections)
          
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L1099-L1232
      def _single_image_nms_fn(per_image_boxes: np.array,
                               per_image_scores: np.array,
                               per_image_masks: np.array,
                               per_image_clip_window: np.array,
                               per_image_additional_fields: Dict[str, np.array],
                               per_image_num_valid_boxes: int,
                               **kwargs) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, int]:
        q = per_image_boxes.shape[1]
        num_classes = per_image_scores.shape[1]
                
        per_image_boxes = np.reshape(
            # TODO: find numpy func corresponding to tf.slice
            tf.slice(per_image_boxes, 3 * [0], np.stack([per_image_num_valid_boxes, -1, -1])), [-1, q, 4])
        per_image_scores = np.reshape(
            # TODO: find numpy func corresponding to tf.slice
            tf.slice(per_image_scores, [0, 0], np.stack([per_image_num_valid_boxes, -1])),
            [-1, num_classes])
        per_image_masks = np.reshape(
            # TODO: find numpy func corresponding to tf.slice
            tf.slice(per_image_masks, 4 * [0], np.stack([per_image_num_valid_boxes, -1, -1, -1])),
            [-1, q, int(per_image_masks.shape[2]), int(per_image_masks.shape[3])])
        for key, array in per_image_additional_fields.items():
          additional_field_shape = array.shape
          additional_field_dim = len(additional_field_shape)
          per_image_additional_fields[key] = np.reshape(
              # TODO: find numpy func corresponding to tf.slice
              tf.slice(
                  per_image_additional_fields[key],
                  additional_field_dim * [0],
                  np.stack([per_image_num_valid_boxes] +
                            (additional_field_dim - 1) * [-1])), [-1] + [
                                int(dim)
                                for dim in additional_field_shape[1:]
                          ])
        nmsed_boxlist, num_valid_nms_boxes = _multiclass_non_max_suppression(
          boxes=per_image_boxes,
          scores=per_image_scores,
          clip_window=per_image_clip_window,
          masks=per_image_masks,
          additional_fields=per_image_additional_fields,
          **kwargs)
        
        max_total_size = kwargs['max_total_size']
        nmsed_boxlist = box_list_ops.pad_or_clip_box_list(
            nmsed_boxlist, max_total_size)
        num_detections = num_valid_nms_boxes
        nmsed_boxes = nmsed_boxlist.get()
        nmsed_scores = nmsed_boxlist.get_field(fields.BoxListFields.scores)
        nmsed_classes = nmsed_boxlist.get_field(fields.BoxListFields.classes)
        nmsed_masks = nmsed_boxlist.get_field(fields.BoxListFields.masks)
        nmsed_additional_fields = []
        
        for key in sorted(per_image_additional_fields.keys()):
          nmsed_additional_fields.append(nmsed_boxlist.get_field(key).numpy())
        
        return ([nmsed_boxes.numpy(), nmsed_scores.numpy(), nmsed_classes.numpy(), nmsed_masks.numpy()] +
                nmsed_additional_fields + [num_detections])
    
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L1200-L1215
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L422-L651
      def _multiclass_non_max_suppression(boxes: np.array,
                                          scores: np.array,
                                          score_thresh: float,
                                          iou_thresh: float,
                                          max_size_per_class: int,
                                          max_total_size: int=0,
                                          clip_window: Optional[np.array]=None,
                                          change_coordinate_frame: bool=False,
                                          masks: Optional[np.array]=None,
                                          boundaries: Optional[np.array]=None,
                                          pad_to_max_output_size: bool=False,
                                          use_partitioned_nms: bool=False,
                                          additional_fields: Optional[Dict[str, np.array]]=None,
                                          soft_nms_sigma: float=0.0,
                                          use_hard_nms: bool=False,
                                          use_cpu_nms: bool=False) -> Tuple[box_list.BoxList, int]:
        
        num_scores = scores.shape[0]
        num_classes = scores.shape[1]
        
        selected_boxes_list = []
        num_valid_nms_boxes_cumulative = np.array(0, dtype=np.int64)
        per_class_boxes_list = np.moveaxis(boxes, 0, 1)
        per_class_masks_list = np.moveaxis(masks, 0, 1)

        boxes_ids = (range(num_classes) if len(per_class_boxes_list) > 1 else [0] * num_classes)
        for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
          per_class_boxes = per_class_boxes_list[boxes_idx]
          boxlist_and_class_scores = box_list.BoxList(tf.convert_to_tensor(per_class_boxes))
          class_scores = np.reshape(
              # TODO: find numpy func corresponding to tf.slice
              tf.slice(scores, [0, class_idx], np.stack([num_scores, 1])), [-1])

          boxlist_and_class_scores.add_field(fields.BoxListFields.scores, class_scores)
          per_class_masks = per_class_masks_list[boxes_idx]
          boxlist_and_class_scores.add_field(fields.BoxListFields.masks,
                                            per_class_masks)        
            
          for key, tensor in additional_fields.items():
            boxlist_and_class_scores.add_field(key, tensor)

          nms_result = None
          selected_scores = None
          
          max_selection_size = np.minimum(max_size_per_class, boxlist_and_class_scores.num_boxes())
          # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L583-L589
          # https://github.com/tensorflow/tensorflow/blob/v2.10.1/tensorflow/python/ops/image_ops_impl.py#L3804-L3891](https://github.com/tensorflow/tensorflow/blob/v2.10.1/tensorflow/python/ops/image_ops_impl.py#L3804-L3891)
          # https://github.com/tensorflow/tensorflow/blob/c7adce4cb2293b66a96b811a0dcdcfb7e361c23f/tensorflow/core/kernels/image/non_max_suppression_op.cc#L829-L907
          # https://github.com/tensorflow/tensorflow/blob/c7adce4cb2293b66a96b811a0dcdcfb7e361c23f/tensorflow/core/kernels/image/non_max_suppression_op.cc#L194-L330
          # TODO: CPP impl?
          (selected_indices, selected_scores
          ) = tf.image.non_max_suppression_with_scores(
              boxlist_and_class_scores.get(),
              boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
              max_selection_size,
              iou_threshold=iou_thresh,
              score_threshold=score_thresh,
              soft_nms_sigma=soft_nms_sigma)
          num_valid_nms_boxes = selected_indices.shape[0]
          selected_indices = np.concatenate([selected_indices, np.zeros(max_selection_size-num_valid_nms_boxes, dtype=np.int32)], 0)
          selected_scores = np.concatenate([selected_scores, np.zeros(max_selection_size-num_valid_nms_boxes, dtype=np.float32)], -1)
          nms_result = box_list_ops.gather(boxlist_and_class_scores, tf.convert_to_tensor(selected_indices))
        
          valid_nms_boxes_indices = np.less(np.arange(max_selection_size), num_valid_nms_boxes)
          
          nms_result.add_field(
              fields.BoxListFields.scores,
              tf.convert_to_tensor(np.where(valid_nms_boxes_indices, selected_scores, -1*np.ones(max_selection_size)))
              )
          num_valid_nms_boxes_cumulative += num_valid_nms_boxes

          nms_result.add_field(
              fields.BoxListFields.classes, 
              tf.convert_to_tensor((np.zeros_like(nms_result.get_field(fields.BoxListFields.scores)) + class_idx))
              )
          selected_boxes_list.append(nms_result)
        
        selected_boxes = box_list_ops.concatenate(selected_boxes_list)
        sorted_boxes = box_list_ops.sort_by_field(selected_boxes, fields.BoxListFields.scores)

        sorted_boxes, num_valid_nms_boxes_cumulative = _clip_window_prune_boxes(
                                                        sorted_boxes, clip_window, pad_to_max_output_size,
                                                        change_coordinate_frame)
        
        max_total_size = np.minimum(max_total_size, sorted_boxes.num_boxes())
        sorted_boxes = box_list_ops.gather(sorted_boxes, tf.convert_to_tensor(np.arange(max_total_size)))
        num_valid_nms_boxes_cumulative = np.where(max_total_size > num_valid_nms_boxes_cumulative,
                                                  num_valid_nms_boxes_cumulative, max_total_size)
      
        sorted_boxes = box_list_ops.gather(sorted_boxes, tf.convert_to_tensor(np.arange(num_valid_nms_boxes_cumulative)))
        return sorted_boxes, num_valid_nms_boxes_cumulative
      
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L636-L638
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/post_processing.py#L345-L388
      def _clip_window_prune_boxes(sorted_boxes: box_list.BoxList, clip_window: np.array, 
                                   pad_to_max_output_size: bool, change_coordinate_frame: bool) -> Tuple[box_list.BoxList, int]:
        
        sorted_boxes = box_list_ops.clip_to_window(sorted_boxes, tf.convert_to_tensor(clip_window),
                                                   filter_nonoverlapping=not pad_to_max_output_size)
        
        sorted_boxes_size = sorted_boxes.get().numpy().shape[0]
        non_zero_box_area = box_list_ops.area(sorted_boxes).numpy().astype(np.bool)
        sorted_boxes_scores = np.where(
            non_zero_box_area, sorted_boxes.get_field(fields.BoxListFields.scores).numpy(),
            -1 * np.ones(sorted_boxes_size))
        sorted_boxes.add_field(fields.BoxListFields.scores, tf.convert_to_tensor(sorted_boxes_scores))
        num_valid_nms_boxes_cumulative = np.sum(np.greater_equal(sorted_boxes_scores, 0).astype(np.int32))
        sorted_boxes = box_list_ops.sort_by_field(sorted_boxes,fields.BoxListFields.scores)
        sorted_boxes = box_list_ops.change_coordinate_frame(sorted_boxes, tf.convert_to_tensor(clip_window))
        return sorted_boxes, num_valid_nms_boxes_cumulative

      import functools
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L764-L770
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L129-L134
      # TODO: where are the default values not specified in configuration?
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L58-L59
      # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/builders/post_processing_builder.py#L70-L109
      _non_max_suppression_fn = functools.partial(
        _batch_multiclass_non_max_suppression,
        score_thresh=9.99999993922529e-09, iou_thresh=0.5, 
        max_size_per_class=100, max_total_size=100, 
        use_static_shapes=False, use_class_agnostic_nms=False, 
        max_classes_per_detection=1, soft_nms_sigma=0.0, 
        use_partitioned_nms=False, use_combined_nms=False, 
        change_coordinate_frame=True, use_hard_nms=False, use_cpu_nms=False
        )
      
      (nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks,
       nmsed_additional_fields,
       num_detections) = _non_max_suppression_fn(
           detection_boxes,
           detection_scores,
           clip_window=_compute_clip_window(preprocessed_images.numpy(), tuple(true_image_shapes)),
           additional_fields=additional_fields,
           masks=prediction_dict.get('mask_predictions'))

      nmsed_boxes = tf.convert_to_tensor(nmsed_boxes) 
      nmsed_scores = tf.convert_to_tensor(nmsed_scores)
      nmsed_classes = tf.convert_to_tensor(nmsed_classes)
      nmsed_additional_fields = {k: tf.convert_to_tensor(v) for k, v in nmsed_additional_fields.items()}
      num_detections = tf.convert_to_tensor(num_detections)
      
      detection_dict = {
          fields.DetectionResultFields.detection_boxes:
              nmsed_boxes,
          fields.DetectionResultFields.detection_scores:
              nmsed_scores,
          fields.DetectionResultFields.detection_classes:
              nmsed_classes,
          fields.DetectionResultFields.num_detections:
              tf.cast(num_detections, dtype=tf.float32),
          fields.DetectionResultFields.raw_detection_boxes:
              tf.squeeze(detection_boxes, axis=2),
          fields.DetectionResultFields.raw_detection_scores:
              detection_scores_with_background
      }
      if (nmsed_additional_fields is not None and
          fields.InputDataFields.multiclass_scores in nmsed_additional_fields):
        detection_dict[
            fields.DetectionResultFields.detection_multiclass_scores] = (
                nmsed_additional_fields[
                    fields.InputDataFields.multiclass_scores])
      if (nmsed_additional_fields is not None and
          'anchor_indices' in nmsed_additional_fields):
        detection_dict.update({
            fields.DetectionResultFields.detection_anchor_indices:
                tf.cast(nmsed_additional_fields['anchor_indices'], tf.int32),
        })
      if (nmsed_additional_fields is not None and
          fields.BoxListFields.keypoints in nmsed_additional_fields):
        detection_dict[fields.DetectionResultFields.detection_keypoints] = (
            nmsed_additional_fields[fields.BoxListFields.keypoints])
      if nmsed_masks is not None:
        detection_dict[
            fields.DetectionResultFields.detection_masks] = nmsed_masks
      return detection_dict

  def loss(self, prediction_dict, true_image_shapes, scope=None):
    """Compute scalar loss tensors with respect to provided groundtruth.

    Calling this function requires that groundtruth tensors have been
    provided via the provide_groundtruth function.

    Args:
      prediction_dict: a dictionary holding prediction tensors with
        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        2) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors. Note that this tensor *includes*
          background class predictions.
      true_image_shapes: int32 tensor of shape [batch, 3] where each row is
        of the form [height, width, channels] indicating the shapes
        of true images in the resized images, as resized images can be padded
        with zeros.
      scope: Optional scope name.

    Returns:
      a dictionary mapping loss keys (`localization_loss` and
        `classification_loss`) to scalar tensors representing corresponding loss
        values.
    """
    with tf.name_scope(scope, 'Loss', prediction_dict.values()):
      keypoints = None
      if self.groundtruth_has_field(fields.BoxListFields.keypoints):
        keypoints = self.groundtruth_lists(fields.BoxListFields.keypoints)
      weights = None
      if self.groundtruth_has_field(fields.BoxListFields.weights):
        weights = self.groundtruth_lists(fields.BoxListFields.weights)
      confidences = None
      if self.groundtruth_has_field(fields.BoxListFields.confidences):
        confidences = self.groundtruth_lists(fields.BoxListFields.confidences)
      (batch_cls_targets, batch_cls_weights, batch_reg_targets,
       batch_reg_weights, batch_match) = self._assign_targets(
           self.groundtruth_lists(fields.BoxListFields.boxes),
           self.groundtruth_lists(fields.BoxListFields.classes),
           keypoints, weights, confidences)
      match_list = [matcher.Match(match) for match in tf.unstack(batch_match)]
      if self._add_summaries:
        self._summarize_target_assignment(
            self.groundtruth_lists(fields.BoxListFields.boxes), match_list)

      if self._random_example_sampler:
        batch_cls_per_anchor_weights = tf.reduce_mean(
            batch_cls_weights, axis=-1)
        batch_sampled_indicator = tf.cast(
            shape_utils.static_or_dynamic_map_fn(
                self._minibatch_subsample_fn,
                [batch_cls_targets, batch_cls_per_anchor_weights],
                dtype=tf.bool,
                parallel_iterations=self._parallel_iterations,
                back_prop=True), dtype=tf.float32)
        batch_reg_weights = tf.multiply(batch_sampled_indicator,
                                        batch_reg_weights)
        batch_cls_weights = tf.multiply(
            tf.expand_dims(batch_sampled_indicator, -1),
            batch_cls_weights)

      losses_mask = None
      if self.groundtruth_has_field(fields.InputDataFields.is_annotated):
        losses_mask = tf.stack(self.groundtruth_lists(
            fields.InputDataFields.is_annotated))


      location_losses = self._localization_loss(
          prediction_dict['box_encodings'],
          batch_reg_targets,
          ignore_nan_targets=True,
          weights=batch_reg_weights,
          losses_mask=losses_mask)

      cls_losses = self._classification_loss(
          prediction_dict['class_predictions_with_background'],
          batch_cls_targets,
          weights=batch_cls_weights,
          losses_mask=losses_mask)

      if self._expected_loss_weights_fn:
        # Need to compute losses for assigned targets against the
        # unmatched_class_label as well as their assigned targets.
        # simplest thing (but wasteful) is just to calculate all losses
        # twice
        batch_size, num_anchors, num_classes = batch_cls_targets.get_shape()
        unmatched_targets = tf.ones([batch_size, num_anchors, 1
                                    ]) * self._unmatched_class_label

        unmatched_cls_losses = self._classification_loss(
            prediction_dict['class_predictions_with_background'],
            unmatched_targets,
            weights=batch_cls_weights,
            losses_mask=losses_mask)

        if cls_losses.get_shape().ndims == 3:
          batch_size, num_anchors, num_classes = cls_losses.get_shape()
          cls_losses = tf.reshape(cls_losses, [batch_size, -1])
          unmatched_cls_losses = tf.reshape(unmatched_cls_losses,
                                            [batch_size, -1])
          batch_cls_targets = tf.reshape(
              batch_cls_targets, [batch_size, num_anchors * num_classes, -1])
          batch_cls_targets = tf.concat(
              [1 - batch_cls_targets, batch_cls_targets], axis=-1)

          location_losses = tf.tile(location_losses, [1, num_classes])

        foreground_weights, background_weights = (
            self._expected_loss_weights_fn(batch_cls_targets))

        cls_losses = (
            foreground_weights * cls_losses +
            background_weights * unmatched_cls_losses)

        location_losses *= foreground_weights

        classification_loss = tf.reduce_sum(cls_losses)
        localization_loss = tf.reduce_sum(location_losses)
      elif self._hard_example_miner:
        cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
        (localization_loss, classification_loss) = self._apply_hard_mining(
            location_losses, cls_losses, prediction_dict, match_list)
        if self._add_summaries:
          self._hard_example_miner.summarize()
      else:
        cls_losses = ops.reduce_sum_trailing_dimensions(cls_losses, ndims=2)
        localization_loss = tf.reduce_sum(location_losses)
        classification_loss = tf.reduce_sum(cls_losses)

      # Optionally normalize by number of positive matches
      normalizer = tf.constant(1.0, dtype=tf.float32)
      if self._normalize_loss_by_num_matches:
        normalizer = tf.maximum(tf.cast(tf.reduce_sum(batch_reg_weights),
                                        dtype=tf.float32),
                                1.0)

      localization_loss_normalizer = normalizer
      if self._normalize_loc_loss_by_codesize:
        localization_loss_normalizer *= self._box_coder.code_size
      localization_loss = tf.multiply((self._localization_loss_weight /
                                       localization_loss_normalizer),
                                      localization_loss,
                                      name='localization_loss')
      classification_loss = tf.multiply((self._classification_loss_weight /
                                         normalizer), classification_loss,
                                        name='classification_loss')

      loss_dict = {
          'Loss/localization_loss': localization_loss,
          'Loss/classification_loss': classification_loss
      }


    return loss_dict

  def _minibatch_subsample_fn(self, inputs):
    """Randomly samples anchors for one image.

    Args:
      inputs: a list of 2 inputs. First one is a tensor of shape [num_anchors,
        num_classes] indicating targets assigned to each anchor. Second one
        is a tensor of shape [num_anchors] indicating the class weight of each
        anchor.

    Returns:
      batch_sampled_indicator: bool tensor of shape [num_anchors] indicating
        whether the anchor should be selected for loss computation.
    """
    cls_targets, cls_weights = inputs
    if self._add_background_class:
      # Set background_class bits to 0 so that the positives_indicator
      # computation would not consider background class.
      background_class = tf.zeros_like(tf.slice(cls_targets, [0, 0], [-1, 1]))
      regular_class = tf.slice(cls_targets, [0, 1], [-1, -1])
      cls_targets = tf.concat([background_class, regular_class], 1)
    positives_indicator = tf.reduce_sum(cls_targets, axis=1)
    return self._random_example_sampler.subsample(
        tf.cast(cls_weights, tf.bool),
        batch_size=None,
        labels=tf.cast(positives_indicator, tf.bool))

  def _summarize_anchor_classification_loss(self, class_ids, cls_losses):
    positive_indices = tf.where(tf.greater(class_ids, 0))
    positive_anchor_cls_loss = tf.squeeze(
        tf.gather(cls_losses, positive_indices), axis=1)
    visualization_utils.add_cdf_image_summary(positive_anchor_cls_loss,
                                              'PositiveAnchorLossCDF')
    negative_indices = tf.where(tf.equal(class_ids, 0))
    negative_anchor_cls_loss = tf.squeeze(
        tf.gather(cls_losses, negative_indices), axis=1)
    visualization_utils.add_cdf_image_summary(negative_anchor_cls_loss,
                                              'NegativeAnchorLossCDF')

  def _assign_targets(self,
                      groundtruth_boxes_list,
                      groundtruth_classes_list,
                      groundtruth_keypoints_list=None,
                      groundtruth_weights_list=None,
                      groundtruth_confidences_list=None):
    """Assign groundtruth targets.

    Adds a background class to each one-hot encoding of groundtruth classes
    and uses target assigner to obtain regression and classification targets.

    Args:
      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
        containing coordinates of the groundtruth boxes.
          Groundtruth boxes are provided in [y_min, x_min, y_max, x_max]
          format and assumed to be normalized and clipped
          relative to the image window with y_min <= y_max and x_min <= x_max.
      groundtruth_classes_list: a list of 2-D one-hot (or k-hot) tensors of
        shape [num_boxes, num_classes] containing the class targets with the 0th
        index assumed to map to the first non-background class.
      groundtruth_keypoints_list: (optional) a list of 3-D tensors of shape
        [num_boxes, num_keypoints, 2]
      groundtruth_weights_list: A list of 1-D tf.float32 tensors of shape
        [num_boxes] containing weights for groundtruth boxes.
      groundtruth_confidences_list: A list of 2-D tf.float32 tensors of shape
        [num_boxes, num_classes] containing class confidences for
        groundtruth boxes.

    Returns:
      batch_cls_targets: a tensor with shape [batch_size, num_anchors,
        num_classes],
      batch_cls_weights: a tensor with shape [batch_size, num_anchors],
      batch_reg_targets: a tensor with shape [batch_size, num_anchors,
        box_code_dimension]
      batch_reg_weights: a tensor with shape [batch_size, num_anchors],
      match: an int32 tensor of shape [batch_size, num_anchors], containing
        result of anchor groundtruth matching. Each position in the tensor
        indicates an anchor and holds the following meaning:
        (1) if match[x, i] >= 0, anchor i is matched with groundtruth
            match[x, i].
        (2) if match[x, i]=-1, anchor i is marked to be background .
        (3) if match[x, i]=-2, anchor i is ignored since it is not background
            and does not have sufficient overlap to call it a foreground.
    """
    groundtruth_boxlists = [
        box_list.BoxList(boxes) for boxes in groundtruth_boxes_list
    ]
    train_using_confidences = (self._is_training and
                               self._use_confidences_as_targets)
    if self._add_background_class:
      groundtruth_classes_with_background_list = [
          tf.pad(one_hot_encoding, [[0, 0], [1, 0]], mode='CONSTANT')
          for one_hot_encoding in groundtruth_classes_list
      ]
      if train_using_confidences:
        groundtruth_confidences_with_background_list = [
            tf.pad(groundtruth_confidences, [[0, 0], [1, 0]], mode='CONSTANT')
            for groundtruth_confidences in groundtruth_confidences_list
        ]
    else:
      groundtruth_classes_with_background_list = groundtruth_classes_list

    if groundtruth_keypoints_list is not None:
      for boxlist, keypoints in zip(
          groundtruth_boxlists, groundtruth_keypoints_list):
        boxlist.add_field(fields.BoxListFields.keypoints, keypoints)
    if train_using_confidences:
      return target_assigner.batch_assign_confidences(
          self._target_assigner,
          self.anchors,
          groundtruth_boxlists,
          groundtruth_confidences_with_background_list,
          groundtruth_weights_list,
          self._unmatched_class_label,
          self._add_background_class,
          self._implicit_example_weight)
    else:
      return target_assigner.batch_assign_targets(
          self._target_assigner,
          self.anchors,
          groundtruth_boxlists,
          groundtruth_classes_with_background_list,
          self._unmatched_class_label,
          groundtruth_weights_list)

  def _summarize_target_assignment(self, groundtruth_boxes_list, match_list):
    """Creates tensorflow summaries for the input boxes and anchors.

    This function creates four summaries corresponding to the average
    number (over images in a batch) of (1) groundtruth boxes, (2) anchors
    marked as positive, (3) anchors marked as negative, and (4) anchors marked
    as ignored.

    Args:
      groundtruth_boxes_list: a list of 2-D tensors of shape [num_boxes, 4]
        containing corners of the groundtruth boxes.
      match_list: a list of matcher.Match objects encoding the match between
        anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.
    """
    # TODO(rathodv): Add a test for these summaries.
    try:
      # TODO(kaftan): Integrate these summaries into the v2 style loops
      with tf.compat.v2.init_scope():
        if tf.compat.v2.executing_eagerly():
          return
    except AttributeError:
      pass

    avg_num_gt_boxes = tf.reduce_mean(
        tf.cast(
            tf.stack([tf.shape(x)[0] for x in groundtruth_boxes_list]),
            dtype=tf.float32))
    avg_num_matched_gt_boxes = tf.reduce_mean(
        tf.cast(
            tf.stack([match.num_matched_rows() for match in match_list]),
            dtype=tf.float32))
    avg_pos_anchors = tf.reduce_mean(
        tf.cast(
            tf.stack([match.num_matched_columns() for match in match_list]),
            dtype=tf.float32))
    avg_neg_anchors = tf.reduce_mean(
        tf.cast(
            tf.stack([match.num_unmatched_columns() for match in match_list]),
            dtype=tf.float32))
    avg_ignored_anchors = tf.reduce_mean(
        tf.cast(
            tf.stack([match.num_ignored_columns() for match in match_list]),
            dtype=tf.float32))

    tf.summary.scalar('AvgNumGroundtruthBoxesPerImage',
                      avg_num_gt_boxes,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumGroundtruthBoxesMatchedPerImage',
                      avg_num_matched_gt_boxes,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumPositiveAnchorsPerImage',
                      avg_pos_anchors,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumNegativeAnchorsPerImage',
                      avg_neg_anchors,
                      family='TargetAssignment')
    tf.summary.scalar('AvgNumIgnoredAnchorsPerImage',
                      avg_ignored_anchors,
                      family='TargetAssignment')

  def _apply_hard_mining(self, location_losses, cls_losses, prediction_dict,
                         match_list):
    """Applies hard mining to anchorwise losses.

    Args:
      location_losses: Float tensor of shape [batch_size, num_anchors]
        representing anchorwise location losses.
      cls_losses: Float tensor of shape [batch_size, num_anchors]
        representing anchorwise classification losses.
      prediction_dict: p a dictionary holding prediction tensors with
        1) box_encodings: 3-D float tensor of shape [batch_size, num_anchors,
          box_code_dimension] containing predicted boxes.
        2) class_predictions_with_background: 3-D float tensor of shape
          [batch_size, num_anchors, num_classes+1] containing class predictions
          (logits) for each of the anchors.  Note that this tensor *includes*
          background class predictions.
        3) anchors: (optional) 2-D float tensor of shape [num_anchors, 4].
      match_list: a list of matcher.Match objects encoding the match between
        anchors and groundtruth boxes for each image of the batch,
        with rows of the Match objects corresponding to groundtruth boxes
        and columns corresponding to anchors.

    Returns:
      mined_location_loss: a float scalar with sum of localization losses from
        selected hard examples.
      mined_cls_loss: a float scalar with sum of classification losses from
        selected hard examples.
    """
    class_predictions = prediction_dict['class_predictions_with_background']
    if self._add_background_class:
      class_predictions = tf.slice(class_predictions, [0, 0, 1], [-1, -1, -1])

    if 'anchors' not in prediction_dict:
      prediction_dict['anchors'] = self.anchors.get()
    decoded_boxes, _ = self._batch_decode(prediction_dict['box_encodings'],
                                          prediction_dict['anchors'])
    decoded_box_tensors_list = tf.unstack(decoded_boxes)
    class_prediction_list = tf.unstack(class_predictions)
    decoded_boxlist_list = []
    for box_location, box_score in zip(decoded_box_tensors_list,
                                       class_prediction_list):
      decoded_boxlist = box_list.BoxList(box_location)
      decoded_boxlist.add_field('scores', box_score)
      decoded_boxlist_list.append(decoded_boxlist)
    return self._hard_example_miner(
        location_losses=location_losses,
        cls_losses=cls_losses,
        decoded_boxlist_list=decoded_boxlist_list,
        match_list=match_list)

  def _batch_decode(self, box_encodings, anchors):
    """Decodes a batch of box encodings with respect to the anchors.

    Args:
      box_encodings: A float32 tensor of shape
        [batch_size, num_anchors, box_code_size] containing box encodings.
      anchors: A tensor of shape [num_anchors, 4].

    Returns:
      decoded_boxes: A float32 tensor of shape
        [batch_size, num_anchors, 4] containing the decoded boxes.
      decoded_keypoints: A float32 tensor of shape
        [batch_size, num_anchors, num_keypoints, 2] containing the decoded
        keypoints if present in the input `box_encodings`, None otherwise.
    """
    combined_shape = shape_utils.combined_static_and_dynamic_shape(
        box_encodings)
    batch_size = combined_shape[0]
    tiled_anchor_boxes = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])
    tiled_anchors_boxlist = box_list.BoxList(
        tf.reshape(tiled_anchor_boxes, [-1, 4]))
    # print(f'{self._box_coder=}')
    # self._box_coder=<object_detection.box_coders.faster_rcnn_box_coder.FasterRcnnBoxCoder object at 0x7f7bd89d0460>

    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L15-L22
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/meta_architectures/ssd_meta_arch.py#L1197-L1231
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/core/box_coder.py#L80-L92
    # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/box_coders/faster_rcnn_box_coder.py#L92-L118
    def _box_decode(rel_codes:np.ndarray, anchors: box_list.BoxList) -> box_list.BoxList:
        # TODO make anchors np array?
        # https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/object_detection/core/box_list.py#L161-L177
        ycenter_a, xcenter_a, ha, wa = anchors.get_center_coordinates_and_sizes()
        ycenter_a = ycenter_a.numpy()
        xcenter_a = xcenter_a.numpy()
        ha = ha.numpy()
        wa = wa.numpy()
        # scale_factors=[1.0, 1.0, 1.0, 1.0]
        # so omit https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/box_coders/faster_rcnn_box_coder.py#L105-L109
        # TODO where did the scale_factors come from?
        ty, tx, th, tw = np.moveaxis(np.transpose(rel_codes), 0, 0)
        w = np.exp(tw) * wa
        h = np.exp(th) * ha
        ycenter = ty * ha + ycenter_a
        xcenter = tx * wa + xcenter_a
        ymin = ycenter - h / 2.
        xmin = xcenter - w / 2.
        ymax = ycenter + h / 2.
        xmax = xcenter + w / 2.
        decoded_codes = np.transpose(np.stack([ymin, xmin, ymax, xmax]))
        return box_list.BoxList(tf.convert_to_tensor(decoded_codes))

    # replace box decode with np
    # decoded_boxes = self._box_coder.decode(
    #     tf.reshape(box_encodings, [-1, self._box_coder.code_size]),
    #     tiled_anchors_boxlist)
    decoded_boxes = _box_decode(
        tf.reshape(box_encodings, [-1, self._box_coder.code_size]).numpy(),
        tiled_anchors_boxlist)

    decoded_keypoints = None
    # print(f'{decoded_boxes.has_field(fields.BoxListFields.keypoints)=}')
    # decoded_boxes.has_field(fields.BoxListFields.keypoints)=False
    # if decoded_boxes.has_field(fields.BoxListFields.keypoints):
    #   decoded_keypoints = decoded_boxes.get_field(
    #       fields.BoxListFields.keypoints)
    #   num_keypoints = decoded_keypoints.get_shape()[1]
    #   decoded_keypoints = tf.reshape(
    #       decoded_keypoints,
    #       tf.stack([combined_shape[0], combined_shape[1], num_keypoints, 2]))
    decoded_boxes = tf.reshape(decoded_boxes.get(), tf.stack(
        [combined_shape[0], combined_shape[1], 4]))
    return decoded_boxes, decoded_keypoints

  def regularization_losses(self):
    """Returns a list of regularization losses for this model.

    Returns a list of regularization losses for this model that the estimator
    needs to use during training/optimization.

    Returns:
      A list of regularization loss tensors.
    """
    losses = []
    slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # Copy the slim losses to avoid modifying the collection
    if slim_losses:
      losses.extend(slim_losses)
    if self._box_predictor.is_keras_model:
      losses.extend(self._box_predictor.losses)
    if self._feature_extractor.is_keras_model:
      losses.extend(self._feature_extractor.losses)
    return losses

  def restore_map(self,
                  fine_tune_checkpoint_type='detection',
                  load_all_detection_checkpoint_vars=False):
    """Returns a map of variables to load from a foreign checkpoint.

    See parent class for details.

    Args:
      fine_tune_checkpoint_type: whether to restore from a full detection
        checkpoint (with compatible variable names) or to restore from a
        classification checkpoint for initialization prior to training.
        Valid values: `detection`, `classification`. Default 'detection'.
      load_all_detection_checkpoint_vars: whether to load all variables (when
         `fine_tune_checkpoint_type` is `detection`). If False, only variables
         within the feature extractor scope are included. Default False.

    Returns:
      A dict mapping variable names (to load from a checkpoint) to variables in
      the model graph.
    Raises:
      ValueError: if fine_tune_checkpoint_type is neither `classification`
        nor `detection`.
    """
    if fine_tune_checkpoint_type == 'classification':
      return self._feature_extractor.restore_from_classification_checkpoint_fn(
          self._extract_features_scope)

    elif fine_tune_checkpoint_type == 'detection':
      variables_to_restore = {}
      for variable in variables_helper.get_global_variables_safely():
        var_name = variable.op.name
        if load_all_detection_checkpoint_vars:
          variables_to_restore[var_name] = variable
        else:
          if var_name.startswith(self._extract_features_scope):
            variables_to_restore[var_name] = variable
      return variables_to_restore

    else:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
          fine_tune_checkpoint_type))

  def restore_from_objects(self, fine_tune_checkpoint_type='detection'):
    """Returns a map of Trackable objects to load from a foreign checkpoint.

    Returns a dictionary of Tensorflow 2 Trackable objects (e.g. tf.Module
    or Checkpoint). This enables the model to initialize based on weights from
    another task. For example, the feature extractor variables from a
    classification model can be used to bootstrap training of an object
    detector. When loading from an object detection model, the checkpoint model
    should have the same parameters as this detection model with exception of
    the num_classes parameter.

    Note that this function is intended to be used to restore Keras-based
    models when running Tensorflow 2, whereas restore_map (above) is intended
    to be used to restore Slim-based models when running Tensorflow 1.x.

    Args:
      fine_tune_checkpoint_type: A string inidicating the subset of variables
        to load. Valid values: `detection`, `classification`, `full`. Default
        `detection`.
        An SSD checkpoint has three parts:
        1) Classification Network (like ResNet)
        2) DeConv layers (for FPN)
        3) Box/Class prediction parameters
        The parameters will be loaded using the following strategy:
          `classification` - will load #1
          `detection` - will load #1, #2
          `full` - will load #1, #2, #3

    Returns:
      A dict mapping keys to Trackable objects (tf.Module or Checkpoint).
    """
    if fine_tune_checkpoint_type == 'classification':
      return {
          'feature_extractor':
              self._feature_extractor.classification_backbone
      }
    elif fine_tune_checkpoint_type == 'detection':
      fake_model = tf.train.Checkpoint(
          _feature_extractor=self._feature_extractor)
      return {'model': fake_model}

    elif fine_tune_checkpoint_type == 'full':
      return {'model': self}

    else:
      raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
          fine_tune_checkpoint_type))

  def updates(self):
    """Returns a list of update operators for this model.

    Returns a list of update operators for this model that must be executed at
    each training step. The estimator's train op needs to have a control
    dependency on these updates.

    Returns:
      A list of update operators.
    """
    update_ops = []
    slim_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Copy the slim ops to avoid modifying the collection
    if slim_update_ops:
      update_ops.extend(slim_update_ops)
    if self._box_predictor.is_keras_model:
      update_ops.extend(self._box_predictor.get_updates_for(None))
      update_ops.extend(self._box_predictor.get_updates_for(
          self._box_predictor.inputs))
    if self._feature_extractor.is_keras_model:
      update_ops.extend(self._feature_extractor.get_updates_for(None))
      update_ops.extend(self._feature_extractor.get_updates_for(
          self._feature_extractor.inputs))
    return update_ops

