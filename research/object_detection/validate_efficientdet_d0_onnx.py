import copy
import os

import tensorflow.compat.v1 as tf

from object_detection import eval_util, inputs, model_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util, label_map_util, ops

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


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib.py#L52-L68
MODEL_BUILD_UTIL_MAP = {
    "get_configs_from_pipeline_file": config_util.get_configs_from_pipeline_file,
    # "create_pipeline_proto_from_configs": config_util.create_pipeline_proto_from_configs,
    "merge_external_params_with_configs": config_util.merge_external_params_with_configs,
    # "create_train_input_fn": inputs.create_train_input_fn,
    "create_eval_input_fn": inputs.create_eval_input_fn,
    # "create_predict_input_fn": inputs.create_predict_input_fn,
    "detection_model_fn_base": model_builder.build,
}
# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L1022-L1169
def eval_continuously(
    pipeline_config_path,
    config_override=None,
    train_steps=None,
    sample_1_of_n_eval_examples=1,
    sample_1_of_n_eval_on_train_examples=1,
    use_tpu=False,
    override_eval_num_epochs=True,
    postprocess_on_cpu=False,
    model_dir=None,
    checkpoint_dir=None,
    wait_interval=180,
    timeout=3600,
    eval_index=0,
    save_final_config=False,
    **kwargs
):
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
        "get_configs_from_pipeline_file"
    ]
    # create_pipeline_proto_from_configs = MODEL_BUILD_UTIL_MAP[
    #     "create_pipeline_proto_from_configs"
    # ]
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
        "merge_external_params_with_configs"
    ]

    configs = get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override
    )
    kwargs.update(
        {
            "sample_1_of_n_eval_examples": sample_1_of_n_eval_examples,
            # del
            # "use_bfloat16": configs["train_config"].use_bfloat16 and use_tpu,
        }
    )
    # del
    # if train_steps is not None:
    #     kwargs["train_steps"] = train_steps

    # if override_eval_num_epochs:
    #     kwargs.update({"eval_num_epochs": 1})
    #     tf.logging.warning("Forced number of epochs for all eval validations to be 1.")
    configs = merge_external_params_with_configs(configs, None, kwargs_dict=kwargs)
    # if model_dir and save_final_config:
    #     tf.logging.info("Saving pipeline config file to directory %s", model_dir)
    #     pipeline_config_final = create_pipeline_proto_from_configs(configs)
    #     config_util.save_pipeline_config(pipeline_config_final, model_dir)

    model_config = configs["model"]
    train_input_config = configs["train_input_config"]
    eval_config = configs["eval_config"]
    eval_input_configs = configs["eval_input_configs"]
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

    # del
    # if kwargs["use_bfloat16"]:
    #     tf.compat.v2.keras.mixed_precision.set_global_policy("mixed_bfloat16")

    eval_input_config = eval_input_configs[eval_index]
    strategy = tf.compat.v2.distribute.get_strategy()
    with strategy.scope():
        detection_model = MODEL_BUILD_UTIL_MAP["detection_model_fn_base"](
            model_config=model_config, is_training=True
        )

    eval_input = strategy.experimental_distribute_dataset(
        inputs.eval_input(
            eval_config=eval_config,
            eval_input_config=eval_input_config,
            model_config=model_config,
            model=detection_model,
        )
    )

    # global_step = tf.compat.v2.Variable(
    #     0, trainable=False, dtype=tf.compat.v2.dtypes.int64
    # )

    # optimizer, _ = optimizer_builder.build(
    #     configs["train_config"].optimizer, global_step=global_step
    # )

    for latest_checkpoint in tf.train.checkpoints_iterator(
        checkpoint_dir, timeout=timeout, min_interval_secs=wait_interval
    ):
        # ckpt = tf.compat.v2.train.Checkpoint(
        #     step=global_step, model=detection_model, optimizer=optimizer
        # )
        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

        # del by https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L188
        # if eval_config.use_moving_averages:
        #     unpad_groundtruth_tensors = eval_config.batch_size == 1 and not use_tpu
        #     _ensure_model_is_built(
        #         detection_model, eval_input, unpad_groundtruth_tensors
        #     )
        #     optimizer.shadow_copy(detection_model)

        ckpt.restore(latest_checkpoint).expect_partial()

        # del by https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L188
        # if eval_config.use_moving_averages:
        #     optimizer.swap_weights()

        # del we don't need to write metrics
        # summary_writer = tf.compat.v2.summary.create_file_writer(
        #     os.path.join(model_dir, "eval", eval_input_config.name)
        # )
        # with summary_writer.as_default():
        eager_eval_loop(
            detection_model,
            configs,
            eval_input,
            use_tpu=use_tpu,
            postprocess_on_cpu=postprocess_on_cpu,
            # global_step=global_step,
        )

        # if global_step.numpy() == configs["train_config"].num_steps:
        #     tf.logging.info("Exiting evaluation at step %d", global_step.numpy())
        #     return
    return


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L833-L1019
def eager_eval_loop(
    detection_model,
    configs,
    eval_dataset,
    use_tpu=False,
    postprocess_on_cpu=False,
    global_step=None,
):
    del postprocess_on_cpu
    train_config = configs["train_config"]
    eval_input_config = configs["eval_input_config"]
    eval_config = configs["eval_config"]
    add_regularization_loss = train_config.add_regularization_loss

    is_training = False
    detection_model._is_training = is_training  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(is_training)
    evaluator_options = eval_util.evaluator_options_from_eval_config(eval_config)
    batch_size = eval_config.batch_size

    class_agnostic_category_index = (
        label_map_util.create_class_agnostic_category_index()
    )

    class_agnostic_evaluators = eval_util.get_evaluators(
        eval_config, list(class_agnostic_category_index.values()), evaluator_options
    )

    class_aware_evaluators = None
    if eval_input_config.label_map_path:
        class_aware_category_index = label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path
        )
        class_aware_evaluators = eval_util.get_evaluators(
            eval_config, list(class_aware_category_index.values()), evaluator_options
        )

    evaluators = None
    loss_metrics = {}

    agnostic_categories = label_map_util.create_class_agnostic_category_index()
    per_class_categories = label_map_util.create_category_index_from_labelmap(
        eval_input_config.label_map_path
    )
    # keypoint_edges = [(kp.start, kp.end) for kp in eval_config.keypoint_edge]

    strategy = tf.compat.v2.distribute.get_strategy()

    for i, (features, labels) in enumerate(eval_dataset):
        # try:
        losses_dict, prediction_dict, groundtruth_dict, eval_features = strategy.run(
            compute_eval_dict, args=(detection_model, features, labels)
        )
        # except Exception as exc:  # pylint:disable=broad-except
        #     tf.logging.info("Encountered %s exception.", exc)
        #     tf.logging.info(
        #         "A replica probably exhausted all examples. Skipping "
        #         "pending examples on other replicas."
        #     )
        #     break
        # (
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

        eval_dict, class_agnostic = prepare_eval_dict(
            local_prediction_dict, local_groundtruth_dict, local_eval_features
        )
        for loss_key, loss_tensor in iter(losses_dict.items()):
            losses_dict[loss_key] = strategy.reduce(
                tf.distribute.ReduceOp.MEAN, loss_tensor, None
            )
        if class_agnostic:
            category_index = agnostic_categories
        else:
            category_index = per_class_categories

        if i % 100 == 0:
            print("Finished eval step %d", i)

        # use_original_images = fields.InputDataFields.original_image in features

        # if use_original_images and i < eval_config.num_visualizations:
        #     sbys_image_list = vutils.draw_side_by_side_evaluation_image(
        #         eval_dict,
        #         category_index=category_index,
        #         max_boxes_to_draw=eval_config.max_num_boxes_to_visualize,
        #         min_score_thresh=eval_config.min_score_threshold,
        #         use_normalized_coordinates=False,
        #         keypoint_edges=keypoint_edges or None,
        #     )
        #     for j, sbys_image in enumerate(sbys_image_list):
        #         tf.compat.v2.summary.image(
        #             name="eval_side_by_side_{}_{}".format(i, j),
        #             step=global_step,
        #             data=sbys_image,
        #             max_outputs=eval_config.num_visualizations,
        #         )
        #     if eval_util.has_densepose(eval_dict):
        #         dp_image_list = vutils.draw_densepose_visualizations(eval_dict)
        #         for j, dp_image in enumerate(dp_image_list):
        #             tf.compat.v2.summary.image(
        #                 name="densepose_detections_{}_{}".format(i, j),
        #                 step=global_step,
        #                 data=dp_image,
        #                 max_outputs=eval_config.num_visualizations,
        #             )

        if evaluators is None:
            if class_agnostic:
                evaluators = class_agnostic_evaluators
            else:
                evaluators = class_aware_evaluators

        for evaluator in evaluators:
            evaluator.add_eval_dict(eval_dict)

        for loss_key, loss_tensor in iter(losses_dict.items()):
            if loss_key not in loss_metrics:
                loss_metrics[loss_key] = []
            loss_metrics[loss_key].append(loss_tensor)

    eval_metrics = {}

    for evaluator in evaluators:
        eval_metrics.update(evaluator.evaluate())
    for loss_key in loss_metrics:
        eval_metrics[loss_key] = tf.reduce_mean(loss_metrics[loss_key])

    eval_metrics = {str(k): v for k, v in eval_metrics.items()}
    for k in eval_metrics:
        tf.compat.v2.summary.scalar(k, eval_metrics[k], step=global_step)
        print("\t+ %s: %f", k, eval_metrics[k])
    return eval_metrics


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L896-L926
@tf.function
def compute_eval_dict(detection_model, features, labels):
    """Compute the evaluation result on an image."""
    # For evaling on train data, it is necessary to check whether groundtruth
    # must be unpadded.
    use_tpu = False  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L1028
    batch_size = 1  # https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config#L189
    add_regularization_loss = False
    boxes_shape = labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list()
    unpad_groundtruth_tensors = (
        boxes_shape[1] is not None and not use_tpu and batch_size == 1
    )
    groundtruth_dict = labels
    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors
    )

    losses_dict, prediction_dict = _compute_losses_and_predictions_dicts(
        detection_model,
        features,
        labels,
        training_step=None,
        add_regularization_loss=add_regularization_loss,
    )
    prediction_dict = detection_model.postprocess(
        prediction_dict, features[fields.InputDataFields.true_image_shape]
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
        inputs.HASH_KEY: features[inputs.HASH_KEY],
    }
    return losses_dict, prediction_dict, groundtruth_dict, eval_features


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L54-L186
def _compute_losses_and_predictions_dicts(
    model, features, labels, training_step=None, add_regularization_loss=True
):
    model_lib.provide_groundtruth(model, labels, training_step=training_step)
    preprocessed_images = features[fields.InputDataFields.image]

    prediction_dict = model.predict(
        preprocessed_images,
        features[fields.InputDataFields.true_image_shape],
        **model.get_side_inputs(features)
    )
    prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

    losses_dict = model.loss(
        prediction_dict, features[fields.InputDataFields.true_image_shape]
    )
    losses = [loss_tensor for loss_tensor in losses_dict.values()]
    if add_regularization_loss:
        # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
        ## need to convert these regularization losses from bfloat16 to float32
        ## as well.
        regularization_losses = model.regularization_losses()
        if regularization_losses:
            regularization_losses = ops.bfloat16_to_float32_nested(
                regularization_losses
            )
            regularization_loss = tf.add_n(
                regularization_losses, name="regularization_loss"
            )
            losses.append(regularization_loss)
            losses_dict["Loss/regularization_loss"] = regularization_loss

    total_loss = tf.add_n(losses, name="total_loss")
    losses_dict["Loss/total_loss"] = total_loss

    return losses_dict, prediction_dict


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L733-L823
def prepare_eval_dict(detections, groundtruth, features):
    groundtruth_boxes = groundtruth[fields.InputDataFields.groundtruth_boxes]
    groundtruth_boxes_shape = tf.shape(groundtruth_boxes)
    # For class-agnostic models, groundtruth one-hot encodings collapse to all
    # ones.
    class_agnostic = fields.DetectionResultFields.detection_classes not in detections

    if class_agnostic:
        groundtruth_classes_one_hot = tf.ones(
            [groundtruth_boxes_shape[0], groundtruth_boxes_shape[1], 1]
        )
    else:
        groundtruth_classes_one_hot = groundtruth[
            fields.InputDataFields.groundtruth_classes
        ]
    label_id_offset = 1  # Applying label id offset (b/63711816)
    groundtruth_classes = (
        tf.argmax(groundtruth_classes_one_hot, axis=2) + label_id_offset
    )
    groundtruth[fields.InputDataFields.groundtruth_classes] = groundtruth_classes

    label_id_offset_paddings = tf.constant([[0, 0], [1, 0]])
    if fields.InputDataFields.groundtruth_verified_neg_classes in groundtruth:
        groundtruth[fields.InputDataFields.groundtruth_verified_neg_classes] = tf.pad(
            groundtruth[fields.InputDataFields.groundtruth_verified_neg_classes],
            label_id_offset_paddings,
        )
    if fields.InputDataFields.groundtruth_not_exhaustive_classes in groundtruth:
        groundtruth[fields.InputDataFields.groundtruth_not_exhaustive_classes] = tf.pad(
            groundtruth[fields.InputDataFields.groundtruth_not_exhaustive_classes],
            label_id_offset_paddings,
        )
    if fields.InputDataFields.groundtruth_labeled_classes in groundtruth:
        groundtruth[fields.InputDataFields.groundtruth_labeled_classes] = tf.pad(
            groundtruth[fields.InputDataFields.groundtruth_labeled_classes],
            label_id_offset_paddings,
        )

    use_original_images = fields.InputDataFields.original_image in features
    if use_original_images:
        eval_images = features[fields.InputDataFields.original_image]
        true_image_shapes = features[fields.InputDataFields.true_image_shape][:, :3]
        original_image_spatial_shapes = features[
            fields.InputDataFields.original_image_spatial_shape
        ]
    else:
        eval_images = features[fields.InputDataFields.image]
        true_image_shapes = None
        original_image_spatial_shapes = None

    eval_dict = eval_util.result_dict_for_batched_example(
        eval_images,
        features[inputs.HASH_KEY],
        detections,
        groundtruth,
        class_agnostic=class_agnostic,
        scale_to_absolute=True,
        original_image_spatial_shapes=original_image_spatial_shapes,
        true_image_shapes=true_image_shapes,
    )

    return eval_dict, class_agnostic


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L826-L830
def concat_replica_results(tensor_dict):
    new_tensor_dict = {}
    for key, values in tensor_dict.items():
        new_tensor_dict[key] = tf.concat(values, axis=0)
    return new_tensor_dict


if __name__ == "__main__":
    root_dir = "object_detection/efficientdet_d0_coco17_tpu-32/eval"
    pipeline_config_path = os.path.join(root_dir, "pipeline.config")
    checkpoint_dir = os.path.join(root_dir, "checkpoint")
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
