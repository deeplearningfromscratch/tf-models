import copy
import os

import tensorflow.compat.v1 as tf

from object_detection import eval_util, inputs, model_lib
from object_detection.builders import model_builder
from object_detection.core import standard_fields as fields
from object_detection.utils import config_util, label_map_util, ops
from object_detection.validate_efficientdet_d0_tf import (
    concat_replica_results, prepare_eval_dict)

import numpy as np
import onnxruntime as ort

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
    onnx_path=None,
    **kwargs,
):
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
        "get_configs_from_pipeline_file"
    ]
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
        "merge_external_params_with_configs"
    ]
    configs = get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override
    )
    kwargs.update({"sample_1_of_n_eval_examples": sample_1_of_n_eval_examples})
    configs = merge_external_params_with_configs(configs, None, kwargs_dict=kwargs)

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

    for latest_checkpoint in tf.train.checkpoints_iterator(
        checkpoint_dir, timeout=timeout, min_interval_secs=wait_interval
    ):

        ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)

        ckpt.restore(latest_checkpoint).expect_partial()

        eager_eval_loop(
            detection_model,
            configs,
            eval_input,
            use_tpu=use_tpu,
            postprocess_on_cpu=postprocess_on_cpu,
        )

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
    eval_input_config = configs["eval_input_config"]
    eval_config = configs["eval_config"]

    is_training = False
    detection_model._is_training = is_training  # pylint: disable=protected-access
    tf.keras.backend.set_learning_phase(is_training)
    evaluator_options = eval_util.evaluator_options_from_eval_config(eval_config)

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

    strategy = tf.compat.v2.distribute.get_strategy()

    for i, (features, labels) in enumerate(eval_dataset):
        prediction_dict, groundtruth_dict, eval_features = strategy.run(
            compute_eval_dict, args=(detection_model, features, labels)
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

        eval_dict, class_agnostic = prepare_eval_dict(
            local_prediction_dict, local_groundtruth_dict, local_eval_features
        )

        if i % 100 == 0:
            print(f"Finished eval step %{i}")

        if evaluators is None:
            if class_agnostic:
                evaluators = class_agnostic_evaluators
            else:
                evaluators = class_aware_evaluators

        for evaluator in evaluators:
            evaluator.add_eval_dict(eval_dict)

    eval_metrics = {}

    for evaluator in evaluators:
        eval_metrics.update(evaluator.evaluate())
    eval_metrics = {str(k): v for k, v in eval_metrics.items()}
    for k in eval_metrics:
        tf.compat.v2.summary.scalar(k, eval_metrics[k], step=global_step)
        print("\t+ %s: %f", k, eval_metrics[k])
    return eval_metrics


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L896-L926
def compute_eval_dict(detection_model, features, labels):
    """Compute the evaluation result on an image."""
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

    prediction_dict = _compute_predictions_dicts(
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
    return prediction_dict, groundtruth_dict, eval_features


# https://github.com/tensorflow/models/blob/3afd339ff97e0c2576300b245f69243fc88e066f/research/object_detection/model_lib_v2.py#L54-L186
def _compute_predictions_dicts(
    model,
    features,
    labels,
    training_step=None,
    add_regularization_loss=True,
):
    model_lib.provide_groundtruth(model, labels, training_step=training_step)
    preprocessed_images = features[fields.InputDataFields.image]
    

    session = ort.InferenceSession(ONNX_PATH, providers=["CUDAExecutionProvider"])
    input_name = session.get_inputs()[0].name
    raw_bboxes, raw_scores = session.run(
        [], input_feed={input_name: preprocessed_images.numpy().transpose(0, 3, 1, 2)}
    )
    # raw_scores: sigmoid output
    # class_predictions_with_background: inverse sigmoid output
    class_predictions_with_background = np.log(raw_scores/(1-raw_scores))

    prediction_dict = model.predict(
        preprocessed_images,
        features[fields.InputDataFields.true_image_shape],
        **model.get_side_inputs(features),
    )

    prediction_dict['box_encodings'] = tf.convert_to_tensor(raw_bboxes)
    prediction_dict['class_predictions_with_background'] = tf.convert_to_tensor(class_predictions_with_background)
    prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)
    return prediction_dict


if __name__ == "__main__":
    root_dir = "object_detection/efficientdet_d0_coco17_tpu-32/eval"
    pipeline_config_path = os.path.join(root_dir, "pipeline.config")
    checkpoint_dir = os.path.join(root_dir, "checkpoint")
    ONNX_PATH = os.path.join(root_dir, "efficientdet_d0.onnx")
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
