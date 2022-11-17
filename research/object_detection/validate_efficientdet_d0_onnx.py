import copy
import os

import tensorflow.compat.v1 as tf

from object_detection import inputs
from object_detection.builders import model_builder
from object_detection.utils import config_util

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
