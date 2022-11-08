# Installation
- git clone The Tensorflow Model Garden
    ```
    git clone git@github.com:deeplearningfromscratch/tf-models.git
    ```
- pre-requisite
    - protoc: https://grpc.io/docs/protoc-installation/
- python package installation
    ```
    # Assmue "tf-models/research" as working directory here after
    cd tf-models/research
    
    # Compile protos.
    protoc object_detection/protos/*.proto --python_out=.

    # Install TensorFlow Object Detection API.
    cp object_detection/packages/tf2/setup.py .
    pip install -e .

    # Verify Install
    python object_detection/builders/model_builder_tf2_test.py
    ```
    - source
        - https://github.com/deeplearningfromscratch/tf-models/blob/master/research/object_detection/g3doc/tf2.md
    - (optional) to use GPU with tensorflow: https://www.tensorflow.org/install/pip
        - for example, for linux:
            ```
            conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
            python3 -m pip install tensorflow
            # Verify install:
            python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
            ```

# Validation Dataset Preparation
- Download MSCOCO data https://cocodataset.org/#home
    ```
    DOWNLOAD_DIR=dataset/mscoco
    mkdir -p ${DOWNLOAD_DIR}

    # Download mscoco data
    wget http://images.cocodataset.org/zips/val2017.zip -O ${DOWNLOAD_DIR}/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O ${DOWNLOAD_DIR}/annotations_trainval2017.zip
    unzip ${DOWNLOAD_DIR}/val2017.zip -d ${DOWNLOAD_DIR}
    unzip ${DOWNLOAD_DIR}/annotations_trainval2017.zip -d ${DOWNLOAD_DIR}
    ```
- Create tfrecord files
    ```
    python object_detection/dataset_tools/create_coco_val_tf_record.py \
        --val_image_dir dataset/mscoco/val2017 \
        --val_annotations_file dataset/mscoco/annotations/instances_val2017.json \
        --output_dir dataset/mscoco/coco_val.record
    ```
- source: https://github.com/google/automl/tree/master/efficientdet#7-eval-on-coco-2017-val-or-test-dev

# ONNX Model Export
- Download tensorflow pretrained EfficientNet-D0
    ```
    wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
    tar -xvf efficientdet_d0_coco17_tpu-32.tar.gz -C object_detection/
    rm efficientdet_d0_coco17_tpu-32.tar.gz
    ```
- Export to convert traininig graph to inference graph(saved_model)
    ```
    python object_detection/exporter_main_v2.py \
        --input_type image_tensor \
        --pipeline_config_path object_detection/efficientdet_d0_coco17_tpu-32/pipeline.config \
        --trained_checkpoint_dir object_detection/efficientdet_d0_coco17_tpu-32/checkpoint \
        --output_directory object_detection/efficientdet_d0_coco17_tpu-32/eval
    ```
- Convert tensorflow saved model to ONNX
    ```
    pip install tf2onnx
    python -m tf2onnx.convert --saved-model object_detection/efficientdet_d0_coco17_tpu-32/eval/saved_model/ --output object_detection/efficientdet_d0_coco17_tpu-32/eval/efficientdet_d0_orig.onnx --opset 13
    ```
- Remove pre/post-processing sub-graphs
    ```
    pip install furiosa-sdk
    python object_detection/onnx_efficientdet_0_extracter.py
    ```
- source
    - https://github.com/deeplearningfromscratch/tf-models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    - https://github.com/onnx/tensorflow-onnx#getting-started
    - https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#extracting-sub-model-with-inputs-outputs-tensor-names
