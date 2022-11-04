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

# Dataset Preparation
- Download coco data
    ```
    # From tf-models/research
    mkdir mscoco
    cd mscoco
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    ```
- Create tfrecord files
    ```
    python data_tools
    ```
