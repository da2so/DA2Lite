## Getting started with imagenet!

We will train ResNet50 model with imagenet.

Let's see the config file.

```yaml
# configs/train/imagenet/resnet50.yaml
DATASET:
  NAME: imagenet
  IMG_SHAPE: [3, 224, 224] 
  ROOT_DIR: '/data/shared2/imagenet/'
  NUM_WORKERS: 12
  DATA_AUG: True

MODEL:
  NAME: resnet50_imagenet
  NUM_CLASSES: 1000
  PRE_TRAINED:
    IS_USE: False
  #   PATH: './saved_models/origin_imagenet_resnet50.pt'

TRAIN:
  IS_USE: True 
  OPTIMIZER: sgd
  OPTIMIZER_ARGS:
      MOMENTUM : 0.9
      WEIGHT_DECAY: 5e-4
  LOSS: categorical_crossentropy 
  SCHEDULER: cosineannealLR 
  SCHEDULER_ARGS:
    T_MAX: 200
  BATCH_SIZE: 256
  EPOCHS: 200
  LR: 0.1

GPU:
  IS_USE: True
```

First we define Imagenet dataset. Set input image shape and root directory where data is stored.
The imagenet dataset is hue dataset. So, We need to set proper `NUM_WORKERS`.

Next is model definision. Imagenet's input size is bigger then CIFAR10. The Resnet model is quit different. In this example, We use `resnet50_imagenet` model and We will train from scratch.

Now, it's show time!

```shell
python main --train_config_file=./configs/train/imagnet/resnet50.yaml --compress_config_file=./configs/compress/pruning/nuclear_norm.yaml
```
