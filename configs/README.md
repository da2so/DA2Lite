
## Configuration setting

There are two configuration folders to operate <span style="color:DodgerBlue">**DA2Lite**</span>. 

The first is a **train** folder that contains .yaml files for train confuration options (e.g. model, dataset, optimizer, loss, ...).
And, a configuration file in **compress** folder allows to user select what compressipon methods will be used sequentially. Besides, when after compression a model, you can control post-train configurations using its .yaml file.


## CONFIGURATION STRUCTURE

The followings indicate example configurations and available choices of configurations (represented by comments).

### Train configurations (Train/all.yaml)


   ```shell
    DATASET:
    NAME: cifar10 # Choice of ['cifar10', 'cifar100']
    IMG_SHAPE: [3, 32, 32] # [Channel, Height, Width] of the dataset
    DATA_AUG: True  # If True, standard data augmentation will be used.

    MODEL: 
    NAME: resnet50 # 
    NUM_CLASSES: 100 # The number of classes 
    PRE_TRAINED:
        IS_USE: False # Load your pretrained model???
        PATH: './cifar10_resnet50.pt' # Model path for the pretrained model path
    
    TRAIN:
    IS_USE: True # Train a model??
    OPTIMIZER: sgd # Choice of ['sgd', 'adam']
    OPTIMIZER_ARGS: # Arguments of a choosed optimizer
        MOMENTUM : 0.9 #for sgd
        WEIGHT_DECAY: 5e-4 #for sgd
        """
        BETAS = (0.9, 0.999) for adam
        EPSILON = 1e-8  for adam
        """
    LOSS: categorical_crossentropy # Choice of [categorical_crossentropy, kld, mae, mse]
    SCHEDULER: stepLR # [stepLR, exponentialLR, cosineannealLR]
    SCHEDULER_ARGS: # Arguments of a choosed scheduler
        STEP_SIZE: 60 # for stepLR
        GAMMA: 0.1 # for stepLR
        """
        GAMMA: 0.95 for exponentialLR

        T_MAX: 140(The number of epochs) for cosineannealLR
        """ 
    BATCH_SIZE: 128 # Batch size
    EPOCHS: 140 # Epochs
    LR: 0.1 # Learning rate

    GPU:
    IS_USE: True # Use CUDA GPU?
   ```


### Compression configurations (Compress/all.yaml)


   ```shell
    PRUNING:
        POST_TRAIN:
            IS_USE: True # Use post-train after pruning?
            OPTIMIZER: sgd # Choice of ['sgd', 'adam']
            OPTIMIZER_ARGS:  # Arguments of a choosed optimizer
            MOMENTUM: 0.9 #for sgd
            WEIGHT_DECAY: 5e-4 #for sgd
            """
            BETAS = (0.9, 0.999) for adam
            EPSILON = 1e-8  for adam
            """
            LOSS: categorical_crossentropy # Choice of [categorical_crossentropy, kld, mae, mse]
            SCHEDULER: cosineannealLR # [stepLR, exponentialLR, cosineannealLR]
            SCHEDULER_ARGS:  # Arguments of a choosed scheduler
            T_MAX: 30 #for 
            """
            STEP_SIZE: 60 # for stepLR
            GAMMA: 0.1 # for stepLR

            GAMMA: 0.95 # for exponentialLR
            """
            BATCH_SIZE: 256 # Batch size
            EPOCHS: 30 # Epochs
            LR: 0.001 # Learning rate
  
    METHOD:
        CRITERIA: L1Criteria # Choice of ['L1Criteria', 'L2Criteria', 'RandomCriteria']
        STRATEGY: MinMaxStrategy # Choice of ['MinMaxStrategy', 'RandomStrategy', 'StaticStrategy']
        PRUNING_RATIO: [0.0, 0.5] # for MinMaxStrategy
        """
        PRUNING_RATIO: None # for RandomStrategy

        PRUNING_RATIO: 0.5 # for StaticStrategy
        """


    FD:
        POST_TRAIN:
            IS_USE: True # Use post-train after filter decomposition??
            OPTIMIZER: sgd
            OPTIMIZER_ARGS:
            MOMENTUM: 0.9
            WEIGHT_DECAY: 5e-4
            LOSS: categorical_crossentropy 
            SCHEDULER: cosineannealLR 
            SCHEDULER_ARGS:
            T_MAX: 30
            BATCH_SIZE: 256
            EPOCHS: 30
            LR: 0.001
        
        METHOD:
            DECOMPOSITION: Tucker # Choice of ['Tucker']
            START_IDX: 6 # Start index of filter decomposition (based on an order of conv layer)
            RANK: 6 # Choice of ['4', '5', '6', 'VBMF']


   ```

