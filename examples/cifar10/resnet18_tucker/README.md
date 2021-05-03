
## Get pre-trained resnet18 model

- Download pre-trained [resnet18 model](https://drive.google.com/file/d/1tDy73OOWlO1B0tZJEbq5NkjZLVlfVt6x/view) and save the model in `./saved_models` directory.

## Run main.py with tucker decomposition

   ```shell
   CUDA_VISIBLE_DEVICES=0 python main.py --train_config_file=./configs/train/cifar10/cifar10/resnet18.yaml --compress_config_file=./configs/compress/tucker.yaml
   ```


## Get results 

- The results are saved in `./log` directory..