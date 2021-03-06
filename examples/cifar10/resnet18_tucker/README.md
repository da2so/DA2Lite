
## Get pre-trained resnet18 model

- Download pre-trained [resnet18 model](https://drive.google.com/file/d/1tDy73OOWlO1B0tZJEbq5NkjZLVlfVt6x/view) and save the model in `./saved_models` directory.
   - You should make `./saved_models` directory.

## Run main.py with tucker decomposition

   ```shell
   CUDA_VISIBLE_DEVICES=0 python main.py --train_config_file=./configs/train/cifar10/resnet18.yaml --compress_config_file=./configs/compress/fd/tucker.yaml
   ```

The following figure respresents a procedure log of tucker decomposition.

<p align="center">
<img src="imgs/1.PNG" width="700" />
</p>

## Get results 

- The results will be saved in `./log` directory.
   - The original and compressed models are saved in `./models`(sub-directory of `./log`).

   - The below figure shows a fine-tuning procedure right after tucker decomposion.
   
   <p align="center">
   <img src="imgs/2.png" width="700" />
   </p>

The result files can download in [Here](https://drive.google.com/file/d/1M2azkTqapPaciVZwIhischEg9hJXd66g/view?usp=sharing).