# Clothes Segmentation using U2NET #

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EhEy3uQh-5oOSagUotVOJAf8m7Vqn0D6?usp=sharing)






# Techinal details

* **U2NET** : This project uses an amazing [U2NET](https://arxiv.org/abs/2005.09007) as a deep learning model. Instead of having 1 channel output from u2net for typical salient object detection task it outputs 4 channels each respresting upper body cloth, lower body cloth, fully body cloth and background. Only categorical cross-entropy loss is used for a given version of the checkpoint.

# Training 

- For training this project requires,
<ul>
    <ul>
    <li>&nbsp; PyTorch > 1.3.0</li>
    <li>&nbsp; tensorboardX</li>
    <li>&nbsp; gdown</li>
    </ul>
</ul>

- 
- Set path of `train` folder which contains training images and `train.csv` which is label csv file in `options/base_options.py`
- To port original u2net of all layer except last layer please run `python setup_model_weights.py` and it will generate weights after model surgey in `prev_checkpoints` folder.
- You can explore various options in `options/base_options.py` like checkpoint saving folder, logs folder etc.
- For single gpu set `distributed = False` in `options/base_options.py`, for multi gpu set it to `True`.
- For single gpu run `python train.py`
- For multi gpu run <br>
&nbsp;`python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=4 --use_env train.py` <br>
Here command is for single node, 4 gpu. Tested only for single node.
- You can watch loss graphs and samples in tensorboard by running tensorboard command in log folder.


# Testing/Inference
- Download pretrained model from this [link](https://drive.google.com/file/d/1dG6VfQPWXPdSnw6r4izthFXa0sW5UkPx/view?usp=drive_link)(165 MB) in `trained_checkpoint` folder.
- Put input images in `input_images` folder
- Run `python infer.py` for inference.
- Output will be saved in `output_images`


