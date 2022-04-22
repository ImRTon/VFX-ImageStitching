# VFX Project2 Image Stitching
Combine a set of images into a larger image by registering, warping, resampling and blending them together.

We implemented .

> NTUST CGLab M11015117 湯濬澤\
> NTUST CGLab M11015029 張立彥

## Environment
```
Python 3.8
    opencv-python
    Pillow
    numpy
    tqdm
    matplotlib
    ...
```

## Install
```
conda create --name VFX python=3.8
conda activate VFX
pip install -r requirements.txt
```

## Run
-i --input_dir INPUT_DIR\
-p --plot `True` / `False`\
-r --match_ratio [0 ~ 1]\
-f --focal_length FOCAL_LENGTH

```
python main.py -i INPUT_DIR -f FOCAL_LENGTH
```