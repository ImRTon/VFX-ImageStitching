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
-a --align_img `True` / `False`\
-p --plot `True` / `False`\
-s --sample_method `uniform` / `random`\
-k --key [0 ~ 1]

```
python main.py -i INPUT_DIR -a ALIGN_IMG_OR_NOT -s SAMPLE_METHOD
```

## Code