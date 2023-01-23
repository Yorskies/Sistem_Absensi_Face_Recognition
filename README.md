# YOLOv8-Face

<a target="_blank" href="https://colab.research.google.com/drive/14QfCaIClnfSmHjjVkMNoMtZ0MlRhCwr6?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

YOLOv8 for Face Detection. The project is a fork over [ultralytics](https://github.com/ultralytics/ultralytics) repo. They made a simple interface for training and run inference. Model detects faces on images and returns bounding boxes, score and class.


**Model** | YOLOv8 nano | YOLOv8 medium | RetinaFace-R50
--- | --- | --- | --- 
**Avg. FPS (Colab T4 GPU)** | 82 | 31 | ---
**WiderFace Easy Val. AP** | 0.8831 | 0.9761 | 0.9650 
**WiderFace Medium Val. AP** | 0.8280 | 0.9487 | 0.9560
**WiderFace Hard Val. AP** | 0.6055 | 0.7709 | 0.9040

## Installation

I recommend to use a new virtual environment and install the following packages:

```bash
# Install pytorch for CUDA 11.7 from pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```
or alternatively if you are using conda

```bash
# Install pytorch for CUDA 11.7 from conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

then you install the ultralytics package (this code works for *ultralytics-8.0.10*)
```bash
pip install ultralytics
```

## Usage

There are three python scripts, [train.py](train.py) is from fine tune a yolov8 model, [test.py](test.py) is to test the model with images and [demo.py](demo.py) is to launch a real-time demo of the model with your webcam.

You must configure [wider.yaml](datasets/wider.yaml) according to the path in your pc (default settings are relative to [datasets](datasets) folder).

The training arguments
```python
parser = argparse.ArgumentParser(description='Train YOLOv8 for Face Detection')
parser.add_argument('-b', '--batch', type=int, help='Batch Size', default=8)
parser.add_argument('-e', '--epochs', type=int, help='Number of Epochs', default=100)
parser.add_argument('-w', '--workers', type=int, help='Workers for datalaoder', default=2)
parser.add_argument('--pretrained', action='store_true', help='Finetune')
parser.add_argument('--no-pretrained', dest='pretrained', action='store_false', help='Train from zero.')
parser.set_defaults(pretrained=False)
parser.add_argument('-i', '--imgsize', type=int, help='Image size', default=640)
parser.add_argument('-m', '--model',
                    type=str,
                    choices=['yolov8n','yolov8s', 'yolov8m', 'yolov8l', 'yolov8xl'],
                    default='yolov8m',
                    help='YOLOv8 size model')
```

the test arguments

```python
parser.add_argument('-w', '--weights', type=str, help='Path to trained  weights', default='runs/detect/train/weights/best.pt')

parser.add_argument('-t', '--threshold', type=float, help='Score threshold',
                    default=0.5)

parser.add_argument('-i', '--input', type=str, help='Sample input image path',
                    default='test_input.jpg')

parser.add_argument('-o', '--output', type=str, help='Output image path',
                    default='test_output.jpg')
```

and finally the demo arguments

```python
parser.add_argument('-w', '--weights', type=str, help='Path to trained  weights', default='runs/detect/train/weights/best.pt')

```
## Result example
<img src="test_images/test_output.jpg" width="600"/>
<img src="test_images/test_output_3.jpg" width="600"/>

## Downloads

You can download the pretrained weights (YOLOv8 medium) v0.2 for Face Detection [here](https://drive.google.com/file/d/1IJZBcyMHGhzAi0G4aZLcqryqZSjPsps-/view?usp=sharing).

You can download the pretrained weights (YOLOv8 nano) v0.1 for Face Detection [here](https://drive.google.com/file/d/1ZD_CEsbo3p3_dd8eAtRfRxHDV44M0djK/view?usp=sharing). 

You can also download the WiderFace dataset properly formatted to train your own model [here](https://drive.google.com/file/d/1roNilRaLMz4uLqZvINDAyxasG8ncwb5n/view?usp=share_link).
