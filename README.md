# ScreenTask
CV/ML Engineer test, a simple classifier for MNIST data set.
## Clone and Install
#### The framework used was PyTorch 1.5.0 (current stable version).
        $ pip install torch torchvision
        $ git clone https://github.com/RyanLBWoods/ScreenTask.git
## Test Usage
Please modify the following code in your script to run test.
```python
infer = simpleInfer()
model_path = 'model_path'
infer.load_model(path)
imgs = 'load_your_images_as_numpy_arrays'
outputs = infer.infer(imgs)
```
