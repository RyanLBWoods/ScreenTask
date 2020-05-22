import torch
from torch.nn.functional import log_softmax


class simpleInfer(object):
    def __init__(self):
        self.load_model()
        self.model = None

    def load_model(model_path: str):
        model = torch.load(model_path)

    def infer(self, inp: list):
        """
        Infer Method
        inputs:
        - inp: list, a list of np array, with size(28, 28)

        :return:
        - list, a list of int, which is th result of model inference
        """
        assert self.model is not None, "Model is not loaded!"
        outputs = self.model(inp)
        outputs = log_softmax(outputs, dim=1)
        pred = torch.max(outputs, dim=1)[1]
        print('The prediction is ' + pred.item())
        return pred.item()


if __name__ == '__main__':
    path = './simpleClassifier_20.pth'
    infer = simpleInfer()
    infer.load_model(path)
    pred = infer.infer()
