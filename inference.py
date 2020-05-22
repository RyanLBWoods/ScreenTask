import torch
import torch.utils.data as data
from torch.autograd import Variable
from torch.nn.functional import log_softmax
from classifier import simpleClassifier
from data_loader import dataloader


class simpleInfer(object):
    def __init__(self):
        self.model = None

    def load_model(self, model_path: str):
        self.model = torch.load(model_path)

    def infer(self, inp: list):
        """
        Infer Method
        inputs:
        - inp: list, a list of np array, with size(28, 28)

        :return:
        - list, a list of int, which is th result of model inference
        """
        assert self.model is not None, "Model is not loaded! Please call load_model()"
        self.model.eval()
        with torch.no_grad():
            inputs = torch.from_numpy(inp)
            outputs = self.model(inputs)
            outputs = log_softmax(outputs, dim=1)
            preds = torch.max(outputs, dim=1)[1]
        return preds.tolist()


# if __name__ == '__main__':
#     path = './t10k-images-idx3-ubyte'
#     l_path = './t10k-labels-idx1-ubyte'
#     test_data = data.DataLoader(dataloader(path, l_path), batch_size=20, shuffle=True)
#     path = './simpleClsifier_model.pth'
#     infer = simpleInfer()
#     infer.load_model(path)
#     for imgs, labels in test_data:
#         imgs = imgs.numpy()
#         outputs = infer.infer(imgs)
#         for i in range(20):
#             print(outputs[i], labels[i])
#         exit(0)
