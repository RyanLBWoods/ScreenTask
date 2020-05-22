import torch
import torch.utils.data as data
import struct


class dataloader(data.Dataset):
    def __init__(self, img_path, label_path, transform=None):
        """
        Initiate data loader
        :param img_path: str, path of data set
        :param label_path: str, path of label
        :param transform: torchvision.transform, transform functions in torchvision.transform
        """
        super(dataloader, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.imgs = self.load_imgs(self.img_path)
        self.labels = self.load_labels(self.label_path)
        self.transform = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        img = img.unsqueeze(0)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def load_imgs(path):
        with open(path, 'rb') as f:
            data_buf = f.read()
            _, num_imgs, rows, cols = struct.unpack_from('>IIII', data_buf, 0)
            idx = struct.calcsize('>IIII')
            img_bits = num_imgs * rows * cols
            bits_str = '>' + str(img_bits) + 'B'
            imgs = struct.unpack_from(bits_str, data_buf, idx)
        imgs = torch.Tensor(imgs)
        imgs = torch.reshape(imgs, [num_imgs, rows, cols])
        return imgs

    @staticmethod
    def load_labels(path):
        with open(path, 'rb') as f:
            data_buf = f.read()
            _, num_labels = struct.unpack_from('>II', data_buf, 0)
            idx = struct.calcsize('>II')
            lb_str = '>' + str(num_labels) + 'B'
            labels = struct.unpack_from(lb_str, data_buf, idx)
        return labels
