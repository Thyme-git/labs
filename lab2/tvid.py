import os
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image
import torch
# import matplotlib.pyplot as plt

class TvidDataset(data.Dataset):

    # TODO Implement the dataset class inherited
    # from `torch.utils.data.Dataset`.
    # tips: Use `transforms`.

    def __init__(self, root='./tiny_vid', mode='train', transform=True):
        super(TvidDataset, self).__init__()
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5,0.5,0.5],
                std=[0.5,0.5,0.5]
            )
        ])
        self.transform = transform

        self.class_list = ['bird', 'car', 'dog', 'lizard', 'turtle']
        self.root = root
        self.mode = mode
        self.boxes = []
        for i in range(len(self.class_list)):
            self.boxes.append(self.load_gt(self.class_list[i], i))
        

    def load_gt(self, cls: str, cls_id: int):
        path = self.root + '/' + cls + '_gt.txt'
        cls_boxes = []
        with open(path) as f:
            for line in f:
                box = [int(each) for each in line.split(' ')]
                box[0] = cls_id
                box = torch.Tensor(box)
                cls_boxes.append(box)
        return cls_boxes

    def __len__(self):
        return 150 if self.mode == 'test' else 750

    def __getitem__(self, index):
        if self.mode == 'train':
            cls_id = index//150
            cls = self.class_list[cls_id]
            id = index%150+1
            path = self.root + '/' + cls + '/' + f'{id:0>6d}.JPEG'
            img = read_image(path)
            box = self.boxes[cls_id][id-1]
        else:
            index = 149-index
            cls_id = index//30
            cls = self.class_list[cls_id]
            id = index%30+1
            path = self.root + '/' + cls + '/' + f'{id:0>6d}.JPEG'
            img = read_image(path)
            box = self.boxes[cls_id][id-1]
        
        if self.transform:
            img = self.tf(img)

        return img, box


    # End of todo


if __name__ == '__main__':

    dataset = TvidDataset(root='./tiny_vid', mode='train')
    import pdb; pdb.set_trace()