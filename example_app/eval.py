import argparse
import torch
# import torchvision.datasets as dsets
from contrastive import ContrastiveLoss
from torch.autograd import Variable
import torch.utils.data as data
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from net import SiameseNetwork
from PIL import Image
import cv2
import torchvision
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', type=int, default=120,
                    help='Number of sweeps over the dataset to train')
parser.add_argument('--batchsize', '-b', type=int, default=64,
                    help='Number of images in each mini-batch')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--model', '-m', default='',
                    help='Give a model to test')
parser.add_argument('--train-plot', action='store_true', default=True,
                    help='Plot train loss')
parser.add_argument('--opt', type=str, default="sgd",
                    help='optimizer to use')
parser.add_argument('--custom', type=str, default="hard",
                    help='model to use')
args = parser.parse_args()
OUTPUT="RETRIVED"
class ListDataset(data.Dataset):
    '''Load image/labels/boxes from a list file.
    The list file is like:
      a.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
    '''
    def __init__(self, list_file, transform=None):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str/[str]) path to index file.
          transform: (function) image/box transforms.
        '''
        # self.root = root
        self.transform = transform

        self.fnames = list_file
        # self.boxes = []
        self.labels = []
        #
        # if isinstance(list_file, list):
        #     # Cat multiple list files together.
        #     # This is especially useful for voc07/voc12 combination.
        #     tmp_file = '/tmp/listfile.txt'
        #     os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
        #     list_file = tmp_file

        # with open(list_file) as f:
        #     lines = f.readlines()
        # self.num_imgs = len(self.fnames)
        self.mapping = {"epithelial":0,"fibroblast":1,"inflammatory":2,"others":3}
        for line in self.fnames:
            # splited = line.strip().split()
            # self.fnames.append(splited[0])
            # num_boxes = (len(splited) - 1) // 5
            # box = []
            label = []
            for key in self.mapping.keys():
                if key in line:
                    label.append(self.mapping[key])
            # self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.
        Args:
          idx: (int) image index.
        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # boxes = self.boxes[idx].clone()  # use clone to avoid any potential change.
        labels = self.labels[idx].clone()

        # if self.transform:
        #     img, boxes, labels = self.transform(img, boxes, labels)
        return torchvision.transforms.functional.to_tensor(torchvision.transforms.functional.resize(img,(28,28))), fname,labels

    def __len__(self):
        return len(self.fnames)

train_data  = ListDataset(list_file=glob("/home/god/MedHack/CRC_Dataset_TechFesia/gallery/*/**"))
test_data  = ListDataset(list_file=glob("test/*/**"))
train_data = torch.utils.data.DataLoader(
    train_data,
    batch_size=1, shuffle=False)
test_data = torch.utils.data.DataLoader(
    test_data,
    batch_size=1, shuffle=True)
k = 500
saved_model = torch.load('model-epoch-60.pth',map_location=lambda storage, loc: storage)
model = SiameseNetwork()
model.load_state_dict(saved_model)
def test(model):
    model.eval()
    # all = []
    names = []
    train_vectors = []
    labels_all = []
    for batch_idx, (x,file_name, labels) in enumerate(train_data):
        # if args.cuda:
        #     x = x.cuda(),
        x = Variable(x, volatile=True)
        output = model.forward_once(x)
        # print(output)
        train_vectors.extend(output.data.cpu().numpy().tolist())
        labels_all.extend(labels.cpu().numpy().tolist())
        names.append(file_name[0])
        # print(train_vectors)
    train_vectors = np.array(train_vectors)
    # data = {"vectors":train_vectors,"labels":labels_all,"names"}
    # names.append(file_name[0])
    # print(train_vectors.shape,)
    count = [0,0,0,0]
    classes = [set(),set(),set(),set()]
    retrie = set()
    # rec = [0,0,0,0]
    pre = 0
    for batch_idx, (x,file_name,labels) in enumerate(test_data):
        # if args.cuda:
        #     x = x.cuda(),
        # path_file = os.path.join("RETRIVED",file_name.replace(".jpeg",""))
        # os.makedirs(path_file)
        x = Variable(x, volatile=True)
        output = model.forward_once(x).detach().cpu().numpy()
        distance = np.sqrt(np.sum(np.power(train_vectors-output,2),axis=1))
        # print(distance.shape,train_vectors.shape)
        # print()
        mink = np.argsort(distance)[:k]
        # print("MINDITANCE:{}".format(distance[mink[0]]))
        # print("DISTANCE:{}",distance)
        retrieved = np.flatnonzero(distance < 0.3)

        # print("retrieved:{}".format(retrieved.shape))
        # mink = np.argsort(distance)[:k]
        # print("HIII",distance,labels_all,labels)
        tp = 0
        # print(distance[mink])
        for i in retrieved:
            retrie.add(file_name[0])
            # os.link(names[i],os.path_file)
            if int(labels[0]) == int(labels_all[i][0]):
                classes[int(labels_all[i][0])].add(file_name[0])
                # tp+=1
        # pre+=tp/len(retrie)
        # rec+=tp/
        # print("Precision:{}".format(tp/len(retrieved)))
    print("AP of epithelial:{}".format(len(classes[0])/len(retrie)))
    print("AP of fibroblast:{}".format(len(classes[1])/len(retrie)))
    print("AP of inflammatory:{}".format(len(classes[2])/len(retrie)))
    print("AP of others:{}".format(len(classes[3])/len(retrie)))
    # print("Recall")
    print("Recall of epithelial:",len(classes[0])/5792)
    print("Recall of fibroblast:",len(classes[1])/4284)
    print("Recall of inflammatory:",len(classes[2])/5093)
    print("Recall of others:",len(classes[3])/1529)

with torch.no_grad():
    test(model)
        # train_vectors.extend(output.data.cpu().numpy().tolist())
        # labels_all.extend(labels.cpu().numpy().tolist())


        # all_labels.extend(labels.data.cpu().numpy().tolist())
