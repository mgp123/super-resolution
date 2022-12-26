from os import makedirs
from os.path import exists
import cv2
import numpy as np
import torch
from tqdm import tqdm
from data_loader import get_data_loaders
from torch.utils.tensorboard import SummaryWriter
import torch.autograd
import math
from model.generator import Generator
from random import randrange
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import torchvision.io
from torchvision.transforms import transforms
import torch.nn.functional as F


def make_superres(g, lr, overlap, spatial_size=64):
    with torch.no_grad():
        lr = lr.type(torch.FloatTensor)
        lr = lr.to("cuda:0")
        print("lr", lr.shape)

        # return g(lr.view(1,3,64,64))[0]
        B, F, W, H = lr.shape
        sr = lr.unfold(dimension=1, size=spatial_size[0], step=spatial_size[0]-overlap)
        sr = sr.unfold(dimension=2, size=spatial_size[1], step=spatial_size[1]-overlap)
        sr = sr.unfold(dimension=3, size=spatial_size[2], step=spatial_size[2]-overlap)

        n = sr.shape[1:-3]
        sr = sr.reshape(1,-1, *spatial_size)
        sr = torch.transpose(sr, 0, 1)


        print("lr", lr.shape)

        print("sr", sr.shape)

        batch_size = 15
        srs = torch.split(sr,batch_size)
        srs2 = []
        for x in tqdm(srs):
            srs2.append(g(x))
        print("srs2 sample", srs2[0].shape)
        sr = torch.cat(srs2,dim=0)
        # sr = torch.flatten(sr, start_dim=0, end_dim=1)
        print("sr!!!!!!!!", sr.shape)

        sr = torch.clamp(sr,min=0,max=1)

        sr = torch.transpose(sr, 0, 1)
        sr = sr.reshape(3,*n, *spatial_size)

        print("sr", sr.shape)

        # slow method but it doesnt matter because its done on inference
        res = torch.zeros(3,F,W,H, device=sr.device)
        counter = torch.zeros(3,F,W,H, device=sr.device)

        step = [x-overlap for x in spatial_size]

        for i in range(sr.shape[1]):
            for j in range(sr.shape[2]):
                for k in range(sr.shape[3]):

                    ki = step[0] * i
                    kj = step[1] * j
                    kk = step[2] * k

                    res[:,ki:ki+spatial_size[0], kj:kj+spatial_size[1],kk:kk+spatial_size[2]] += sr[:,i,j,k]
                    counter[:,ki:ki+spatial_size[0], kj:kj+spatial_size[1],kk:kk+spatial_size[2]] += 1
        
        res = res/counter

        print("sr", sr.shape)

    return res

# print(get_n_params(d))

model_path = "saved_weights/trainning_video.model"
spatial_size = 64
batch_size = 1
low_pass_filter_cut_bin = 3
g = Generator(
    dimensions=3,
    in_channels=1,
    out_channels=3,
    n_dense_blocks=8,
    layers_per_dense_block=6
)


if exists(model_path):
    save = torch.load(model_path)
    g.load_state_dict(save['generator_dict'])
else:
    raise ValueError


g = g.to("cuda:0")
g = g.eval()
video_path = "local/scenes/0039.mp4"
cap = cv2.VideoCapture(video_path)

imgs = []



selected_frames = []
current_frame = 0
start_frame = 0
collected_frames = 10
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
with tqdm(total=min(length,collected_frames)) as pbar:
    while(cap.isOpened() and current_frame < 10 ):
            pbar.update(1)
            ret, frame = cap.read()
            if ret == True:
                current_frame += 1
                if current_frame >= start_frame:
                    # frame = frame[start_h:start_h+crop_size,start_w:start_w+crop_size,:]
                    frame = np.flip(frame,axis=2)
                    # frame = np.mean(frame,axis=2,keepdims=True)
                    selected_frames.append(frame)
            else:
                break

cap.release()
cv2.destroyAllWindows()
print(len(selected_frames))
t = np.stack(selected_frames)
t = torch.tensor(t)*1.0/255
plt.imshow(t[0])
plt.show()
t = torch.mean(t,dim=3,keepdim=True)

t = t.permute((3,0,1,2))
t = torchvision.transforms.functional.gaussian_blur(t,21)

t = t.permute((1,2,3,0))
plt.imshow(t[0], cmap="gray")
plt.show()
t = t.permute((3,0,1,2))

print("running enchance...")
s = make_superres(g,t,0,(10,76,76)).cpu()

print("reconstructed shape", s.shape)
s = s.permute((1,2,3,0))
plt.imshow(s[0])
plt.show()
