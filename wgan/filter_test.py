import torch.fft
import torchvision
from torchvision.transforms import transforms
from matplotlib import pyplot as plt
import torch

def low_pass_filter(x: torch.Tensor, cut_bin):
    nX = x.shape[-2]
    nY = x.shape[-1]

    fftn = torch.fft.fftn(x, dim=[1,2])
    if nX == nY:
        n = nX
        maskX = (torch.arange(n**2).reshape(n,n) % n)
        mask = ( maskX <= cut_bin ) 
        mask =  mask * mask.T
    else:
        mask = torch.ones_like(x)
        for i, n in enumerate([nX,nY]):
            maskK = torch.arange(n)

            if i == 1:
                maskK = maskK.T
            else:
                maskK = maskK.unsqueeze(0).unsqueeze(2)
                print(mask.shape)
                print(maskK.shape)

            maskK = ( maskK <= cut_bin )

            print(i)
            mask =   mask * maskK

    # mask = (maskX >= -(cut_bin/2)) * ( maskX <= cut_bin/2 )
    # mask = (maskX >= -(0)) * ( maskX <= cut_bin )
    fftn = fftn*mask
    return torch.fft.ifftn(fftn, dim=[1,2]).real

path = "local/example.png"
image = torchvision.io.read_image(path)/255

image2 = low_pass_filter(image, 500) 
print(image2.shape)

# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)


# ax1.imshow(image.permute(1, 2, 0))
# ax1.axis('off')


plt.imshow(image2.permute(1, 2, 0))
plt.tight_layout()
plt.show()