from model.generator import Generator
from model.discriminator import Discriminator
import torch
import torch.nn
import torchvision

vgg = torchvision.models.vgg16(pretrained=True)
vgg.classifier = vgg.classifier[:-1]
vgg.eval()
vgg = vgg.to("cuda:0")
vgg = vgg.requires_grad_(False)
class ImageGenerator(Generator):
    def __init__(self, n_dense_blocks, layers_per_dense_block, out_channels_per_layer_dense=12):
        super(ImageGenerator,self).__init__(2, 3, 3, n_dense_blocks, layers_per_dense_block, out_channels_per_layer_dense)

        self.encoder = torch.nn.Linear(4096,64*64)
        

    def forward(self, x):
        y = x

        vgg_enc = self.encoder(vgg(x)).reshape(x.shape[0],1,64,64)
        vgg_enc = torch.nn.functional.interpolate(vgg_enc,(x.shape[2],x.shape[3]))


        for i in range(len(self.dense_blocks)):
            y2 = self.dense_blocks[i](self.compressors[i](y)) 
            y2 += vgg_enc*0.2
            if i != 0:
                y = torch.cat([y, y2], dim=1)
            else:
                y = y2

        return self.reconstruction(y)


class ImageDiscriminator(Discriminator):
    def __init__(self, spatial_size):
        super(ImageDiscriminator,self).__init__(2, 3, spatial_size)
        self.encoder = torch.nn.Linear(4096,64*64)

    def forward(self, x):
        vgg_enc = self.encoder(vgg(x)).reshape(x.shape[0],1,64,64)
        vgg_enc = torch.nn.functional.interpolate(vgg_enc,(x.shape[2],x.shape[3]))

        return super(ImageDiscriminator,self).forward( x+0.5*vgg_enc)