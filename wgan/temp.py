import torch
import torchvision

from model.image_model import ImageDiscriminator, ImageGenerator

g = ImageGenerator(
    n_dense_blocks=8,
    layers_per_dense_block=6
)

d = ImageDiscriminator(spatial_size=(75,64))

# model=torchvision.models.vgg16(pretrained=True)
# # model = torchvision.models.mobilenet_v2(pretrained=True)
# model.classifier = model.classifier[:-1]
x = torch.randn((1,3,75,64))
features=g(x)
print(features.shape)
y = d(features)
print(y.shape)


# print(features.shape)