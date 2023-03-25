from data_loader import BrainDataset
from nii_handler import Nii_Handler
from gan import WGAN


model = WGAN.load_from_checkpoint("runs/lightning_logs/version_0/checkpoints/epoch=13-step=6300.ckpt")
dataset = BrainDataset("local", None, low_size=model.training_parameters["low_res_size"])
lr, hr = next(iter(dataset))
model = model.to("cuda:0")
sr = model._superres(lr.unsqueeze(0).unsqueeze(0),(16,16,16) )
print(sr.shape)
sr = sr.squeeze(0).squeeze(0)
Nii_Handler.save(sr, "out.nii.gz")

Nii_Handler.save(lr, "in.nii.gz")