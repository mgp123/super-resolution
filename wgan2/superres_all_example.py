from os import makedirs
from data_loader import BrainDataset
from nii_handler import Nii_Handler
from gan import WGAN
import glob 
import re

if __name__ == "__main__":
    save_path = "sr_outputs"
    model_paths = glob.glob(f"**/*.ckpt", recursive=True) 
    makedirs(save_path, exist_ok=True)

    for model_path in model_paths:
        model = WGAN.load_from_checkpoint(model_path)
        version = (re.findall(r'version_\d+', model_path)[0])
        slices = model.training_parameters["low_res_size"][0]
        makedirs(f"{save_path}/{slices}_slices/{version}", exist_ok=True)
        dataset = BrainDataset("local", None, low_size=model.training_parameters["low_res_size"])
        lr, hr = next(iter(dataset))
        model = model.to("cuda:0")
        sr = model._superres(lr.unsqueeze(0).unsqueeze(0))
        sr = sr.squeeze(0).squeeze(0)
        Nii_Handler.save(sr, f"{save_path}/{slices}_slices/{version}/out_from_{slices}_slices.nii.gz")
        Nii_Handler.save(lr, f"{save_path}/{slices}_slices/{version}/in_with_{slices}_slices.nii.gz")
        model = model.cpu()
        del model