from data_loader import BrainDataset
from nii_handler import Nii_Handler


data_loader = BrainDataset("local", None, (50,50,50)
                           )

for lr, hr in data_loader:
    print(hr.shape)
    Nii_Handler.save(hr , "hr.nii.gz")
    Nii_Handler.save(lr , "lr.nii.gz")

    break
