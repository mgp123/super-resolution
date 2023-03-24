import SimpleITK as sitk
import torch 

class Nii_Handler:

    @staticmethod
    def load(path):
        itkimage = sitk.ReadImage(path)
        numpyImage = sitk.GetArrayFromImage(itkimage)*1.0
        numpyImage = torch.from_numpy(numpyImage)   
        return numpyImage
    
    @staticmethod
    def save(x:torch.Tensor, path:str):
        if len(x.shape) == 4:
            if x.shape[0] == 0:
                x = x.squeeze(0)
            else:
                raise ValueError(f"Can only handle 3d tensors but got {x.shape}")
        result_image = sitk.GetImageFromArray(x.cpu())
        sitk.WriteImage(result_image, path)
