import SimpleITK as sitk
from matplotlib import pyplot as plt
import os

import torch
from torchvision.transforms.functional import gaussian_blur
from data_loader import low_pass_filter

# robado de algun lado
def simpleitk_wrapper(dicom_dir: str, output_file: str, **kwargs):
    output = None
    data_directory = dicom_dir
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(data_directory)
    if not series_IDs:
        print("ERROR: given directory \""+data_directory+"\" does not contain a DICOM series.")
        raise
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(data_directory, series_IDs[0])

    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    image3D = series_reader.Execute()

    sitk.WriteImage(image3D, output_file)
        

def create_dataset(samples_path, dataset_path):

 for i, sample_folder in enumerate(os.listdir(samples_path)):
    simpleitk_wrapper(samples_path+"/"+ sample_folder,  dataset_path + "/" +  "sample_" + str(i) + ".tif" )

# simpleitk_wrapper("local/MRt1", "local/out.tif")

# itkimage = sitk.ReadImage("local/dataset/sample_0.tif")
# numpyImage = sitk.GetArrayFromImage(itkimage)
# p = 196
# numpyImage = numpyImage[:p,:p,:p]
# print(numpyImage.shape)

# plt.imshow(-numpyImage[127,:,:], cmap="Greys")

# plt.show()
# i = 0
# figure, axs = plt.subplots(2, 2, sharey=True)
# for axs_row in axs:
#     for ax in (axs_row):
#         numpyImage = low_pass_filter(torch.tensor(numpyImage), cut_bin=196-i*2)
#         ax.imshow(-numpyImage[127,:,:], cmap="Greys")
#         ax.set_title(str("bins removed " + str(i*2)))
#         i += 1
#         ax.axis('off')

# figure.tight_layout(pad=0.1)
# plt.savefig("fft_cut_example.png")
# plt.show()

# for axs_row in axs:
#     for ax in (axs_row):
#         numpyImage = gaussian_blur(torch.tensor(numpyImage), kernel_size= 1 + i*2*3)
#         ax.imshow(-numpyImage[127,:,:], cmap="Greys")
#         ax.set_title(str("kernel size " + str(1+i*2*3)))
#         i += 1
#         ax.axis('off')

# figure.tight_layout(pad=0.1)
# plt.savefig("gaussain.png")
# plt.show()
# create_dataset("local/samples", "local/dataset")