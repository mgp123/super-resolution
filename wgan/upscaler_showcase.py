from data_loader import get_data_loaders, get_data_loaders_dummy, low_pass_filter
from model.generator import Generator
import torch 
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap

def plot_data_3d(dataset, samples = 1, transformation = None):

    ax = plt.axes(projection='3d')
    iterator=iter(dataset)
    samples_hr = next(iterator)[0]
    cube_size = samples_hr.shape[-1]

    cmap = plt.cm.Greys

    decreased_alpha_cmap = cmap(torch.arange(cmap.N))
    decreased_alpha_cmap[:,-1] = torch.linspace(0, 0.05, cmap.N)
    # decreased_alpha_cmap[:,-1][-1] = 1.0
    decreased_alpha_cmap = ListedColormap(decreased_alpha_cmap)

    for _ in range(samples):
        if transformation != None:
            samples_hr = transformation(samples_hr)

        X = (torch.arange(cube_size**3).reshape(cube_size,cube_size,cube_size) % cube_size)
        X = X/cube_size - 0.5
        # X = X[l*5] + l*5
        Y = torch.transpose(X, 0, 1)
        Z = torch.transpose(X, 0, 2)

        X = torch.flatten(X)
        Y = torch.flatten(Y)
        Z = torch.flatten(Z)

        X = torch.arange(cube_size**3)%cube_size**2
        Y = torch.arange(cube_size**3)%cube_size
        Z = torch.arange(cube_size**3)

        ax.scatter(X, Y,Z, c=samples_hr[0,0],  cmap=decreased_alpha_cmap)
        plt.show()


        samples_hr = next(iterator)[0]



model_path = "saved_weights/trainning.model"
spatial_size = 40
batch_size = 1
low_pass_filter_cut_bin = 3


g = Generator(
    in_channels=1,
    n_dense_blocks=8,
    layers_per_dense_block=4
)

data_loader_train, data_loader_test = get_data_loaders_dummy(batch_size, spatial_size)
save = torch.load(model_path)
g.load_state_dict(save['generator_dict'])
del save
g.eval()



with torch.no_grad():
    # plot_data_3d(data_loader_test, transformation=None)
    # plot_data_3d(data_loader_test, transformation=low_pass_filter)
    # plot_data_3d(data_loader_test, transformation=g)

    for samples_hr, _ in data_loader_test:
        samples_lr = low_pass_filter(samples_hr, cut_bin=low_pass_filter_cut_bin)
        samples_sr = g(samples_lr)
        k = 3

        samples_to_plot = [samples_lr, samples_sr, samples_hr]
        figure, axs = plt.subplots(k, len(samples_to_plot))
        cube_size = samples_hr.shape[-1]
        factor = (cube_size-1)/k

        for i, sample in enumerate(samples_to_plot):
            mlow = torch.min(torch.tensor([torch.min(x) for x in samples_to_plot]))
            mmax = torch.max(torch.tensor([torch.max(x) for x in samples_to_plot]))
            for j in range(k):
                s = sample[0,0]
                s = torch.transpose(s, 0, j%3)
                # s = s[:,5 + j*10,:]
                # s = s[int(factor*j)]
                s = s[19+ 5*(j//3)]

                # s = s[0,0,:,:,19]

                axs[j][i].imshow(s, vmin=mlow, cmap="Greys")
                axs[j][i].axis('off')

        for ax, col in zip(axs[0], ["LR", "SR", "HR"]):
            ax.set_title(col)

        figure.tight_layout(pad=0.3)
        # plt.savefig("test.png",bbox_inches='tight')
        plt.show()

        break



