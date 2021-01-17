import model, data, train
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), ):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

if __name__ == "__main__":
    z_dim = 64
    size  = 128
    test_noise = model.get_noise(z_dim, z_dim)
    gen = model.Generator(z_dim)
    gen_save_name = "gen.pt"
    path = '/Users/ipsihou/Documents/anime/{gen_save_name}'
    gen.load_state_dict(torch.load(path))
    unloader = data.transforms.ToPILImage()
    show_tensor_images(gen(test_noise))







