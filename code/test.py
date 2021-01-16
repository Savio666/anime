import model, data
from PIL import Image
import matplotlib.pyplot as plt



def imshow(tensor, unloader, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

if __name__ == "__main__":
    unloader = data.transforms.ToPILImage()
    test_noise = model.get_noise()
    imshow(model.gen(test_noise),unloader= unloader)







