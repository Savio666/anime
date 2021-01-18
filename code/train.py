import torch
from torch import nn
import data
import model
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

torch.manual_seed(0) # Set for testing purposes, please do not change!

# function to demonstrate the anime images from tensor
def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

# initialize the weights
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

# get the gradient
def get_gradient(crit, fake, real, epsilon):
    mixed_image = real * epsilon + fake * (1-epsilon)
    mixed_score = crit(mixed_image)
    gradient = torch.autograd.grad(
        inputs = mixed_image,
        outputs = mixed_score,
        grad_outputs=torch.ones_like(mixed_score),
        create_graph = True,
        retain_graph = True
    )[0]
    return gradient

# calculate the gradient penalty
def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

# calculate the loss for generator
def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss

# calculate the loss for critic
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss

# Function to keep track of gradients for visualization purposes
def make_grad_hook():
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook

if __name__ == "__main__":

    # specify the parameters
    z_dim = 64
    device = 'cpu'
    gen = model.Generator(z_dim).to(device)
    crit = model.Critic().to(device)
    batch_size=data.batch_size
    lr=0.0002
    beta_1 = 0.5
    beta_2 = 0.999
    c_lambda = 10
    crit_repeats = 5
    n_epochs = 20
    mean_generator_loss = 0
    display_step = 50
    gen_opt=torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1,beta_2))
    crit_opt=torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1,beta_2))
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)
    cur_step = 0
    generator_losses = []
    critic_losses = []
    # training the model
    for e in range(n_epochs):
        for real, _ in tqdm(data.dataloader):
            size = len(real)
            real = real.to(device)
            mean_iteration_critic_loss = 0
            for _ in range(crit_repeats):
                crit_opt.zero_grad()
                noise = model.get_noise(size, z_dim, device=device)
                fake = gen(noise)
                fake_pred = crit(fake.detach())
                real_pred = crit(real)
                epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(fake_pred, real_pred, gp, c_lambda)
                mean_iteration_critic_loss += crit_loss.item() / crit_repeats
                crit_loss.backward(retain_graph=True)
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]
            gen_opt.zero_grad()
            fake_noise_2 = model.get_noise(size, z_dim, device=device)
            fake_2 = gen(fake_noise_2)
            fake_pred_2 = crit(fake_2)
            gen_loss = get_gen_loss(fake_pred_2)
            gen_loss.backward()
            gen_opt.step()
            generator_losses += [gen_loss.item()]
            mean_generator_loss += gen_loss.item() / display_step
            if cur_step % display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-display_step:]) / display_step
                crit_mean = sum(critic_losses[-display_step:]) / display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                show_tensor_images(fake)
                show_tensor_images(real)
                step_bins = 20
                num_examples = (len(generator_losses) // step_bins) * step_bins
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Generator Loss"
                )
                plt.plot(
                    range(num_examples // step_bins),
                    torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                    label="Critic Loss"
                )
                plt.legend()
                plt.show()
            cur_step += 1
    # saving the parameters of the model
    gen_save_name = "gen.pt"
    path = '/Users/ipsihou/Documents/anime/{gen_save_name}'
    torch.save(gen.state_dict(), path)
    crit_save_name = "crit.pt"
    path = '/Users/ipsihou/Documents/anime/{crit_save_name}'
    torch.save(crit.state_dict(), path)






