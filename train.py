import torch
from torch import nn
import data, model, config
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

torch.manual_seed(0) # Set for testing purposes, please do not change!

# function to demonstrate the anime images from tensor
def save_tensor_images_with_step(image_tensor, cur_step, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.savefig(f"./img_result/Step {cur_step}.jpg")
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
    gen = model.Generator(config.z_dim).to(config.device)
    crit = model.Critic().to(config.device)
    batch_size=config.batch_size
    gen_opt=torch.optim.Adam(gen.parameters(), lr=config.lr, betas=(config.beta_1,config.beta_2))
    crit_opt=torch.optim.Adam(crit.parameters(), lr=config.lr, betas=(config.beta_1,config.beta_2))
    gen = gen.apply(weights_init)
    crit = crit.apply(weights_init)
    mean_generator_loss = 0
    cur_step = 0
    generator_losses = []
    critic_losses = []
    # training the model
    for e in range(config.n_epochs):
        for real, _ in tqdm(data.dataloader):
            size = len(real)
            real = real.to(config.device)
            mean_iteration_critic_loss = 0
            for _ in range(config.crit_repeats):
                crit_opt.zero_grad()
                noise = model.get_noise(size, config.z_dim, device=config.device)
                fake = gen(noise)
                fake_pred = crit(fake.detach())
                real_pred = crit(real)
                epsilon = torch.rand(len(real), 1, 1, 1, device=config.device, requires_grad=True)
                gradient = get_gradient(crit, real, fake.detach(), epsilon)
                gp = gradient_penalty(gradient)
                crit_loss = get_crit_loss(fake_pred, real_pred, gp, config.c_lambda)
                mean_iteration_critic_loss += crit_loss.item() / config.crit_repeats
                crit_loss.backward(retain_graph=True)
                crit_opt.step()
            critic_losses += [mean_iteration_critic_loss]
            gen_opt.zero_grad()
            fake_noise_2 = model.get_noise(size, config.z_dim, device=config.device)
            fake_2 = gen(fake_noise_2)
            fake_pred_2 = crit(fake_2)
            gen_loss = get_gen_loss(fake_pred_2)
            gen_loss.backward()
            gen_opt.step()
            generator_losses += [gen_loss.item()]
            mean_generator_loss += gen_loss.item() / config.display_step
            if cur_step % config.display_step == 0 and cur_step > 0:
                gen_mean = sum(generator_losses[-config.display_step:]) / config.display_step
                crit_mean = sum(critic_losses[-config.display_step:]) / config.display_step
                print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
                save_tensor_images_with_step(fake, cur_step)
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
    path = f"./weights/{config.gen_save_name}"
    torch.save(gen.state_dict(), path)
    path = f"./weights/{config.crit_save_name}"
    torch.save(crit.state_dict(), path)






