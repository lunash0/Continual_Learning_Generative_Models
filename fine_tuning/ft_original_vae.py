import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch.nn.functional as F
import os
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from visualization import visualize_latent_space

import wandb


class mlp(nn.Module):
    def __init__(self, indim, outdim=10):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(indim, indim // 2)
        self.out = nn.Linear(indim // 2, outdim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.out(x)

    def infer_latent(self, x):
        return self.fc1(x)


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

        self.mlp = mlp(indim=latent_dim, outdim=10)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)

        cls_out = self.mlp(z.detach())

        return self.decode(z), mu, logvar, cls_out


def elbo_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE, KLD


def contrastive_loss(features_cls3, features_9cls, temperature=0.5):
    positive_sim = F.cosine_similarity(features_cls3.unsqueeze(1), features_cls3.unsqueeze(0), dim=2) / temperature
    negative_sim = F.cosine_similarity(features_cls3.unsqueeze(1), features_9cls.unsqueeze(0), dim=2) / temperature

    positive_sim_exp = torch.exp(positive_sim)
    negative_sim_exp = torch.exp(negative_sim)

    positive_loss = -torch.log(positive_sim_exp.diag() / (positive_sim_exp.sum(dim=1) + negative_sim_exp.sum(dim=1)))

    contra_loss = positive_loss.mean()
    return contra_loss


def calculate_ssim(original, reconstructed):
    original = original.squeeze(1)
    # reconstructed = reconstructed.squeeze(1)
    reconstructed = reconstructed.view(-1, 28, 28)

    ssim_scores = []
    for i in range(original.size(0)):
        orig_img = original[i].cpu().numpy()
        recon_img = reconstructed[i].cpu().detach().numpy()
        ssim_scores.append(ssim(orig_img, recon_img, data_range=1.0))

    return sum(ssim_scores) / len(ssim_scores)



def train_vae(dataset, validation_dataset=None, epochs=50, batch_size=200, learning_rate=1e-3, model_path="models/common_vae_ep50_bs200_lr1e-3/common_vae_ep50_bs200_lr1e-3_epoch50_final.pth"):
    cls_loss = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAE()
    model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    # Initialize wandb
    wandb.init(project="unsupervised_generation_using_contrastive_learning", name="only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3",
               entity="")

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_mse = 0
        train_kld = 0
        train_accuracy = 0

        train_ssim = 0
        prog_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}",
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for batch_idx, (data, label) in enumerate(prog_bar):
            optimizer.zero_grad()
            recon_batch, mu, logvar, cls_out = model(data)

            ssim_score = calculate_ssim(data, recon_batch)
            train_ssim += ssim_score

            total_loss, mse_loss, kld_loss = elbo_loss_function(recon_batch, data, mu, logvar)
            loss_cls = cls_loss(cls_out, label)
            total_loss += loss_cls
            total_loss.backward()
            train_loss += total_loss.item()
            train_mse += mse_loss.item()
            train_kld += kld_loss.item()
            optimizer.step()
            acc = (cls_out.argmax(dim=1) == label).sum().item() / len(label)
            train_accuracy += acc

            prog_bar.set_postfix(loss=total_loss.item(), acc=acc)

        avg_loss = train_loss / len(dataloader.dataset)
        avg_mse = train_mse / len(dataloader.dataset)
        avg_kld = train_kld / len(dataloader.dataset)
        avg_accuracy = train_accuracy / len(dataloader)
        avg_ssim = train_ssim / len(dataloader)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, Time: {epoch_time:.2f} seconds")

        # Log train avg_loss, avg_ssim, and avg_accuracy to wandb
        wandb.log({
            "train_avg_loss": avg_loss,
            "train_avg_mse": avg_mse,
            "train_avg_kld": avg_kld,
            "train_avg_accuracy": avg_accuracy,
            "train_avg_ssim": avg_ssim,
        })

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3_epoch{epoch + 1}.pth")
            visualize_latent_space(model, original_test_dataset,
                                   f"only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3_epoch{epoch + 1}",
                                   f"only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3_epoch{epoch + 1}")
        if (epoch + 1) == epochs:
            torch.save(model.state_dict(), f"only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3_epoch{epoch + 1}_final.pth")
            visualize_latent_space(model, original_test_dataset,
                                   f"only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3_epoch{epoch + 1}",
                                   f"only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3_epoch{epoch + 1}")

        if validation_dataset is not None:
            model.eval()
            val_loss = 0
            val_mse = 0
            val_kld = 0
            val_accuracy = 0

            val_ssim = 0
            train_ssim_rare = 0

            val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

            with torch.no_grad():
                for val_data, val_label in val_dataloader:
                    recon_val_batch, mu_val, logvar_val, cls_out_val = model(val_data)
                    total_val_loss, mse_val_loss, kld_val_loss = elbo_loss_function(recon_val_batch, val_data.view(-1, 784),
                                                                               mu_val, logvar_val)
                    loss_cls_val = cls_loss(cls_out_val, val_label)
                    total_val_loss += loss_cls_val
                    val_loss += total_val_loss.item()
                    val_mse += mse_val_loss.item()
                    val_kld += kld_val_loss.item()
                    val_acc = (cls_out_val.argmax(dim=1) == val_label).sum().item() / len(val_label)
                    val_accuracy += val_acc

                    val_ssim_score = calculate_ssim(val_data, recon_val_batch)
                    val_ssim += val_ssim_score

            avg_val_loss = val_loss / len(val_dataloader.dataset)
            avg_val_mse = val_mse / len(val_dataloader.dataset)
            avg_val_kld = val_kld / len(val_dataloader.dataset)
            avg_val_accuracy = val_accuracy / len(val_dataloader)

            avg_val_ssim = val_ssim / len(val_dataloader)
            wandb.log({
                "val_avg_loss": avg_val_loss,
                "val_avg_mse": avg_val_mse,
                "val_avg_kld": avg_val_kld,
                "val_avg_accuracy": avg_val_accuracy,
                "val_avg_ssim": avg_val_ssim,
            })

    wandb.finish()  # End the WandB session
    return model


def gen_imgs(model, dataset, dir_name):
    os.makedirs(dir_name, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i in range(10):
            data, _ = dataset[i]
            gt_img = data.view(28, 28).numpy() * 255  # Scale image to [0, 255]
            gt_img_pil = Image.fromarray(gt_img.astype('uint8'), mode='L')
            gt_img_pil.save(f'{dir_name}/gt_img_{i}.png')

            for iter in range(10):
                recon_img = model(data.unsqueeze(0))[0].view(28, 28).detach().numpy() * 255  # Scale image to [0, 255]
                recon_img_pil = Image.fromarray(recon_img.astype('uint8'), mode='L')
                recon_img_pil.save(f'{dir_name}/gn_img_{i}_{iter}.png')


if __name__ == "__main__":
    from data_preparation import create_datasets

    data_dir = "./mnist_data"
    (original_train_dataset, original_test_dataset,
     only_cls3_train_dataset, only_cls3_test_dataset,
     common_train_dataset, common_test_dataset) = create_datasets(data_dir)

    trained_vae = train_vae(only_cls3_train_dataset, only_cls3_test_dataset)  # #####
    gen_imgs(trained_vae, only_cls3_test_dataset, dir_name="fig_only_cls3_original_vae_finetuning_epoch50_bs200_lr1e-3")
