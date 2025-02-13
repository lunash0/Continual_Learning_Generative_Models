import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from visualization import visualize_latent_space

import wandb
import argparse

parser = argparse.ArgumentParser(description='fine-tuning contra VAE or vanilla VAE for rare class(class 5) using pre-trained model')
parser.add_argument('--model_type', type=str, default="fine-tuning_vanilla_vae", choices=["fine-tuning_vanilla_vae", "fine-tuning_contra_vae"])
parser.add_argument('--dataset_type', type=str, default="rare", choices=["rare"])
parser.add_argument('--train_dataset', type=str, default="rare_train_dataset")  # ,choices=[], help=""
parser.add_argument('--test_dataset', type=str, default="rare_test_dataset")  # choices=[], help=""
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--lr', type=int, default=1e-3)
parser.add_argument('--pretrained_model', type=str, default="models/common_vanilla_vae_ep50_bs200_lr0.001/common_vanilla_vae_ep50_bs200_lr0.001_epoch50_final.pth")
parser.add_argument('--contra_lambda', type=float, default=1.0, help="This is a hyperparameter to control contrastive loss")

args = parser.parse_args()

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

        return self.decode(z), mu, logvar, cls_out, z


def elbo_loss_function(recon_x, x, mu, logvar):
    MSE = nn.functional.mse_loss(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD, MSE, KLD


def contrastive_loss(features_cls3, features_common, temperature=0.5):
    positive_sim = F.cosine_similarity(features_cls3.unsqueeze(1), features_cls3.unsqueeze(0), dim=2) / temperature
    negative_sim = F.cosine_similarity(features_cls3.unsqueeze(1), features_common.unsqueeze(0), dim=2) / temperature

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


def fine_tuning_vanilla_vae(dataset, validation_dataset=None, original_test_dataset=None, epochs=50, batch_size=200, learning_rate=1e-3,
                            model_path="models/common_vanilla_vae_ep50_bs200_lr0.001/common_vanilla_vae_ep50_bs200_lr0.001_epoch50_final.pth"):
    cls_loss = nn.CrossEntropyLoss()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = VAE()
    model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # rare_train_dataloader = DataLoader(rare_train_dataset, batch_size=batch_size, shuffle=True)
    rare_test_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    all_cls_test_dataloader = DataLoader(original_test_dataset, batch_size=batch_size, shuffle=False)

    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_mse = 0
        train_kld = 0
        train_accuracy = 0

        train_ssim_rare = 0
        train_ssim_all_cls = 0
        prog_bar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch + 1}/{epochs}",
                        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for batch_idx, (data, label) in enumerate(prog_bar):
            optimizer.zero_grad()
            recon_batch, mu, logvar, cls_out, _ = model(data)

            ssim_score = calculate_ssim(data, recon_batch)
            train_ssim_rare += ssim_score

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

        for batch_idx, (data, _) in enumerate(all_cls_test_dataloader):
            recon_batch, _, _, _, _ = model(data)
            ssim_score = calculate_ssim(data, recon_batch)
            train_ssim_all_cls += ssim_score

        avg_ssim_all_cls = train_ssim_all_cls / len(all_cls_test_dataloader)

        avg_loss = train_loss / len(dataloader.dataset)
        avg_mse = train_mse / len(dataloader.dataset)
        avg_kld = train_kld / len(dataloader.dataset)
        avg_accuracy = train_accuracy / len(dataloader)
        avg_ssim_rare = train_ssim_rare / len(dataloader)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, SSIM: {avg_ssim_rare:.4f}, Time: {epoch_time:.2f} seconds")

        wandb.log({
            "train_avg_loss": avg_loss,
            "train_avg_mse": avg_mse,
            "train_avg_kld": avg_kld,
            "train_avg_accuracy": avg_accuracy,
            "train_avg_ssim_all_classes": avg_ssim_all_cls,
            "train_avg_ssim_class_five": avg_ssim_rare
        })

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}_epoch{epoch + 1}.pth")
            visualize_latent_space(model, original_test_dataset, f"{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}_epoch{epoch + 1}", f"{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}_epoch{epoch + 1}")
        if (epoch + 1) == epochs:
            torch.save(model.state_dict(), f"{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}_epoch{epoch + 1}_final.pth")
            visualize_latent_space(model, original_test_dataset, f"{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}_epoch{epoch + 1}", f"{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}_epoch{epoch + 1}")

        if validation_dataset is not None:
            model.eval()
            val_loss = 0
            val_mse = 0
            val_kld = 0
            val_accuracy = 0

            val_ssim_rare = 0
            val_ssim_all_cls = 0
            val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

            with torch.no_grad():
                for val_data, val_label in val_dataloader:
                    recon_val_batch, mu_val, logvar_val, cls_out_val, _ = model(val_data)

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
                    val_ssim_rare += val_ssim_score

                    for _, (val_data, _) in enumerate(all_cls_test_dataloader):
                        recon_val_batch, _, _, _, _ = model(val_data)
                        val_ssim_score = calculate_ssim(val_data, recon_val_batch)
                        val_ssim_all_cls += val_ssim_score

            val_avg_ssim_all_cls = val_ssim_all_cls / len(all_cls_test_dataloader)

            avg_val_loss = val_loss / len(val_dataloader.dataset)
            avg_val_mse = val_mse / len(val_dataloader.dataset)
            avg_val_kld = val_kld / len(val_dataloader.dataset)
            avg_val_accuracy = val_accuracy / len(val_dataloader)
            avg_val_ssim_rare = val_ssim_rare / len(val_dataloader)

            wandb.log({
                "val_avg_loss": avg_val_loss,
                "val_avg_mse": avg_val_mse,
                "val_avg_kld": avg_val_kld,
                "val_avg_accuracy": avg_val_accuracy,
                "val_avg_ssim_all_classes": val_avg_ssim_all_cls,
                "val_avg_ssim_class_five": avg_val_ssim_rare
            })

    wandb.finish()
    return model

def fine_tuning_contra_vae(rare_train_dataset, rare_test_dataset=None, common_train_dataset=None,
                           common_test_dataset=None, original_test_dataset_to_visualize=None, contra_lamba=1.0, epochs=50, batch_size=200, learning_rate=1e-3,
                           model_path="models/common_vanilla_vae_ep50_bs200_lr0.001/common_vanilla_vae_ep50_bs200_lr0.001_epoch50_final.pth"):
    cls_loss = nn.CrossEntropyLoss()

    common_train_dataloader = DataLoader(common_train_dataset, batch_size=batch_size, shuffle=True)
    rare_train_dataloader = DataLoader(rare_train_dataset, batch_size=batch_size, shuffle=True)

    if rare_test_dataset is not None:
        rare_test_dataloader = DataLoader(rare_test_dataset, batch_size=batch_size, shuffle=False)
    if common_test_dataset is not None:
        common_test_dataloader = DataLoader(common_test_dataset, batch_size=batch_size, shuffle=False)

    model = VAE()
    model.load_state_dict(torch.load(model_path))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_mse = 0
        train_kld = 0
        train_contrastive_loss = 0
        train_accuracy = 0

        train_ssim = 0
        train_ssim_rare = 0

        for batch_common, batch_rare in zip(common_train_dataloader, rare_train_dataloader):
            optimizer.zero_grad()

            mu_common, logvar_common = model.encode(batch_common[0].view(-1, 784))
            z_common = model.reparameterize(mu_common, logvar_common)
            z_common = z_common.detach()

            mu_rare, logvar_rare = model.encode(batch_rare[0].view(-1, 784))
            z_rare = model.reparameterize(mu_rare, logvar_rare)

            recon_rare = model.decoder(z_rare)

            recon_loss, mse_loss, kld_loss = elbo_loss_function(recon_rare, batch_rare[0].view(-1, 784),  mu_rare, logvar_rare)
            train_mse += mse_loss.item()
            train_kld += kld_loss.item()

            contrastive_loss_value = contrastive_loss(z_rare, z_common)
            train_contrastive_loss += contrastive_loss_value.item()

            total_loss = recon_loss + contra_lamba * contrastive_loss_value

            cls_output = model.mlp(z_rare)
            loss_cls = cls_loss(cls_output, batch_rare[1])
            total_loss += loss_cls

            ssim_value = calculate_ssim(recon_rare.view(-1, 28, 28), batch_rare[0].view(-1, 28, 28))
            train_ssim += ssim_value.item()

            total_loss.backward()
            optimizer.step()

            acc = (cls_output.argmax(dim=1) == batch_rare[1]).sum().item() / len(batch_rare[1])
            train_accuracy += acc

            train_loss += total_loss.item()

        for batch_idx, (data, _) in enumerate(rare_train_dataloader):
            recon_batch, _, _, _ = model(data)
            ssim_score = calculate_ssim(data, recon_batch)
            train_ssim_rare += ssim_score

        avg_ssim_rare = train_ssim_rare / len(rare_train_dataloader)

        avg_loss = train_loss / len(rare_train_dataloader.dataset)
        avg_mse = train_mse / len(rare_train_dataloader.dataset)
        avg_kld = train_kld / len(rare_train_dataloader.dataset)
        avg_accuracy = train_accuracy / len(rare_train_dataloader)
        avg_contra_loss = train_contrastive_loss / len(rare_train_dataloader)
        avg_ssim = train_ssim / len(rare_train_dataloader)

        epoch_time = time.time() - start_time
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Contrastive Loss: {avg_contra_loss:.4f}, SSIM: {avg_ssim:.4f}, Time: {epoch_time:.2f} seconds")

        wandb.log({
            "train_avg_loss": avg_loss,
            "train_avg_mse": avg_mse,
            "train_avg_kld": avg_kld,
            "train_avg_accuracy": avg_accuracy,
            "train_avg_contrastive_loss": avg_contra_loss,
            "train_avg_ssim_all_classes": avg_ssim,
            "train_avg_ssim_class_five": avg_ssim_rare
        })

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{args.dataset_type}_{args.model_type}_ep{args.epoch}_bs{args.batch_size}_lr{args.lr}_contra-lambda{args.contra_lamba}_epoch{epoch + 1}.pth")
            visualize_latent_space(model, original_test_dataset_to_visualize, f"{args.dataset_type}_{args.model_type}_ep{args.epoch}_bs{args.batch_size}_lr{args.lr}_contra-lambda{args.contra_lamba}_epoch{epoch + 1}",
                                   f"{args.dataset_type}_{args.model_type}_ep{args.epoch}_bs{args.batch_size}_lr{args.lr}_contra-lambda{args.contra_lamba}_epoch{epoch + 1}")

        if (epoch + 1) == epochs:
            torch.save(model.state_dict(), f"c{args.dataset_type}_contra_vae_ep{args.epoch}_bs{args.batch_size}_lr{args.lr}_contra-lambda{args.contra_lamba}_epoch{epoch + 1}_final.pth")
            visualize_latent_space(model, original_test_dataset_to_visualize, f"{args.dataset_type}_{args.model_type}_ep{args.epoch}_bs{args.batch_size}_lr{args.lr}_contra-lambda{args.contra_lamba}_epoch{epoch + 1}",
                                   f"{args.dataset_type}_{args.model_type}_ep{args.epoch}_bs{args.batch_size}_lr{args.lr}_contra-lambda{args.contra_lamba}_epoch{epoch + 1}")

        if rare_test_dataset is not None and common_test_dataset is not None:
            model.eval()

            val_loss = 0
            val_mse = 0
            val_kld = 0
            val_accuracy = 0
            val_contrastive_loss = 0

            val_ssim = 0
            val_ssim_rare = 0

            for val_batch_common, val_batch_rare in zip(common_test_dataloader, rare_test_dataloader):
                optimizer.zero_grad()

                with torch.no_grad():
                    mu_common, logvar_common = model.encode(val_batch_common[0].view(-1, 784))
                    z_common = model.reparameterize(mu_common, logvar_common)
                    z_common = z_common.detach()

                    mu_rare, logvar_rare = model.encode(val_batch_rare[0].view(-1, 784))
                    z_rare = model.reparameterize(mu_rare, logvar_rare)

                    recon_rare = model.decoder(z_rare)

                    elbo_total_loss, mse_val_loss, kld_val_loss = elbo_loss_function(recon_rare, val_batch_rare[0].view(-1, 784),  mu_rare, logvar_rare)
                    val_mse += mse_val_loss.item()
                    val_kld += kld_val_loss.item()

                    contrastive_loss_value = contrastive_loss(z_rare, z_common)
                    val_contrastive_loss += contrastive_loss_value.item()

                    val_total_loss = elbo_total_loss + contra_lamba * contrastive_loss_value

                    cls_output = model.mlp(z_rare)
                    loss_cls_val = cls_loss(cls_output, val_batch_rare[1])
                    val_total_loss += loss_cls_val

                    ssim_value = calculate_ssim(recon_rare.view(-1, 28, 28), val_batch_rare[0].view(-1, 28, 28))
                    val_ssim += ssim_value.item()

                    val_acc = (cls_output.argmax(dim=1) == val_batch_rare[1]).sum().item() / len(val_batch_rare[1])
                    val_accuracy += val_acc

            for batch_idx, (val_data, _) in enumerate(rare_test_dataloader):
                recon_batch, _, _, _ = model(val_data)
                ssim_score = calculate_ssim(val_data, recon_batch)
                val_ssim_rare += ssim_score

            avg_val_ssim_rare = val_ssim_rare / len(rare_test_dataloader)

            avg_val_loss = val_loss / len(rare_test_dataloader)
            avg_val_mse = val_mse / len(rare_test_dataloader)
            avg_val_kld = val_kld / len(rare_test_dataloader)
            avg_val_accuracy = val_accuracy / len(rare_test_dataloader)
            avg_val_contra_loss = val_contrastive_loss / len(rare_test_dataloader)
            avg_val_ssim = val_ssim / len(rare_test_dataloader)

            wandb.log({
                "val_avg_loss": avg_val_loss,
                "val_avg_mse": avg_val_mse,
                "val_avg_kld": avg_val_kld,
                "val_avg_accuracy": avg_val_accuracy,
                "val_avg_contrastive_loss": avg_val_contra_loss,
                "val_avg_ssim_all_classes": avg_val_ssim,
                "val_avg_ssim_class_five": avg_val_ssim_rare
            })

    wandb.finish()
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

    wandb.init(project="unsupervised_generation_using_contrastive_learning_2",
               name=f"{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}",
               entity="")

    data_dir = "./mnist_data"
    (original_train_dataset, original_test_dataset,
     rare_train_dataset, rare_test_dataset,
     common_train_dataset, common_test_dataset) = create_datasets(data_dir)

    if args.model_type == "fine-tuning_vanilla_vae":
        fine_tuned_vae = fine_tuning_vanilla_vae(rare_train_dataset, rare_test_dataset, original_test_dataset)
    else:
        fine_tuned_vae = fine_tuning_contra_vae(rare_train_dataset, rare_test_dataset, common_train_dataset, common_test_dataset, original_test_dataset, args.contra_lambda)
    gen_imgs(fine_tuned_vae, original_test_dataset, dir_name=f"fig_{args.dataset_type}_{args.model_type}_epoch{args.epoch}_bs{args.batch_size}_lr{args.lr}_contra-lambda{args.contra_lambda}")
