import os
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader


def save_scatter_plot(data, labels, title, filename):
    width, height = 800, 600
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # Define colors for each class
    colors = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44),
        (214, 39, 40), (148, 103, 189), (140, 86, 75),
        (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207)
    ]

    x_min, x_max = min(data[:, 0]), max(data[:, 0])
    y_min, y_max = min(data[:, 1]), max(data[:, 1])

    def scale(value, min_val, max_val, scale_min, scale_max):
        return int((value - min_val) / (max_val - min_val) * (scale_max - scale_min) + scale_min)

    for point, label in zip(data, labels):
        x = scale(point[0], x_min, x_max, 0, width)
        y = scale(point[1], y_min, y_max, height, 0)  # y ì¢Œí‘œëŠ” ìƒ í•˜ ë°˜ì „
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=colors[label])

    # Add legend
    legend_width = 200
    legend_height = 20 * len(colors)
    legend_image = Image.new("RGB", (legend_width, legend_height), "white")
    legend_draw = ImageDraw.Draw(legend_image)

    font = ImageFont.load_default()
    for i, color in enumerate(colors):
        legend_draw.rectangle([(10, 10 + i * 20), (30, 30 + i * 20)], fill=color)
        legend_draw.text((40, 10 + i * 20), f'Class {i}', fill="black", font=font)

    image.paste(legend_image, (width - legend_width, 0))

    if not os.path.exists('visualization_output'):
        os.makedirs('visualization_output')
    image.save(f"visualization_output/{filename}.png")


def visualize_latent_space(model, dataset, title, filename):
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    with torch.no_grad():
        latents = []
        labels = []
        for data, label in dataloader:
            mu, _ = model.encode(data.view(-1, 784))
            latents.append(mu)
            labels.append(label)
        latents = torch.cat(latents).numpy()
        labels = torch.cat(labels).numpy()

    tsne = TSNE(n_components=2)
    tsne_result = tsne.fit_transform(latents)
    save_scatter_plot(tsne_result, labels, f't-SNE: {title}', f'{filename}_tsne')

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(latents)
    save_scatter_plot(pca_result, labels, f'PCA: {title}', f'{filename}_pca')


# if __name__ == "__main__":
#     data_dir = "./mnist_data"
#     (original_train_dataset, original_test_dataset,
#      imbalanced_train_dataset, imbalanced_test_dataset,
#      removed_train_dataset, removed_test_dataset) = create_datasets(data_dir)
#
#     original_vae = VAE()
#     original_vae.load_state_dict(torch.load("original_vae.pth"))
#     visualize_latent_space(original_vae, original_test_dataset, "Original Dataset", "original_test")
#
#     imbalanced_vae = VAE()
#     imbalanced_vae.load_state_dict(torch.load("imbalanced_vae.pth"))
#     visualize_latent_space(imbalanced_vae, original_test_dataset, "Imbalanced Dataset", "imbalanced_test")
#
#     removed_vae = VAE()
#     removed_vae.load_state_dict(torch.load("removed_vae.pth"))
#     visualize_latent_space(removed_vae, original_test_dataset, "Class Removed Dataset", "removed_test")
