import os
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg import Vgg16
import glob


def train(style_image, dataset="coco", batch_size=4, image_size=128,
epochs=2, seed=42, content_weight=1e5, style_weight=1e10, lr=1e-3, log_interval=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_dataset = datasets.ImageFolder(dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    transformer = TransformerNet().to(device)

    style_name = style_image.split("/")[-1].split(".")[0]
    genre = style_image.split("/")[-2]
    model_path = f"models/{genre}/{style_name}.model"

    if os.path.exists(model_path):
        transformer.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    style = utils.load_image(style_image)
    style = style_transform(style)
    style = style.repeat(batch_size, 1, 1, 1).to(device)

    features_style = vgg(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    for e in range(epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()

            x = x.to(device)
            y = transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y)
            features_x = vgg(x)

            content_loss = content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

    # save model
    transformer.eval().cpu()

    style_name = style_image.split("/")[-1].split(".")[0]
    genre = style_image.split("/")[-2]
    save_model_path = f"models/{genre}/{style_name}.model"
    torch.save(transformer.state_dict(), save_model_path)

    # Clean up cuda memory
    del(transformer)
    del(vgg)
    del(x)
    del(y)
    torch.cuda.empty_cache()

    print(f"\nDone, trained model {genre} saved at", save_model_path)


def train_all_styles(retrain=False, epochs=2, style_weight=1e10):
    """
    Perform training for all styles in styles folder
    Retrain: If True, model.state_dict will be loaded and retrained if the model already exists.
    If False, style will be skipped.
    """
    style_list = glob.glob("styles/*/*")

    for style_path in sorted(style_list):
        style_name = style_path.split("/")[-1].split(".")[0]
        genre = style_path.split("/")[-2]
        model_path = f"models/{genre}/{style_name}.model"

        if os.path.exists(model_path):
            if retrain:
                print(f"\n[INFO] Model weights will be loaded...")
                print(f"[INFO] Now training {genre} -> {style_name}")
                try:
                    train(style_path, batch_size=7, epochs=epochs, style_weight=style_weight)
                except Exception as e:
                    print(e)
                    print("[INFO] Not enough memory. Next time try to decrease batch_size")
                    print(f"[INFO] {style_name} is skipped, continue...")
                    torch.cuda.empty_cache()
                    continue

            else:
                print(f"[INFO] Model {style_name} exists, skip...")
                continue
        else:
            print(f"\n[INFO] Model {style_name} doesn't exist.\n[INFO] Train new model...")
            print(f"[INFO] Now training {genre} -> {style_name}")
            try:
                train(style_path, batch_size=7, epochs=epochs, style_weight=style_weight)
            except Exception as e:
                print(e)
                print("[INFO] Not enough memory. Next time try to decrease batch_size")
                print(f"[INFO] {style_name} is skipped, continue...")
                torch.cuda.empty_cache()
                continue

if __name__ == "__main__":
    train_all_styles(retrain=False, epochs=2, style_weight=1e8)