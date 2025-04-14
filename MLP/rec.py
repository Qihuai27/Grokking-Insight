import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
import einops
import random
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Config():
    lr: float = 1e-3
    weight_decay: float = 1.0
    p: int = 97
    d: int = 0
    k: int = 30  # Context size
    d_model: int = 128
    fn_name: str = 'add'
    frac_train: float = 0.3
    num_epochs: int = 30000
    save_models: bool = True
    save_every: int = 100
    stopping_thresh: int = -1
    seed: int = 0
    num_layers: int = 2
    batch_style: str = 'full'
    d_vocab: int = 98
    n_ctx: int = 3
    d_mlp: int = 4*128  # 4*d_model
    bol_pos: bool = True
    num_heads: int = 4
    act_type: str = 'ReLU'
    device: t.device = t.device("cuda")
    use_ln: bool = False

    @property
    def d_head(self):
        return self.d_model // max(self.num_heads, 1)


    @property
    def fns_dict(self):
        return {
            'add': lambda x, y: (x + y) % self.p,
            'subtract': lambda x, y: (x - y) % self.p,
            'xy': lambda x, y: (x**2 + x*y + y**2) % self.p,
        }

    @property
    def fn(self):
        return self.fns_dict[self.fn_name]

class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(t.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])

def calculate_embedding_diff(embed_matrix, a, b, c, d):
    if a >= b or c >= d:
        raise ValueError("Start index must be less than end index.")
    if b > embed_matrix.shape[1] or d > embed_matrix.shape[1]:
        raise ValueError("Index range exceeds embedding matrix dimensions.")
    
    diffs1 = t.norm(embed_matrix[:, a+1:b] - embed_matrix[:, a:b-1], dim=0)
    diff1_mean = diffs1.mean().item()

    diffs2 = t.norm(embed_matrix[:, c+1:d] - embed_matrix[:, c:d-1], dim=0)
    diff2_mean = diffs2.mean().item()

    diffs3 = t.norm(embed_matrix[:, a:b] - embed_matrix[:, c:d])
    diff3_mean = diffs3.mean().item()

    return (diff1_mean + diff2_mean) / 2, diff3_mean

class TwoLayerMLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed = Embed(d_vocab=config.d_vocab, d_model=config.d_model)
        in_dim = config.n_ctx * config.d_model  
        self.hidden1 = nn.Linear(in_dim, config.d_mlp)
        self.hidden2 = nn.Linear(config.d_mlp, config.d_mlp)
        self.out = nn.Linear(config.d_mlp, config.p)
        self.act = nn.ReLU() if config.act_type == 'ReLU' else nn.GELU()

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape(x.size(0), -1)  
        x = self.hidden1(x)
        x = self.act(x)
        x = self.hidden2(x)
        x = self.act(x)
        x = self.out(x)  
        return x

def gen_train_test(config):
    k = config.k
    pairs = [(i, j + config.d, config.d_vocab - 1) for i in range(config.p) for j in range(config.p)]
    test_pairs = [(i, j + config.d, config.d_vocab - 1) for i in range(k + 1) for j in range(k + 1)]
    remaining_pairs = [pair for pair in pairs if pair not in test_pairs]
    random.seed(config.seed)
    random.shuffle(remaining_pairs)
    train_size = int(config.frac_train * len(pairs))
    train_pairs = remaining_pairs[:train_size]
    test_pairs.extend(remaining_pairs[train_size:])
    return train_pairs, test_pairs

def prepare_data(pairs):
    return t.tensor([(i, j, p) for i, j, p in pairs])

def full_loss(config: Config, model: nn.Module, data):
    inputs = prepare_data(data).to(config.device)
    logits = model(inputs)
    labels = t.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
    return F.cross_entropy(logits, labels)

def calculate_accuracy(config: Config, model: nn.Module, data):
    inputs = prepare_data(data).to(config.device)
    logits = model(inputs)
    labels = t.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
    predictions = logits.argmax(dim=-1)
    correct_predictions = (predictions == labels).sum().item()
    return correct_predictions / len(labels)

def remove_outliers(data, threshold, start_point=1000, window_size=500, alpha=0.1):
    data_series = pd.Series(data)
    data = data_series.to_numpy()
    smoothed_data = np.copy(data)
    for i in range(start_point, len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        y_window = np.concatenate((smoothed_data[start:i], smoothed_data[i+1:end]))
        smoothed_data[i] = np.nanmean(y_window)
    gaussdata = gaussian_filter1d(smoothed_data, sigma=10)
    smoothed_data = np.concatenate((data[0:start_point], gaussdata[start_point:]))
    return smoothed_data

class Trainer:
    def __init__(self, config: Config, model=None) -> None:
        self.model = model if model is not None else TwoLayerMLP(config)
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step / 10, 1))
        self.train, self.test = gen_train_test(config=config)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.embedding_diffs = []
        self.diffn = []
        self.config = config

    def do_a_training_step(self, epoch: int):
        train_loss = full_loss(config=self.config, model=self.model, data=self.train)
        test_loss = full_loss(config=self.config, model=self.model, data=self.test)
        self.train_losses.append(train_loss.item())
        self.test_losses.append(test_loss.item())
        train_accuracy = calculate_accuracy(config=self.config, model=self.model, data=self.train)
        test_accuracy = calculate_accuracy(config=self.config, model=self.model, data=self.test)
        self.train_accuracies.append(train_accuracy)
        self.test_accuracies.append(test_accuracy)
        
        embed_matrix = self.model.embed.W_E.data.cpu()
        max_min_diff, diffnt = calculate_embedding_diff(embed_matrix, 0, self.config.p - 1, self.config.d, self.config.d + self.config.p - 1)
        self.embedding_diffs.append(max_min_diff)
        self.diffn.append(diffnt)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, train loss {train_loss.item():.4f}, test loss {test_loss.item():.4f}, "
                  f"train accuracy {train_accuracy:.4f}, test accuracy {test_accuracy:.4f}, "
                  f"embedding diff {max_min_diff:.4f}")

        train_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return train_loss, test_loss

    def post_training_save(self):
        save_dict = {
            'model': self.model.state_dict(),
            'train_loss': self.train_losses[-1],
            'test_loss': self.test_losses[-1],
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
            'embedding_diffs': self.embedding_diffs,
            'diffn': self.diffn,
            'epoch': self.config.num_epochs,
        }
        t.save(save_dict, "final.pth")
        print("Saved model to final.pth")

    def plot_metrics(self):
        size1=85
        size2=90
        width1=16
        width2=14
        epochs = range(len(self.train_losses))
        plt.figure(figsize=(100, 21))

        smoothed_train_losses = remove_outliers(self.train_losses,0.05)
        smoothed_test_losses = remove_outliers(self.test_losses,0.05)

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, color='lightblue', linewidth=width2)
        plt.plot(epochs, self.test_losses, color='lightpink', linewidth=width2)
        plt.plot(epochs, smoothed_train_losses, color='blue', linewidth=width1, label='Training Loss')
        plt.plot(epochs, smoothed_test_losses, color='red', linewidth=width1, label='Testing Loss')
        plt.xlabel('Epochs', fontsize=size2)
        plt.ylabel('Loss', fontsize=size2)
        plt.legend(fontsize=size2)
        plt.title('Loss over Epochs', fontsize=size2)

        smoothed_train_accuracies = remove_outliers(self.train_accuracies,0.03)
        smoothed_test_accuracies = remove_outliers(self.test_accuracies,0.03)

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, color='lightblue', linewidth=width2)
        plt.plot(epochs, self.test_accuracies, color='lightpink', linewidth=width2)
        plt.plot(epochs, smoothed_train_accuracies, color='blue', linewidth=width1, label='Training Accuracy')
        plt.plot(epochs, smoothed_test_accuracies, color='red', linewidth=width1, label='Testing Accuracy')
        plt.xlabel('Epochs', fontsize=size2)
        plt.ylabel('Accuracy', fontsize=size2)
        plt.legend(fontsize=size2)
        plt.title('Accuracy over Epochs', fontsize=size2)

        embed_epochs = range(len(self.embedding_diffs))
        smoothed_embedding_diffs = remove_outliers(self.embedding_diffs, 20)

        plt.subplot(1, 3, 3)
        plt.plot(embed_epochs, self.embedding_diffs, color='lightblue', linewidth=width2)
        plt.plot(embed_epochs, smoothed_embedding_diffs, color='blue', linewidth=width1, label='Embedding Diff')
        plt.xlabel('Epochs', fontsize=size2)
        plt.ylabel('MED', fontsize=size2)
        plt.legend(fontsize=size2)
        plt.title('MED over Epochs', fontsize=size2)

        strname = f"metrics_k{self.config.k}_p{self.config.p}_d_model{self.config.d_model}"
        figname = f'plotfigure/mlp/{strname}.png'
        plt.savefig(figname, bbox_inches='tight', pad_inches=0.0)

        data_to_save = pd.DataFrame({
            'Epochs': list(epochs),
            'Train Loss': self.train_losses,
            'Test Loss': self.test_losses,
            'Train Accuracy': self.train_accuracies,
            'Smoothed Train Accuracy': smoothed_train_accuracies,
            'Test Accuracy': self.test_accuracies,
            'Smoothed Test Accuracy': smoothed_test_accuracies,
            'Embedding Epochs': list(embed_epochs),
            'Embedding Diff': self.embedding_diffs,
            'Smoothed Embedding Diff': smoothed_embedding_diffs
        })
        csvname = f'exresult/mlp/{strname}_data.csv'
        data_to_save.to_csv(csvname, index=False)


def train_model(config: Config):
    world = Trainer(config=config)
    print(f"Run name {int(time.time())}")

    for epoch in range(config.num_epochs):
        train_loss, test_loss = world.do_a_training_step(epoch)
        if test_loss.item() < config.stopping_thresh:
            break

    world.post_training_save()
    world.plot_metrics()
    return world


if __name__ == "__main__":
    k_list = [70]

    for k_val in k_list:
        print(f"\n\n========== Starting training with k = {k_val} ==========")
        config = Config(k=k_val)
        _ = train_model(config)
