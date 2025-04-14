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
from scipy.linalg import toeplitz
import os
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Config():
    lr: float = 1e-3
    weight_decay: float = 1.0
    p: int = 97
    d: int = 0
    d_model: int = 128
    fn_name: str = 'add'
    frac_train: float = 0.5
    num_epochs: int = 30000
    save_models: bool = True
    save_every: int = 100
    stopping_thresh: int = -1
    seed: int = 11
    num_layers: int = 2
    batch_style: str = 'full'
    d_vocab: int = 98
    n_ctx: int = 3
    d_mlp: int = 4 * d_model
    bol_pos: bool = True
    num_heads: int = 4
    act_type: str = 'ReLU'
    device: t.device = t.device("cuda")
    use_ln: bool = False

    @property
    def d_head(self):
        return self.d_model // self.num_heads

    @property
    def random_answers(self):
        return np.random.randint(low=0, high=self.p, size=(self.p, self.p))

    @property
    def fns_dict(self):
        return {
            'add': lambda x, y: (x + y) % self.p,
            'subtract': lambda x, y: (x - y) % self.p,
            'xy': lambda x, y: (x**2 + x*y + y**2) % self.p
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


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model, p):
        super().__init__()
        self.W_U = nn.Parameter(t.randn(d_model, p) / np.sqrt(d_model))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(t.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]


class LayerNorm(nn.Module):
    def __init__(self, d_model, epsilon=1e-4, model=[None]):
        super().__init__()
        self.model = model
        self.w_ln = nn.Parameter(t.ones(d_model))
        self.b_ln = nn.Parameter(t.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        if self.model[0].use_ln:
            x = x - x.mean(axis=-1)[..., None]
            x = x / (x.std(axis=-1)[..., None] + self.epsilon)
            x = x * self.w_ln
            x = x + self.b_ln
            return x
        else:
            return x


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx, model):
        super().__init__()
        self.model = model
        self.W_K = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(t.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(t.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', t.tril(t.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = t.einsum('ihd,bpd->biph', self.W_K, x)
        q = t.einsum('ihd,bpd->biph', self.W_Q, x)
        v = t.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = t.einsum('biph,biqh->biqp', k, q)
        attn_scores_masked = t.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = t.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = t.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type, model):
        super().__init__()
        self.model = model
        self.W_in = nn.Parameter(t.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(t.zeros(d_mlp))
        self.W_out = nn.Parameter(t.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(t.zeros(d_model))
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = t.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        if self.act_type == 'ReLU':
            x = F.relu(x)
        elif self.act_type == 'GeLU':
            x = F.gelu(x)
        x = t.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type, model):
        super().__init__()
        self.model = model
        self.attn = Attention(d_model, num_heads, d_head, n_ctx, model=self.model)
        self.mlp = MLP(d_model, d_mlp, act_type, model=self.model)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config, use_cache=False, use_ln=True):
        super().__init__()
        self.embed = Embed(d_vocab=config.d_vocab, d_model=config.d_model)
        self.pos_embed = PosEmbed(max_ctx=config.n_ctx, d_model=config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model=config.d_model,
                                                      d_mlp=config.d_mlp,
                                                      d_head=config.d_head,
                                                      num_heads=config.num_heads,
                                                      n_ctx=config.n_ctx,
                                                      act_type=config.act_type,
                                                      model=[self]) for _ in range(config.num_layers)])
        self.unembed = Unembed(d_vocab=config.d_vocab, d_model=config.d_model, p=config.p)
        self.use_ln = use_ln

    def forward(self, x):
        x = self.embed(x)
        if config.bol_pos:
            x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = x[:, 2, :]
        x = self.unembed(x)
        return x


def gen_train_test(config: Config):
    pairs = [(i, j + config.d, config.d_vocab - 1) for i in range(config.p) for j in range(config.p)]
    random.seed(config.seed)
    random.shuffle(pairs)
    div = int(config.frac_train * len(pairs))
    return pairs[:div], pairs[div:]


def prepare_data(pairs):
    inputs = t.tensor([(i, j, p) for i, j, p in pairs])
    return inputs


def full_loss(config: Config, model: Transformer, data):
    inputs = prepare_data(data)
    logits = model(inputs)
    labels = t.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
    return F.cross_entropy(logits, labels)


def mse_loss(config: Config, model: Transformer, data):
    inputs = prepare_data(data)
    logits = model(inputs)
    labels = [config.fn(i, j) for i, j, _ in data]
    one_hot_labels = F.one_hot(t.tensor(labels), num_classes=config.p).float().to(config.device)
    logits = logits.to(t.float32)
    return F.mse_loss(logits, one_hot_labels)


def calculate_accuracy(config: Config, model: Transformer, data):
    inputs = prepare_data(data)
    logits = model(inputs)
    labels = t.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
    predictions = logits.argmax(dim=-1)
    correct_predictions = (predictions == labels).sum().item()
    accuracy = correct_predictions / len(labels)
    return accuracy


def calculate_embedding_diff(embed_matrix, a, b, c, d):
    if a >= b or c >= d:
        raise ValueError("Start index must be less than end index.")
    if b > embed_matrix.shape[1] or d > embed_matrix.shape[1]:
        raise ValueError("Index range exceeds embedding matrix dimension.")

    diffs1 = t.norm(embed_matrix[:, a + 1:b] - embed_matrix[:, a:b - 1], dim=0)
    diff1_mean = diffs1.mean().item()

    diffs2 = t.norm(embed_matrix[:, c + 1:d] - embed_matrix[:, c:d - 1], dim=0)
    diff2_mean = diffs2.mean().item()

    return 0.5 * (diff1_mean + diff2_mean)


def apply_gaussian_filter(data, sigma=2):
    return gaussian_filter1d(data, sigma=sigma)


def remove_outliers(data, threshold, start_point=1000, window_size=500, alpha=0.1):
    data_series = pd.Series(data)
    data = data_series.to_numpy()
    smoothed_data = np.copy(data)
    for i in range(start_point, len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        y_window = np.concatenate((smoothed_data[start:i], smoothed_data[i + 1:end]))
        smoothed_data[i] = np.nanmean(y_window)
    gaussdata = gaussian_filter1d(smoothed_data, sigma=10)
    smoothed_data = np.concatenate((data[0:start_point], gaussdata[start_point:]))
    return smoothed_data


class Trainer:
    def __init__(self, config: Config, model=None) -> None:
        self.model = model if model is not None else Transformer(config, use_cache=False)
        self.model.to(config.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: min(step / 10, 1))
        self.train, self.test = gen_train_test(config=config)
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.embedding_diffs = []
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
        max_min_diff = calculate_embedding_diff(embed_matrix, 0, config.p - 1, config.d, config.d + config.p - 1)
        self.embedding_diffs.append(max_min_diff)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, train loss {train_loss.item():.4f}, test loss {test_loss.item():.4f}, train accuracy {train_accuracy:.4f}, test accuracy {test_accuracy:.4f}, embedding diff {max_min_diff:.4f}')
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
            'epoch': self.config.num_epochs,
        }
        t.save(save_dict, f"final.pth")
        print(f"Saved model to {'final.pth'}")

    def plot_metrics(self):
        size1 = 85
        size2 = 90
        width1 = 16
        width2 = 14
        epochs = range(len(self.train_losses))
        plt.figure(figsize=(100, 21))

        smoothed_train_losses = remove_outliers(self.train_losses, 0.05)
        smoothed_test_losses = remove_outliers(self.test_losses, 0.05)

        plt.subplot(1, 3, 1)
        plt.plot(epochs, self.train_losses, color='lightblue', linewidth=width2)
        plt.plot(epochs, self.test_losses, color='lightpink', linewidth=width2)
        plt.plot(epochs, smoothed_train_losses, color='blue', linewidth=width1, label='Training Loss')
        plt.plot(epochs, smoothed_test_losses, color='red', linewidth=width1, label='Testing Loss')
        plt.xlabel('Epochs', fontsize=size2)
        plt.ylabel('Loss', fontsize=size2)
        plt.legend(fontsize=size2)
        plt.title('Loss over Epochs', fontsize=size2)
        plt.tick_params(axis='both', which='major', labelsize=size1, pad=18)
        ax = plt.gca()
        ax.spines['top'].set_linewidth(width2)
        ax.spines['right'].set_linewidth(width2)
        ax.spines['bottom'].set_linewidth(width2)
        ax.spines['left'].set_linewidth(width2)

        smoothed_train_accuracies = remove_outliers(self.train_accuracies, 0.03)
        smoothed_test_accuracies = remove_outliers(self.test_accuracies, 0.03)

        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.train_accuracies, color='lightblue', linewidth=width2)
        plt.plot(epochs, self.test_accuracies, color='lightpink', linewidth=width2)
        plt.plot(epochs, smoothed_train_accuracies, color='blue', linewidth=width1, label='Training Accuracy')
        plt.plot(epochs, smoothed_test_accuracies, color='red', linewidth=width1, label='Testing Accuracy')
        plt.xlabel('Epochs', fontsize=size2)
        plt.ylabel('Accuracy', fontsize=size2)
        plt.legend(fontsize=size2)
        plt.title('Accuracy over Epochs', fontsize=size2)
        plt.tick_params(axis='both', which='major', labelsize=size1, pad=18)
        ax = plt.gca()
        ax.spines['top'].set_linewidth(width2)
        ax.spines['right'].set_linewidth(width2)
        ax.spines['bottom'].set_linewidth(width2)
        ax.spines['left'].set_linewidth(width2)

        embed_epochs = range(len(self.embedding_diffs))
        smoothed_embedding_diffs = remove_outliers(self.embedding_diffs, 20)

        plt.subplot(1, 3, 3)
        plt.plot(embed_epochs, self.embedding_diffs, color='lightblue', linewidth=width2)
        plt.plot(embed_epochs, smoothed_embedding_diffs, color='blue', linewidth=width1, label='Embedding Diff')
        plt.xlabel('Epochs', fontsize=size2)
        plt.ylabel('MED', fontsize=size2)
        plt.legend(fontsize=size2)
        plt.title('MED over Epochs', fontsize=size2)
        plt.tick_params(axis='both', which='major', labelsize=size1, pad=18)
        ax = plt.gca()
        ax.spines['top'].set_linewidth(width2)
        ax.spines['right'].set_linewidth(width2)
        ax.spines['bottom'].set_linewidth(width2)
        ax.spines['left'].set_linewidth(width2)

        strname = config.fn_name + 'metrics_' + str(config.p) + '_seed:' + str(config.seed) + str(config.d) + '_' + str(config.d_model) + '_' + str(config.frac_train)
        if config.bol_pos: 
            figname = 'plotfigure/trans/' + strname + '_pos.png'
        else:
            figname = 'plotfigure/trans/' + strname + '.png'
        plt.savefig(figname, bbox_inches='tight', pad_inches=0.0)

        data_to_save = pd.DataFrame({
            'Epochs': list(epochs),
            'Train Loss': self.train_losses,
            'Smoothed Train Loss': smoothed_train_losses,
            'Test Loss': self.test_losses,
            'Smoothed Test Loss': smoothed_test_losses,
            'Train Accuracy': self.train_accuracies,
            'Smoothed Train Accuracy': smoothed_train_accuracies,
            'Test Accuracy': self.test_accuracies,
            'Smoothed Test Accuracy': smoothed_test_accuracies,
            'Embedding Epochs': list(embed_epochs),
            'Embedding Diff': self.embedding_diffs,
            'Smoothed Embedding Diff': smoothed_embedding_diffs
        })

        if config.bol_pos:
            csvname = 'exresult/trans/' + strname + '_pos_data.csv'
        else:
            csvname = 'exresult/trans/' + strname + '_data.csv'
        data_to_save.to_csv(csvname, index=False)


def train_model(config: Config):
    world = Trainer(config=config)
    print(f'Run name {int(time.time())}')

    for epoch in range(config.num_epochs):
        train_loss, test_loss = world.do_a_training_step(epoch)
        if test_loss.item() < config.stopping_thresh:
            break

    world.post_training_save()
    world.plot_metrics()
    return world


if __name__ == "__main__":
    p_list = [97]

    for p_val in p_list:
        print(f"\n\n========== Starting training p = {p_val} ==========")
        config = Config(p=p_val)
        _ = train_model(config)  
