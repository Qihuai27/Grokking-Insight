# Grokking-Insight
Code for the paper **Beyond Progress Measures: Theoretical Insights into the Mechanism of Grokking**.

The code is divided into two parts, one is the code for arithmetic research on small transformer or DNN, and the other is the code for the grokking phenomenon on resnet-18.

### Quick Start
```
cd transformer/MLP
python base.py /str.py / rec.py
```
or
```
cd resnet
python dic_gen.py
python data_gen.py
python resnet_test.py
```

### Grokking on Transformer or DNN

There are three files in the Transformer or DNN folder, base, rec, and str. Each file is configured with MED plots, which can be tested by modifying p and fn_name in the config parameters. rec and str correspond to the two data set setting methods in the first experiment of the paper, and the corresponding effects can be achieved by adjusting the parameter k.

If you want to use MSE loss, you can replace the full_loss function with the following code：
```
def full_loss(config: Config, model: nn.Module, data):
    inputs = prepare_data(data)
    logits = model(inputs)  # shape: [batch, p]
    labels = [config.fn(i, j) for i, j, _ in data]
    one_hot_labels = F.one_hot(t.tensor(labels), num_classes=config.p).float().to(config.device)
    logits = logits.to(t.float32)
    return F.mse_loss(logits, one_hot_labels)
```

If you want to test fixed initialization, such as fixed one-hot initialization, you need to replace the embed class with
```
class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        W = t.zeros(d_model, d_vocab)
        eye_size = min(d_vocab, d_model)
        W[:eye_size, :eye_size] = t.eye(eye_size)
        self.register_buffer("W_E", W) 

    def forward(self, x):
        return t.einsum('dbp -> bpd', self.W_E[:, x])
```
At this point, MED is constant, so we need to input the first hidden layer to calculate MED:
```
def calculate_embedding_diff(model: nn.Module, a: int, b: int, c: int, d: int):
    device = model.embed.W_E.device
    tokens1 = t.tensor([[i, model.config.d + i, model.config.d_vocab - 1] for i in range(a, b)]).to(device)
    tokens2 = t.tensor([[i, model.config.d + i, model.config.d_vocab - 1] for i in range(c, d)]).to(device)

    emb1 = model.embed(tokens1).reshape(tokens1.size(0), -1)  # [b-a, 3*d_model]
    emb2 = model.embed(tokens2).reshape(tokens2.size(0), -1)  # [d-c, 3*d_model]

    hidden1_out_1 = model.act(model.hidden1(emb1))  # [b-a, d_mlp]
    hidden1_out_2 = model.act(model.hidden1(emb2))  # [d-c, d_mlp]

    diffs1 = t.norm(hidden1_out_1[1:] - hidden1_out_1[:-1], dim=1)
    diff1_mean = diffs1.mean().item()

    diffs2 = t.norm(hidden1_out_2[1:] - hidden1_out_2[:-1], dim=1)
    diff2_mean = diffs2.mean().item()

    min_len = min(hidden1_out_1.shape[0], hidden1_out_2.shape[0])
    diffs3 = t.norm(hidden1_out_1[:min_len] - hidden1_out_2[:min_len], dim=1)
    diff3_mean = diffs3.mean().item()

    return (diff1_mean + diff2_mean) / 2, diff3_mean
```

### Grokking on Resnet-18

There are three files in the resnet folder, one is dictionary generation, one is data set generation, and one is resnet-18 test. You can observe the grokking phenomenon on resnet-18 by running them in sequence.
