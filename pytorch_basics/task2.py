# -*- coding: utf-8 -*-
"""
## GloVe下载并保存至云盘（只需要运行一次）

import urllib.request
import zipfile
import os

if not os.path.exists('glove.6B.300d.txt'):
    print("正在下载 GloVe 数据集，压缩包约822MB，请耐心等待...")
    urllib.request.urlretrieve('http://nlp.stanford.edu/data/glove.6B.zip', 'glove.zip')

    print("下载完成，正在提取 300d 向量文件...")
    with zipfile.ZipFile('glove.zip','r') as z:
        z.extract('glove.6B.300d.txt')

    os.remove('glove.zip')
    print("提取完成，已清理原始压缩包！")
else:
    print("glove.6B.300d.txt 已存在，跳过下载阶段。")

import shutil
save_dir = '/content/drive/MyDrive/pytorch_beginner/task2/glove'
os.makedirs(save_dir, exist_ok=True)
target_file = os.path.join(save_dir, 'glove.6B.300d.txt')
shutil.move("/content/glove.6B.300d.txt",target_file)

## 导入必要的包
"""

import os
import pandas as pd
import numpy as np
import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from collections import Counter
import matplotlib.pyplot as plt
from IPython import display
from matplotlib_inline import backend_inline
from sklearn.model_selection import train_test_split

"""## Tokenization & word embedding"""

train_data  = pd.read_csv("/content/drive/MyDrive/pytorch_beginner/task2/new_train.tsv", sep = "\t", header = None)
test_data  = pd.read_csv("/content/drive/MyDrive/pytorch_beginner/task2/new_test.tsv", sep = "\t", header = None)
train_data.columns = ["text", "label"]
test_data.columns = ["text", "label"]
X_train, X_dev, Y_train, Y_dev = train_test_split(train_data[['text']], train_data[['label']], test_size = 0.2, random_state = 42)

X_train['tokens'] = X_train['text'].apply(lambda x: nltk.word_tokenize(str(x).lower()))
words = [word for line in X_train['tokens'] for word in line]
word_counts = Counter(words)
vocab = {"<pad>":0, "<unk>":1}
for word, _ in word_counts.items():
  vocab[word] = len(vocab)

MAX_LEN = 35 #训练集句子长度的95%分位数
def encode_and_pad(tokens):
  indices = [vocab.get(x,vocab["<unk>"]) for x in tokens]
  if len(indices) >= MAX_LEN:
    indices = indices[: MAX_LEN]
  else:
    indices = indices + [vocab["<pad>"]] * (MAX_LEN - len(indices))
  return indices

X_train['indices'] = X_train['tokens'].apply(encode_and_pad)
X_dev['tokens'] = X_dev['text'].apply(lambda x: nltk.word_tokenize(str(x).lower()))
X_dev['indices'] = X_dev['tokens'].apply(encode_and_pad)

X_train_vec = torch.tensor(X_train['indices'].to_list(), dtype = torch.long)
X_dev_vec = torch.tensor(X_dev['indices'].to_list(), dtype = torch.long)
Y_train_vec = torch.tensor(Y_train['label'].to_list(), dtype = torch.int16)
Y_dev_vec = torch.tensor(Y_dev['label'].to_list(), dtype = torch.int16)

test_data['tokens'] = test_data['text'].apply(lambda x: nltk.word_tokenize(str(x).lower()))
test_data['indices'] = test_data['tokens'].apply(encode_and_pad)
X_test_vec = torch.tensor(test_data['indices'].to_list(), dtype=torch.long)
Y_test_vec = torch.tensor(test_data['label'].to_list(), dtype=torch.int16)

VOCAB_SIZE = len(vocab)
EMBED_DIM = 300
glove_vectors = {}
with open("/content/drive/MyDrive/pytorch_beginner/task2/glove/glove.6B.300d.txt", "r", encoding = "utf8") as f:
  for line in f:
    parts = line.split()
    glove_vectors[parts[0]] = torch.tensor([float(x) for x in parts[1:]], dtype = torch.float32)

embedding_matrix = torch.zeros(VOCAB_SIZE, EMBED_DIM)
hit_count = 0
hit_indices = []
unhitted_words = []
for word, idx in vocab.items():
  if word in glove_vectors:
    embedding_matrix[idx] = glove_vectors.get(word)
    hit_count += 1
    hit_indices.append(idx)
if hit_indices:
    hit_vectors = embedding_matrix[hit_indices]
    glove_variance = torch.var(hit_vectors).item()
    # 均匀分布方差 V = a^2 / 3，反推 a = sqrt(3 * V)
    a = math.sqrt(3 * glove_variance)
else:
    # 防御性设定（防止没有任何词命中的极端情况）
    a = 0.1
for word, idx in vocab.items():
    if word not in glove_vectors:
        unhitted_words.append(word)
        if word == '<pad>':
            # Padding 必须保持全 0，不参与梯度更新的影响
            embedding_matrix[idx] = torch.zeros(EMBED_DIM)
        else:
            # OOV 词（包括 <unk>）使用对应的均匀分布 U[-a, a] 初始化
            embedding_matrix[idx] = torch.empty(EMBED_DIM).uniform_(-a, a)

print(f"GloVe 命中率: {hit_count/VOCAB_SIZE * 100:.2f}%")
print(f"GloVe 未命中率词: {unhitted_words[:20]}")
print(f"GloVe 预训练方差: {glove_variance:.6f}")
print(f"OOV 词初始化均匀分布边界 a: ±{a:.6f}")

unhitted_words

class CustomDataset(Dataset):
  def __init__(self, data, label):
    self.data = data
    self.label = label

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    return self.data[idx], self.label[idx].long()

"""## CNN"""

#不同尺寸的卷积核类似N-gram，但是无法建模长距离
class CNN(nn.Module):
  def __init__(self, embedding_layer, embed_dim, num_classes, kernels, num_filters):
    super(CNN, self).__init__()
    self.embedding = embedding_layer
    self.branches = nn.ModuleList([
        nn.Sequential(
        nn.Conv1d(in_channels = embed_dim, out_channels = num_filters ,kernel_size = k),
        nn.ReLU(),
        nn.AdaptiveMaxPool1d(1)
    ) for k in kernels
    ])
    linear_input_dim = num_filters*len(kernels)
    self.FC = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(linear_input_dim, num_classes)
    )

  def forward(self, x):
    x = self.embedding(x)
    x = x.permute(0,2,1)
    output = [branch(x).squeeze(-1) for branch in self.branches]
    outputs = torch.cat(output, dim=1)
    return self.FC(outputs)

def init_cnn(module):
  if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
    nn.init.xavier_normal_(module.weight)
    if module.bias is not None:
      nn.init.constant_(module.bias,0.0)

"""## BiLSTM+Attention，解决长距离语境"""

class BiLSTMAttention(nn.Module):
  def __init__(self, embedding_layer, embed_dim, hidden_size, num_classes):
    super().__init__()
    self.embedding = embedding_layer
    self.lstm = nn.LSTM(input_size = embed_dim, hidden_size = hidden_size, batch_first = True, bidirectional = True)
    self.attention_k = nn.Linear(2*hidden_size,2*hidden_size)
    self.attention_q = nn.Parameter(torch.Tensor(2*hidden_size,1))
    self.linear = nn.Linear(2*hidden_size, num_classes)
    nn.init.uniform_(self.attention_q,-0.1,0.1)

  def _init_weights(self):
    for name,param in self.lstm.named_parameters():
      if "weight_ih" in name:
        nn.init.xavier_uniform_(param.data)
      elif "weight_hh" in name:
        nn.init.orthogonal_(param.data)
      elif "bias" in name:
        nn.init.constant_(param.data,0)
        n = param.size(0)
        start, end = n // 4, n // 2
        param.data[start:end].fill_(1.0)

      nn.init.xavier_uniform_(self.attention_k.weight)
      nn.init.constant_(self.attention_k.bias, 0.0)

      nn.init.uniform_(self.attention_q, -0.1, 0.1)
      nn.init.xavier_uniform_(self.linear.weight)
      nn.init.constant_(self.linear.bias, 0.0)

  def forward(self,x):
    #x:(batch_size, seq_len)
    embedded = self.embedding(x)
    #embedded:(batch_size, seq_len, embed_dim)
    output,(h_n,c_n) = self.lstm(embedded)
    #output:(batch_size, seq_len, 2*hidden_size)
    key = torch.tanh(self.attention_k(output))
    #key:(batch_size, seq_len, 2*hidden_size)
    attention_score = torch.matmul(key,self.attention_q).squeeze(-1)
    #attention_score:(batch_size, seq_len)
    attention_score = F.softmax(attention_score,dim=1)
    v = torch.sum(output*attention_score.unsqueeze(-1),dim = 1)
    logits = self.linear(v)
    #logits:(batch_size, num_classes)
    return logits #, attention_score

"""## Transformer"""

#encoder学习序列特征，然后送入全连接层
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_len=100):
    super(PositionalEncoding, self).__init__()
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # 计算分母：10000^(2i/d_model)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    # x shape: (batch_size, seq_len, d_model)
    x = x + self.pe[:, :x.size(1), :]
    return x

class TransformerClassifier(nn.Module):
    def __init__(self, embedding_layer, num_classes, pad_idx, d_model=100, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.5):
        super(TransformerClassifier, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = embedding_layer
        self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # PyTorch 的 src_key_padding_mask 期望 True 的位置被忽略
        padding_mask = (x == self.pad_idx) # shape: (batch_size, seq_len)

        embedded = self.embedding(x) # (batch, seq_len, 100)
        embedded = self.pos_encoder(embedded)

        encoded_output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        # encoded_output: (batch_size, seq_len, 100)

        # 将 padding_mask 反转，False(实际pad)变为0，True(真实词)变为1
        mask = (~padding_mask).unsqueeze(-1).float() # (batch_size, seq_len, 1)
        masked_output = encoded_output * mask
        sum_embeddings = masked_output.sum(dim=1) # (batch_size, 100)
        # 计算每个句子的真实长度 (加上 1e-9 防止除以 0)
        valid_lens = mask.sum(dim=1).clamp(min=1e-9)

        pooled = sum_embeddings / valid_lens # (batch_size, 100)

        logits = self.fc(pooled)
        return logits

def init_weights_t(m):
  if isinstance(m, nn.Linear):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

class metrics_accumulator:
    def __init__(self, n):
        self.metrics = [0.0]*n

    def add(self,*args):
        self.metrics = [a + float(b) for a,b in zip(self.metrics, args)]

    def reset(self):
        self.metrics = [0.0] * len(self.metrics)

    def get_item(self, idx):
        return self.metrics[idx]

class Animator:
    def __init__(self, x_label='Epoch', legend=None, figsize=(6, 4)):
        if legend is None:
            legend = ['Train Loss', 'Train Acc', 'Dev Loss', 'Dev Acc']
        self.legend_labels = legend

        # 全局学术字体与分辨率设定
        plt.style.use('default')
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['figure.dpi'] = 300

        backend_inline.set_matplotlib_formats("svg")
        self.fig, self.ax1 = plt.subplots(figsize=figsize)
        self.ax2 = self.ax1.twinx()

        self.line_styles = [
            {'color': '#1f77b4', 'linestyle': '-',  'linewidth': 1.5}, # Train Loss
            {'color': '#d62728', 'linestyle': '-',  'linewidth': 1.5}, # Train Acc
            {'color': '#1f77b4', 'linestyle': '--', 'linewidth': 1.5}, # Dev Loss
            {'color': '#d62728', 'linestyle': '--', 'linewidth': 1.5}  # Dev Acc
        ]

        def config_axes():
            self.ax1.set_xlabel(x_label, fontsize=11)

            # 强制右侧坐标轴保持在右侧
            self.ax2.yaxis.tick_right()
            self.ax2.yaxis.set_label_position("right")

            self.ax1.set_ylabel('Cross Entropy Loss', fontsize=11, color='#1f77b4')
            self.ax2.set_ylabel('Accuracy', fontsize=11, color='#d62728')

            self.ax1.tick_params(axis='y', labelcolor='#1f77b4')
            self.ax2.tick_params(axis='y', labelcolor='#d62728')

            self.ax1.spines['top'].set_visible(False)
            self.ax2.spines['top'].set_visible(False)
            self.ax1.spines['left'].set_color('#1f77b4')
            self.ax2.spines['right'].set_color('#d62728')

            self.ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)

        self.config_axes = config_axes
        # 确保初始化了数据存储器 (解决 AttributeError 的核心)
        self.X, self.Y = None, None

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]

        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)

        self.ax1.cla()
        self.ax2.cla()

        lines = []
        for i, (x_data, y_data) in enumerate(zip(self.X, self.Y)):
            style = self.line_styles[i % len(self.line_styles)]
            if i % 2 == 0:
                line, = self.ax1.plot(x_data, y_data, **style, zorder=3)
            else:
                line, = self.ax2.plot(x_data, y_data, **style, zorder=3)
            lines.append(line)

        self.config_axes()

        if self.legend_labels:
            self.ax1.legend(lines, self.legend_labels, frameon=False,
                            fontsize=9, loc='center right', bbox_to_anchor=(0.95, 0.5))

        plt.tight_layout()
        display.display(self.fig)
        display.clear_output(wait=True)

    def save_fig(self, filename):
        self.fig.savefig(filename, format='pdf', bbox_inches='tight')

def calc_accuracy(y_hat, y):
  label_predict = torch.max(y_hat,axis=1).indices
  correct_predict = label_predict == y.type(label_predict.dtype)
  num_correct = correct_predict.sum().item()
  return num_correct

def build_model(config, vocab_size, pad_idx, random_init_a, embedding_matrix=None):
    if config['embed_type'] == 'random':
        embed_layer = nn.Embedding(vocab_size, config['embed_dim'], padding_idx=pad_idx)
        nn.init.uniform_(embed_layer.weight,-random_init_a, random_init_a)
        with torch.no_grad():
          embed_layer.weight[pad_idx] = 0
    elif config['embed_type'] == 'glove':
        embed_layer = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=config['embed_freeze'],
            padding_idx=pad_idx
        )

    if config['model_type'] == 'TextCNN':
        model =  CNN(embed_layer, embed_dim=config['embed_dim'], num_classes=config['num_classes'], kernels=config['kernels'], num_filters=config['num_filters'])
        model.apply(init_cnn)
        return model
    elif config['model_type'] == 'BiLSTM':
        model =  BiLSTMAttention(embed_layer, embed_dim=config['embed_dim'], hidden_size=config['hidden_size'], num_classes=config['num_classes'])
        model._init_weights()
        return model
    elif config['model_type'] == 'Transformer':
        model = TransformerClassifier(
          embedding_layer=embed_layer,
          num_classes=config['num_classes'],
          pad_idx=pad_idx,
          d_model=config['embed_dim'],
          nhead=4,
          num_layers=1,
          dim_feedforward=256,
          dropout=0.5
      ).to(device)
        model.apply(init_weights_t)
        return model
    else:
        raise ValueError("不支持的模型类型")

def build_optimizer(model, config):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

#用于最后一组实验
def build_optimizer(model, config):
  trainable_params = filter(lambda p: p.requires_grad, model.parameters())
  if config['model_type'] == 'Transformer':
    return torch.optim.AdamW(trainable_params, lr=config['lr'], weight_decay=0.01)
  elif config['model_type'] == 'TextCNN':
    return torch.optim.Adadelta(trainable_params, lr=1.0)
  else:
    return torch.optim.Adam(trainable_params, lr=config['lr'])

def train_and_evaluate(config, train_dataloader, dev_dataloader, device, vocab_size, pad_idx, random_init_a, embedding_matrix):
  model = build_model(config, vocab_size, pad_idx, random_init_a, embedding_matrix).to(device)
  optimizer = build_optimizer(model, config)
  criterion = nn.CrossEntropyLoss()
  metrics = metrics_accumulator(6)
  animator = Animator()
  best_dev_loss = float('inf')
  patience_counter = 0
  patience = 5
  for epoch in range(config['num_epochs']):
    metrics.reset()
    model.train()
    for X, y in train_dataloader:
      X = X.to(device)
      y = y.to(device)
      optimizer.zero_grad()
      y_hat = model(X)
      loss = criterion(y_hat, y)
      loss.backward()
      max_norm = 3
      if config['model_type'] == 'TextCNN':
        with torch.no_grad():
          for name,param in model.named_parameters():
            if 'FC' in name and 'weight' in name:
              norm = param.norm(2,dim=1,keepdim=True)
              desired = torch.clamp(norm,max = max_norm)
              param.mul_(desired/(norm+1e-8))
      else:
        # RNN 与 Transformer 标配梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      metrics.add(loss*y.shape[0], calc_accuracy(y_hat,y), y.shape[0],0,0,0)
    model.eval()
    with torch.no_grad():
      for X_dev, y_dev in dev_dataloader:
        X_dev = X_dev.to(device)
        y_dev = y_dev.to(device)
        y_dev_hat = model(X_dev)
        dev_loss = criterion(y_dev_hat,y_dev)
        metrics.add(0,0,0,dev_loss*y_dev.shape[0], calc_accuracy(y_dev_hat,y_dev), y_dev.shape[0])

    train_loss = metrics.get_item(0)/metrics.get_item(2)
    train_acc = metrics.get_item(1)/metrics.get_item(2)
    dev_loss = metrics.get_item(3)/metrics.get_item(5)
    dev_acc = metrics.get_item(4)/metrics.get_item(5)
    animator.add(epoch + 1, (train_loss, train_acc, dev_loss, dev_acc))
    if dev_loss < best_dev_loss:
      best_dev_loss = dev_loss
      best_state_dict = copy.deepcopy(model.state_dict())
      #torch.save(model.state_dict(),f"/content/drive/MyDrive/pytorch_beginner/task2/best{config['model_type']}.pth")
      patience_counter = 0
    else:
      patience_counter += 1
    if patience_counter >= patience:
      print(f"在第{epoch+1}轮触发早停")
      break
  pic_save_path = pic_name = f"/content/drive/MyDrive/pytorch_beginner/task2/{config['exp_name']}_curve.pdf"
  animator.save_fig(pic_save_path)
  return best_dev_loss, best_state_dict

class F1MacroAccumulator:
    def __init__(self, num_classes=5):
        self.num_classes = num_classes
        self.TP = torch.zeros(num_classes, dtype=torch.float32)
        self.FP = torch.zeros(num_classes, dtype=torch.float32)
        self.FN = torch.zeros(num_classes, dtype=torch.float32)

    def add_batch(self, y_true, y_pred):
        """
        y_true: shape (batch_size, num_classes) 或 (batch_size,)
        y_pred: shape (batch_size, num_classes)
        """
        if y_true.ndim > 1:
            y_true = torch.argmax(y_true, dim=1)
        if y_pred.ndim > 1:
            y_pred = torch.argmax(y_pred, dim=1)

        for i in range(self.num_classes):
            true_i = (y_true == i)
            pred_i = (y_pred == i)

            self.TP[i] += torch.sum(true_i & pred_i).item()
            self.FP[i] += torch.sum((~true_i) & pred_i).item()
            self.FN[i] += torch.sum(true_i & (~pred_i)).item()

    def get_macro_f1(self) -> float:
        epsilon = 1e-7
        precision = self.TP / (self.TP + self.FP + epsilon)
        recall = self.TP / (self.TP + self.FN + epsilon)

        f1_per_class = 2 * (precision * recall) / (precision + recall + epsilon)

        # 返回所有类别 F1 的算术平均值
        return torch.mean(f1_per_class).item()

def test(config, state_dict, test_dataloader, device, vocab_size, pad_idx, random_init_a, embedding_matrix):
  # 在测试集上进行最终评估
  f1MacroAccumulator = F1MacroAccumulator(num_classes = config['num_classes'])
  model = build_model(config, vocab_size, pad_idx, random_init_a, embedding_matrix).to(device)
  criterion = nn.CrossEntropyLoss()
  model.load_state_dict(state_dict)
  model.eval()

  test_loss_sum = 0
  correct_samples = 0
  total_samples = 0

  with torch.no_grad():
    for x_test, y_test in test_dataloader:
      x_test = x_test.to(device)
      y_test = y_test.to(device)
      y_test_hat = model(x_test)
      test_loss = criterion(y_test_hat, y_test.long())
      f1MacroAccumulator.add_batch(y_test, y_test_hat)
      test_loss_sum += test_loss.item()*y_test.shape[0]
      correct_samples += calc_accuracy(y_test_hat,y_test)
      total_samples += y_test.shape[0]

  print(f"Test Loss: {test_loss_sum/total_samples:.4f}")
  print(f"Final Test Accuracy: {correct_samples/total_samples:.4f}")
  print(f"Final F1 Macro: {f1MacroAccumulator.get_macro_f1():.4f}")

  return test_loss_sum/total_samples,correct_samples/total_samples,f1MacroAccumulator.get_macro_f1()

batch_size = 30
#VOCAB_SIZE ,EMBED_DIM已定义
train_dataloader = DataLoader(CustomDataset(X_train_vec, Y_train_vec), batch_size = batch_size, shuffle = True)
dev_dataloader = DataLoader(CustomDataset(X_dev_vec, Y_dev_vec), batch_size = batch_size)
test_dataloader = DataLoader(CustomDataset(X_test_vec, Y_test_vec), batch_size = batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_config = {
    'num_epochs': 40,
    'model_type': 'TextCNN',
    'embed_type': 'glove',
    'embed_freeze': True,
    'embed_dim': 300,
    'num_classes': 5,
    'hidden_size': 128,
    'lr': 1,
    'optimizer': 'Adadelta',
    'kernels': [3, 4, 5],
    'num_filters': 100,
    'exp_name': "test"
}

# 动态生成你要测试的实验组配置
experiment_configs = []

# 实验 A：不同学习率与优化器
"""for opt in ['Adam','SGD']:
    for lr in [0.1,0.001,0.0001]:
        cfg = base_config.copy()
        cfg['exp_name'] = f"Opt_{opt}_LR_{lr}"
        cfg['optimizer'] = opt
        cfg['lr'] = lr
        experiment_configs.append(cfg)"""

# 实验 B1：不同卷积核大小
for kernels in [[2,3,4], [3,4,5], [5,7,9]]:
    cfg = base_config.copy()
    cfg['exp_name'] = f"CNN_Kernels_{kernels}"
    cfg['kernels'] = kernels
    experiment_configs.append(cfg)

# 实验 B2：不同卷积核个数
"""for num_filters in [10,30,50,70,90,150]:
    cfg = base_config.copy()
    cfg['exp_name'] = f"CNN_Filters_{num_filters}"
    cfg['num_filters'] = num_filters
    experiment_configs.append(cfg)"""

# 实验 C：不同的词嵌入方式


# 实验 D： 模型结构对比
"""for m_type in ['TextCNN', 'BiLSTM', 'Transformer']:
    cfg = base_config.copy()
    cfg['model_type'] = m_type
    cfg['exp_name'] = f"Final_Compare_{m_type}"

    if m_type == 'Transformer':
        cfg['lr'] = 5e-5
    else:
        cfg['lr'] = 1e-3

    experiment_configs.append(cfg)"""

# 执行批量实验并记录
results = []
for config in experiment_configs:
    print(f"\n======================================")
    print(f"正在运行实验: {config['exp_name']}")
    print(f"======================================")
    best_dev_loss, best_state_dict = train_and_evaluate(
        config, train_dataloader, dev_dataloader,
        device, VOCAB_SIZE, vocab['<pad>'], a, embedding_matrix
    )
    test_loss,final_test_accuracy,final_f1_macro = test(
        config, best_state_dict, test_dataloader,
        device, VOCAB_SIZE, vocab['<pad>'], a, embedding_matrix)

    results.append({
        'Experiment': config['exp_name'],
        'best_Dev_Loss': best_dev_loss,
        'test_loss': test_loss,
        'Test_Acc': final_test_accuracy,
        'final_f1_macro': final_f1_macro,
        'Config': str(config)
    })

# 导出为表格供写报告使用
df_results = pd.DataFrame(results)
df_results.to_csv("/content/drive/MyDrive/pytorch_beginner/task2/experiment_results_3.csv", index=False)
print("所有实验运行完毕，结果已保存至 experiment_results.csv")

import re
df_up = pd.read_csv("/content/drive/MyDrive/pytorch_beginner/task2/experiment_results_5.csv")
df_down = pd.read_csv("/content/drive/MyDrive/pytorch_beginner/task2/experiment_results_7.csv")
df = pd.concat([df_up,df_down])
def plot_filter_ablation(df):
    # 过滤出 Filters 实验组
    df_filters = df[df['Experiment'].str.contains('Filters', case=False)].copy()
    if df_filters.empty:
        return

    # 从 Config 中提取 num_filters 并排序
    df_filters['num_filters'] = df_filters['Config'].apply(
        lambda x: int(re.search(r"'num_filters':\s*(\d+)", x).group(1))
    )
    df_filters = df_filters.sort_values('num_filters')

    x_data = df_filters['num_filters'].tolist()
    y_acc = df_filters['Test_Acc'].tolist()
    y_loss = df_filters['best_Dev_Loss'].tolist()

    # 全局绘图样式
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 300

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    # 绘制 Test Accuracy (左侧 Y 轴，实线)
    color_acc = '#d62728'
    ax1.set_xlabel('Number of Filters (Per Size)', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12, color=color_acc)
    line1 = ax1.plot(x_data, y_acc, color=color_acc, marker='o', linestyle='-', linewidth=2, label='Test Acc')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_xticks(x_data)

    # 绘制 Best Dev Loss (右侧 Y 轴，虚线)
    ax2 = ax1.twinx()
    color_loss = '#1f77b4'
    ax2.set_ylabel('Best Dev Loss', fontsize=12, color=color_loss)
    line2 = ax2.plot(x_data, y_loss, color=color_loss, marker='s', linestyle='--', linewidth=2, label='Best Dev Loss')
    ax2.tick_params(axis='y', labelcolor=color_loss)

    # 图例合并
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=False, fontsize=10)

    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=0)

    plt.title('Impact of Feature Map Quantities on Model Performance', fontsize=13, pad=15)
    plt.tight_layout()

    # 保存并显示
    plt.savefig('/content/drive/MyDrive/pytorch_beginner/task2/filter_ablation_curve.pdf', format='pdf', bbox_inches='tight')
    display.display(fig)
    plt.close()

# 直接调用绘图
plot_filter_ablation(df)

def setup_academic_style():
    """全局学术绘图样式对齐"""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def plot_grouped_bar(categories, acc_data, f1_data, title, filename):
    setup_academic_style()

    x = np.arange(len(categories))
    width = 0.35  # 柱子宽度

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 使用之前折线图的经典红蓝配色
    color_acc = '#d62728'  # 红色系代表 Accuracy
    color_f1 = '#1f77b4'   # 蓝色系代表 F1-Macro

    # 绘制分组柱状图
    rects1 = ax.bar(x - width/2, acc_data, width, label='Test Accuracy', color=color_acc, alpha=0.85)
    rects2 = ax.bar(x + width/2, f1_data, width, label='Final F1-Macro', color=color_f1, alpha=0.85)

    # 标注数值
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)

    # 添加 Y 轴网格线以增强可读性
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True) # 让网格线在柱子下方

    # 图例设置
    ax.legend(loc='lower right', frameon=False, fontsize=10)

    # 在柱子上方自动标注具体数值（保留3位小数）
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移3个像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)

    # 优化 Y 轴显示范围，让柱子差异更明显
    min_val = min(min(acc_data), min(f1_data))
    max_val = max(max(acc_data), max(f1_data))
    ax.set_ylim([min_val * 0.8, max_val * 1.15])

    fig.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    display.display(fig)
    plt.close()

# 1. 卷积核尺寸消融实验绘图
# 数据来源：experiment_results_3.csv
kernels_cats = ['[2, 3, 4]', '[3, 4, 5]', '[5, 7, 9]']
kernels_acc = [0.456, 0.452, 0.428]
kernels_f1 = [0.375, 0.374, 0.327]

plot_grouped_bar(
    categories=kernels_cats,
    acc_data=kernels_acc,
    f1_data=kernels_f1,
    title='Impact of Kernel Sizes on TextCNN Performance',
    filename='/content/drive/MyDrive/pytorch_beginner/task2/kernel_sizes_comparison.pdf'
)

# 2. 跨架构终极对比实验绘图
# 数据来源：experiment_results_6.csv
models_cats = ['TextCNN', 'BiLSTM', 'Transformer']
models_acc = [0.450, 0.483, 0.488]
models_f1 = [0.293, 0.359, 0.423]

plot_grouped_bar(
    categories=models_cats,
    acc_data=models_acc,
    f1_data=models_f1,
    title='Performance Comparison Across Distinct Architectures',
    filename='/content/drive/MyDrive/pytorch_beginner/task2/architecture_comparison.pdf'
)
