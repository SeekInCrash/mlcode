from pathlib import Path
import random
import numpy as np
from PIL import Image
import time
import os
from glob import glob
from tqdm import tqdm
import editdistance
import argparse
from transformer_common import *

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import torchvision.transforms as T
import torch.nn.functional as F

# class OCR(nn.Module):
#     def __init__(self, vocab_len, hidden_dim=512, nheads=8,
#                  num_encoder_layers=4, num_decoder_layers=4):
#         super().__init__()

#         # create ResNet backbone
#         self.backbone = resnet18(pretrained=True)
#         del self.backbone.fc

#         # create conversion layer
#         # self.conv = nn.Conv2d(512, hidden_dim, 1)

#         # create a default PyTorch transformer
#         self.transformer = nn.Transformer(
#             hidden_dim, nheads, num_encoder_layers, num_decoder_layers)

#         # prediction heads with length of vocab
#         # DETR used basic 3 layer MLP for output
#         self.generator = nn.Linear(hidden_dim,vocab_len)

#         # output positional encodings (object queries)
#         self.embedding = nn.Embedding(vocab_len, hidden_dim)
#         self.pos_embedding = PositionalEncoding(hidden_dim)

#         self.tgt_mask = None
  
#     def generate_square_subsequent_mask(self, sz):
#         mask = torch.triu(torch.ones(sz, sz), 1)
#         # mask = mask.masked_fill(mask==1, -1e9)
#         return mask == 1

#     def backbone_forward(self,x):
#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)   
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x = self.backbone.layer1(x)
#         x = self.backbone.layer2(x)
#         x = self.backbone.layer3(x)
#         x = self.backbone.layer4(x)
#         return x


#     def make_len_mask(self, inp):
#         return (inp == 0).transpose(0, 1)


#     def forward(self, inputs, tgt):
#         # propagate inputs through ResNet-101 up to avg-pool layer
#         x = self.backbone_forward(inputs)

#         # convert from 2048 to 256 feature planes for the transformer
#         # x = self.conv(x)

#         # generating subsequent mask for target
#         if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
#             self.tgt_mask = self.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)

#         # Padding mask
#         tgt_pad_mask = self.make_len_mask(tgt)

#         # Getting postional encoding for target
#         tgt = self.embedding(tgt)
#         tgt = self.pos_embedding(tgt)
        
#         output = self.transformer(
#           src=self.pos_embedding(x.flatten(2).permute(2, 0, 1)), 
#           tgt=tgt.permute(1,0,2), 
#           tgt_mask=self.tgt_mask, 
#           tgt_key_padding_mask=tgt_pad_mask.permute(1,0))
#         return self.generator(output.transpose(0,1))
    
#     def get_memory(self, imgs):
#       x = self.backbone_forward(imgs)
#       return self.transformer.encoder(self.pos_embedding(x.flatten(2).permute(2, 0, 1)))
  
#     def get_output(self, decode_in, memory, mask):
#       '''
#       (T, N, E)
#       (T, S)
#       (T, T)
#       '''
#       embeded_in = self.pos_embedding(self.embedding(decode_in))
#       tgt = self.transformer.decoder(embeded_in.transpose(0, 1), memory, tgt_mask=mask)
#       return self.generator(tgt.transpose(0, 1))

# def make_model(vocab_size):
#   model = OCR(vocab_size).to(device)
#   for child in model.children():
#     if child is model.backbone:
#       for m in child.parameters():
#         m.requires_grad = False
#     else:
#       for m in child.parameters():
#         if m.dim() > 1:
#           nn.init.xavier_uniform_(m)
#   return model

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        """
        norm: loss的归一化系数，用batch中所有有效token数即可
        """
        x = self.generator(x)
        x_ = x.contiguous().view(-1, x.size(-1))
        y_ = y.contiguous().view(-1)
        loss = self.criterion(x_, y_)
        loss /= norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm

class OCR_EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. 
    Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, src_position, tgt_embed, generator):
        super(OCR_EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed    # input embedding module(input embedding + positional encode)
        self.src_position = src_position
        self.tgt_embed = tgt_embed    # ouput embedding module
        self.generator = generator    # output generation module
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        memory = self.encode(src, src_mask)
        res = self.decode(memory, src_mask, tgt, tgt_mask)
        return res
    
    def encode(self, src, src_mask):
        # feature extract
        src_embedds = self.src_embed(src)
        # 将src_embedds由shape(bs, model_dim, 1, max_ratio) 处理为transformer期望的输入shape(bs, 时间步, model_dim)
        src_embedds = src_embedds.squeeze(-2)
        src_embedds = src_embedds.permute(0, 2, 1)

        # position encode
        src_embedds = self.src_position(src_embedds)

        return self.encoder(src_embedds, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        target_embedds = self.tgt_embed(tgt)
        return self.decoder(target_embedds, memory, src_mask, tgt_mask)

def make_ocr_model(tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建模型
    params:
        tgt_vocab: 输出的词典大小(82)
        N: 编码器和解码器堆叠基础模块的个数
        d_model: 模型中embedding的size，默认512
        d_ff: FeedForward Layer层中embedding的size，默认2048
        h: MultiHeadAttention中多头的个数，必须被d_model整除
        dropout:
    """
    c = copy.deepcopy

    backbone = resnet18(pretrained=True)
    backbone = nn.Sequential(*list(backbone.children())[:-2])    # 去掉最后两个层 (global average pooling and fc layer)

    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    model = OCR_EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=backbone,
        src_position=c(position),
        tgt_embed=nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        generator=Generator(d_model, tgt_vocab))
    
    # Initialize parameters with Glorot / fan_avg.
    for child in model.children():
        if child is backbone:
            # 将backbone的权重设为不计算梯度
            for param in child.parameters():
                param.requires_grad = False
            # 预训练好的backbone不进行随机初始化，其余模块进行随机初始化
            continue
        for p in child.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    return model


def create_id2lb_map():
  return ("皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", 'O', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

class Tokenizer():
  def __init__(self, token_set, max_len=8):
    self.PAD = 0
    self.SOS = 1
    self.EOS = 2
    self.decode_map = ('*', '<', '>') + tuple(token_set)
    self.VSIZE = len(self.decode_map)
    self.encode_map = self.d2e_map(self.decode_map)
    self.MAX_LEN = max_len
  
  @staticmethod
  def d2e_map(dmap):
    ret = {}
    cnt = 0
    for t in dmap:
      ret[t] = cnt
      cnt += 1
    return ret
  
  def encode(self, seq):
    '''
    seq: original input sequence, without padding, SOS and EOS
    '''
    return [self.SOS] + [self.encode_map[t] for t in seq] + [self.EOS]
  
  def decode(self, seq):
    return "".join([self.decode_map[t] for t in seq if t != self.PAD])

class MyDataset(Dataset):
  def __init__(self, root, tokenizer, max_ratio, mode='train'):
    mode = mode if mode == 'train' else 'val'
    cache = np.load(root + f'/lp_{mode}.cache.npy', allow_pickle=True).item()
    self.root = root
    self.img_paths = np.array(list(cache.keys()))
    self.labels = np.array(list(cache.values()))
    self.max_ratio = max_ratio*3
    self.tokenizer = tokenizer

    p = 0.1 if mode == 'train' else 0
    self.transformer = T.Compose([
      T.ColorJitter(p, p, p),
      T.ToTensor(),
      T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
  
  def __getitem__(self, index):
    img = Image.open(self.root + f'/{self.img_paths[index]}').convert('RGB')
    w, h = img.size
    ratio = round((w / h) * 3)   # 将宽拉长3倍，然后四舍五入
    if ratio == 0:
        ratio = 1
    if ratio > self.max_ratio:
        ratio = self.max_ratio
    h_new = 32
    w_new = h_new * ratio
    img_resize = img.resize((w_new, h_new), Image.BILINEAR)
    img_padd = Image.new('RGB', (32*self.max_ratio, 32), (0,0,0))
    img_padd.paste(img_resize, (0, 0))
    img = self.transformer(img_padd)

    encode_mask = (torch.tensor([1] * ratio + [0] * (self.max_ratio - ratio)) != 0).unsqueeze(0)
    gt = torch.tensor(self.tokenizer.encode(self.labels[index]))

    decode_in = gt[:-1]
    decode_out = gt[1:]
    decode_mask = self.make_std_mask(decode_in, self.tokenizer.PAD)
    ntokens = (decode_out != self.tokenizer.PAD).data.sum()
    return img, encode_mask, decode_in, decode_out, decode_mask, ntokens
  
  @staticmethod
  def make_std_mask(tgt, pad):
    """
    Create a mask to hide padding and future words.
    padd 和 future words 均在mask中用0表示
    """
    tgt_mask = (tgt != pad)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
    tgt_mask = tgt_mask.squeeze(0)   # subsequent返回值的shape是(1, N, N)
    return tgt_mask
  
  def __len__(self):
    return len(self.img_paths)

class Recorder():
  def __init__(self, root="."):
    log_files = glob(f"{root}/log-*.npy")
    l = len(log_files)
    assert l == 1 or l == 0, "log file conflict"
    if l == 1:
      log = np.load(log_files[0], allow_pickle=True).item()
    self.save_path = f"{root}/log-{time.strftime('%y-%m-%d_%H:%M:%S', time.localtime())}" if l == 0 else log_files[0]
    self.train_loss = [] if l == 0 else log['train_loss']
    self.val_loss = [] if l == 0 else log['val_loss']
    self.wacc = [] if l == 0 else log['wacc'] # word accuracy
    self.sacc = [] if l == 0 else log['sacc'] # sentence accuracy
  
  def save(self):
    np.save(self.save_path, {
        "train_loss": self.train_loss,
        "val_loss": self.val_loss,
        "wacc": self.wacc,
        "sacc": self.sacc
    })

class ComputeLoss(nn.Module):
  def __init__(self, length) -> None:
      super().__init__()
      self.length = length
  
  def forward(self, x, y):
  #   return F.kl_div(x.log_softmax(-1).contiguous().view(-1, self.vocab_size), 
  #     F.one_hot(y.contiguous().view(-1), num_classes=self.vocab_size).float())
    bs = y.size(0)
    return F.ctc_loss(x.log_softmax(-1).transpose(0, 1), y, (self.length,) * bs, (self.length,)*bs)

def train(model, train_dataloader, val_dataloader, optimizer, loss_fn, epochs, save_path, checkpoint=None, device='cuda'):
  os.makedirs(save_path, exist_ok=True)
  np.random.seed(int(time.time()))
  recorder = Recorder(save_path)
  model.to(device)

  start_epoch = 1
  val_interval = round(epochs * 0.05) # 2.5% epochs
  best_val_loss = np.inf

  last = os.path.join(save_path, 'last.ckpt')
  best = os.path.join(save_path, 'best.pt') 

  lr_scheduler = WarmupCosineScheduler(optimizer, 10, warmup_end=round(epochs*.1), total_epochs=epochs)
  if checkpoint is not None:
    assert str(checkpoint).endswith('.ckpt'), "checkpoint file should be end with '.ckpt'"
    ckpt = torch.load(Path(save_path) / checkpoint)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    # lr_scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['start_epoch']
    best_val_loss = ckpt['best_val_loss']
    val_interval = round((epochs - start_epoch + 1) * 0.05)
    print(f"\nresume traing from checkpoint '{checkpoint}'\nstart epoch: {start_epoch}, best val loss: {best_val_loss}")
    del ckpt
  print(f"val interval: {val_interval}")
  # train_len = len(train_dataloader.dataset)
  # valid_len = len(val_dataloader.dataset)
  # vocab_size = train_dataloader.dataset.tokenizer.VSIZE
  # nbatches = len(train_dataloader) 

  try:
    for epoch in range(start_epoch, epochs+1):
      # train
      total_train_loss = 0. # or 滑动平均损失
      total_tokens = 0
      model.train()
      tloss_fn = SimpleLossCompute(model.generator, loss_fn, optimizer)
      for ibatch, (imgs, encode_mask, decode_in, decode_out, decode_mask, ntokens) in tqdm(enumerate(train_dataloader), desc=f"[ Epoch {epoch} / {epochs}, train ]", total=len(train_dataloader)):
        imgs = imgs.to(device)                        
        encode_mask = encode_mask.to(device)                                
        decode_in = decode_in.to(device)                                
        decode_out = decode_out.to(device)                    
        decode_mask = decode_mask.to(device)
        ntokens = torch.sum(ntokens).to(device)

        # step = epoch * nbatches + ibatch + 1
        pred = model(imgs, decode_in, encode_mask, decode_mask)
        train_loss = tloss_fn(pred, decode_out, ntokens)
        # ema_train_loss = train_loss.item() if ibatch == 0 else (0.75*ema_train_loss + 0.25*train_loss.item())
        total_train_loss += train_loss
        total_tokens += ntokens
      lr_scheduler.step()
      total_train_loss /= total_tokens
      print(f"train loss: {total_train_loss}")
      recorder.train_loss.append(total_train_loss)

      if epoch == start_epoch or epoch % val_interval == 0 or epoch == epochs:
        # validation
        total_valid_loss = 0.
        total_tokens = 0
        gwacc, gsacc = [], []
        model.eval()
        vloss_fn = SimpleLossCompute(model.generator, loss_fn, None)
        for ibatch, (imgs, encode_mask, decode_in, decode_out, decode_mask, ntokens) in tqdm(enumerate(val_dataloader), desc=f"[ Epoch {epoch} / {epochs}, validation ]", total=len(val_dataloader)):
          imgs = imgs.to(device)                        
          encode_mask = encode_mask.to(device)                                
          decode_in = decode_in.to(device)                                
          decode_out = decode_out.to(device)                    
          decode_mask = decode_mask.to(device)
          ntokens = torch.sum(ntokens).to(device)

          pred = model(imgs, decode_in, encode_mask, decode_mask)
          valid_loss = vloss_fn(pred, decode_out, ntokens)
          # ema_valid_loss = valid_loss.item() if ibatch == 0 else (0.75*ema_valid_loss + 0.25*valid_loss.item())
          total_valid_loss += valid_loss
          total_tokens += ntokens

          # Test accuracy
          # bs = imgs.size(0)
          # for i in range(bs):
          #   cur_img = imgs[i].unsqueeze(0)
          #   cur_encode_mask = encode_mask[i].unsqueeze(0)
          #   result = greedy_decode(model, tokenizer, cur_img, cur_encode_mask).view(1, -1)
          #   preds = result if i == 0 else torch.cat((preds, result))
          # w, s = metrics(preds[:, :-1].tolist(), decode_out[:, :-1], tokenizer)
          preds = bgreedy_decode(model, tokenizer, imgs, encode_mask)
          w, s = metrics(preds, decode_out[:, :-1], tokenizer)
          gwacc.append(w)
          gsacc.append(s)

        total_valid_loss /= total_tokens
        recorder.val_loss.append(total_valid_loss)
        gwacc = np.mean(gwacc)
        gsacc = np.mean(gsacc)
        recorder.wacc.append(gwacc)
        recorder.sacc.append(gsacc)
        print(f"[ epoch: {epoch} / {epochs} ], train EMA loss: {total_train_loss}, valid EMA loss: {total_valid_loss}\nwacc: {gwacc}, sacc: {gsacc}\n")
      
        # Saving
        recorder.save()
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "scheduler": lr_scheduler.state_dict(),
            "start_epoch": epoch + 1,
            "best_val_loss": best_val_loss
          }, last
        )
        if total_valid_loss < best_val_loss:
          best_val_loss = total_valid_loss
          torch.save(model.state_dict(), best)
          print("[====== update best model ========]\n")
  except KeyboardInterrupt or RuntimeError:
    print("Wait for saving...")
    recorder.save()
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # "scheduler": lr_scheduler.state_dict(),
        "start_epoch": epoch + 1,
        "best_val_loss": best_val_loss
      }, os.path.join(save_path, 'brk_last.ckpt') 
    )
    if total_valid_loss < best_val_loss:
      print("[====== update best model ========]\n")
      best_val_loss = total_valid_loss
      torch.save(model.state_dict(), os.path.join(save_path, 'brk_best.pt'))
  torch.cuda.empty_cache()
  
def greedy_decode(model, tokenizer, src, src_mask):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(tokenizer.SOS).type_as(src.data).long()
    for i in range(tokenizer.MAX_LEN):
        out = model.decode(memory, src_mask, 
                           Variable(ys), 
                           Variable(subsequent_mask(ys.size(-1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        next_word = torch.ones(1, 1).type_as(src.data).fill_(next_word).long()
        ys = torch.cat([ys, next_word], dim=1)
        # next_word = int(next_word)
        # if next_word == tokenizer.EOS:
        #     break
        #ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    ys = ys[0, 1:]
    return ys

def bgreedy_decode(model, tokenizer, src, src_mask):
  '''
  img: (bs, C, H, W)
  '''
  bs = src.size(0)
  memory = model.encode(src, src_mask)
  out_seq = torch.ones(bs, 1).fill_(tokenizer.SOS).long().to(src.device)
  for i in range(tokenizer.MAX_LEN):
    out = model.decode(memory, src_mask, Variable(out_seq), 
      Variable(subsequent_mask(i+1).type_as(src.data)))
    prob = model.generator(out[:, -1])
    tk = prob.argmax(-1).view(-1, 1)
    out_seq = torch.cat((out_seq, tk), dim=-1)
  return out_seq[:, 1:-1].tolist()

def metrics(preds, gts, tokenizer):
  if isinstance(gts, torch.Tensor):
   gts = gts.tolist() 
  assert len(preds) == len(gts), "prediction size should be same as ground truth labels"
  l = float(max(len(preds[0]), len(gts[0])))

  wdist = [1- editdistance.eval(pred, gt)/l for pred, gt in zip(preds, gts)] # word distence
  sdist = editdistance.eval([tokenizer.decode(p) for p in preds], [tokenizer.decode(g) for g in gts]) # sentence distence
  return np.mean(wdist), 1-sdist/float(len(preds))

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
  def __init__(self, optimizer, multiplier, warmup_end, total_epochs):
    self.multiplier = multiplier
    self.warmup_end = warmup_end
    self.after_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    self.finished = False
    super().__init__(optimizer)

  def get_lr(self):
    if self.last_epoch > self.warmup_end:
      if self.after_scheduler:
        if not self.finished:
          self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
          self.finished = True
        return self.after_scheduler.get_lr()
      return [base_lr * self.multiplier for base_lr in self.base_lrs]
    return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_end + 1.) for base_lr in self.base_lrs]


  def step(self, epoch=None, metrics=None):
    if self.finished and self.after_scheduler:
      if epoch is None:
        self.after_scheduler.step(None)
      else:
        self.after_scheduler.step(epoch - self.warmup_end)
    else:
      return super().step(epoch)

def set_random_seed(seed = 0):
  import torch.backends.cudnn as cudnn
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--datasets_root", type=str, default="lp")
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--epochs", type=int, default=300)
  parser.add_argument("--batch_size", type=int, default=256)
  parser.add_argument("--save_path", type=str, default=".")
  parser.add_argument("--test_use", type=str, default=None)
  parser.add_argument("--resume", action="store_true")
  cfg = parser.parse_args()
  # set_random_seed()

  # cfg.datasets_root = "/home/undo/code/ML/src/gp/datasets/CCPD2019/lp"
  # cfg.batch_size = 16
  # cfg.resume = True
  # cfg.epochs=2000
  # cfg.test_use = 'best.pt'

  batch_size = cfg.batch_size
  dataset_root = cfg.datasets_root
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  print("Configuration: ", cfg)
  tokenizer = Tokenizer(create_id2lb_map())
  model = make_ocr_model(tokenizer.VSIZE, N=5).to(device)
  # ========= Profile ===========
  print(f"Parameters: {sum(m.numel() for m in model.parameters()) / (2**20) * 4:.3f}MB, \
    Gradients: {sum(m.numel() for m in model.parameters() if m.requires_grad) / 1e6:.3f}MFLOPS")

  if cfg.test_use is None:
    # train
    train_set = MyDataset(dataset_root, tokenizer, 6, mode='train')
    val_set = MyDataset(dataset_root, tokenizer, 6, mode='val')
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_set, batch_size*2, shuffle=False, num_workers=8)
    criterion = LabelSmoothing(tokenizer.VSIZE, tokenizer.PAD, 0.)
    # optimizer = torch.optim.AdamW(model.parameters(), cfg.lr/10, betas=(0.9, 0.998), weight_decay=0.0005)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    train(model, train_dataloader, val_dataloader, optimizer, criterion, cfg.epochs, \
      save_path=cfg.save_path,
      checkpoint= 'last.ckpt' if cfg.resume else None,
      device=device)
  else:
    assert cfg.test_use.endswith('.pt') or cfg.test_use.endswith('.ckpt'), "can not recognize"
    val_set = MyDataset(dataset_root, tokenizer, 6, mode='val')
    val_dataloader = DataLoader(val_set, batch_size, shuffle=False, num_workers=8)
    if cfg.test_use.endswith('.ckpt'):
      model.load_state_dict(torch.load(Path(cfg.save_path) / cfg.test_use)['model'])
    else:
      model.load_state_dict(torch.load(Path(cfg.save_path) / cfg.test_use))
    
    model.eval()
    gwacc, gsacc = [], []
    start_time = time.time()
    bt = np.random.randint(0, len(val_dataloader))
    for ibatch, batch in enumerate(val_dataloader):
      imgs, encode_mask, decode_out = batch[0].to(device), batch[1].to(device), batch[3].to(device)
      preds = bgreedy_decode(model, tokenizer, imgs, encode_mask)
      if ibatch == bt:
        for i, (pred, gt) in enumerate(zip(preds, decode_out[:, :-1].tolist())):
          print(pred, gt, tokenizer.decode(pred), tokenizer.decode(gt))
      w, s = metrics(preds, decode_out[:, :-1], tokenizer)
      gwacc.append(w)
      gsacc.append(s)
    end_time = time.time()
    
    print(f"wacc: {np.mean(gwacc)}, sacc: {np.mean(gsacc)}, {batch_size*len(val_dataloader) / (end_time - start_time):.2f}it/s")
    # for i, (pred, gt) in enumerate(zip(preds, decode_out[:, :-1].tolist())):
    #   print(pred, gt, tokenizer.decode(pred), tokenizer.decode(gt))
    # print(metrics(preds, decode_out[:, :-1], tokenizer), f"{batch_size / (end_time - start_time):.2f}it/s")