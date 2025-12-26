#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Braille translation with BERT encoder + MoE-enhanced Transformer decoder.
ASCII-only logs to avoid Unicode console/encoding issues.
"""

import os
import math
import json
import random
import argparse
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup

# -----------------------------
# Device & Repro
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Config
# -----------------------------
class Config:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
        parser.add_argument("--bert_path", type=str, default="bert-base-chinese", help="HF model id or local path")
        parser.add_argument("--max_src_len", type=int, default=128)
        parser.add_argument("--max_tgt_len", type=int, default=128)

        # Model
        parser.add_argument("--d_model", type=int, default=768)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--num_decoder_layers", type=int, default=6)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--moe_n_experts", type=int, default=4)
        parser.add_argument("--moe_aux_coef", type=float, default=1e-2)

        # Train
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--learning_rate", type=float, default=1e-4)
        parser.add_argument("--warmup_ratio", type=float, default=0.06)
        parser.add_argument("--grad_accumulation", type=int, default=1)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--seed", type=int, default=42)

        # Paths
        parser.add_argument("--model_dir", type=str, default="models")
        parser.add_argument("--log_dir", type=str, default="logs")

        args, _ = parser.parse_known_args()

        self.data_dir = args.data_dir
        self.bert_path = args.bert_path

        self.train_file = os.path.join(self.data_dir, "train.json")
        self.val_file = os.path.join(self.data_dir, "val.json")
        self.test_file = os.path.join(self.data_dir, "test.json")

        self.max_src_len = args.max_src_len
        self.max_tgt_len = args.max_tgt_len

        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_decoder_layers = args.num_decoder_layers
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout

        self.moe_n_experts = args.moe_n_experts
        self.moe_aux_coef = args.moe_aux_coef

        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.warmup_ratio = args.warmup_ratio
        self.grad_accumulation = args.grad_accumulation
        self.max_grad_norm = args.max_grad_norm

        self.seed = args.seed

        self.model_dir = args.model_dir
        self.log_dir = args.log_dir


config = Config()
set_seed(config.seed)
os.makedirs(config.model_dir, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)


# -----------------------------
# Metrics JSON helpers
# -----------------------------
def _read_metrics_json(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    return []


def _append_metrics(path: str, record: dict):
    data = _read_metrics_json(path)
    data.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)  # ensure_ascii=True -> ASCII-only file


# -----------------------------
# Dataset
# -----------------------------
class BrailleDataset(Dataset):
    """
    Each record in JSON must have:
      { "input_text": "<Chinese sentence>", "output_text": "<Braille string>" }
    """
    def __init__(self, file_path: str, tokenizer: BertTokenizer, max_src_len: int, max_tgt_len: int, is_train: bool = True):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = pd.DataFrame(data)
        assert 'input_text' in self.data.columns and 'output_text' in self.data.columns, \
            "JSON requires fields: input_text and output_text"

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.is_train = is_train

        if is_train:
            self.build_braille_vocab()
        else:
            self.load_braille_vocab()

    def build_braille_vocab(self):
        all_braille = ''.join(self.data['output_text'].astype(str).tolist())
        braille_chars = sorted(set(all_braille))
        self.braille_vocab = {ch: idx for idx, ch in enumerate(braille_chars)}
        self.braille_vocab['<pad>'] = len(self.braille_vocab)
        self.braille_vocab['<sos>'] = len(self.braille_vocab)
        self.braille_vocab['<eos>'] = len(self.braille_vocab)
        self.braille_vocab['<unk>'] = len(self.braille_vocab)
        self.inv_braille_vocab = {v: k for k, v in self.braille_vocab.items()}
        self.braille_vocab_size = len(self.braille_vocab)
        torch.save({
            'braille_vocab': self.braille_vocab,
            'inv_braille_vocab': self.inv_braille_vocab
        }, os.path.join(config.model_dir, 'braille_vocab.pt'))

    def load_braille_vocab(self):
        vocab_path = os.path.join(config.model_dir, 'braille_vocab.pt')
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path, map_location="cpu")
            self.braille_vocab = vocab_data['braille_vocab']
            self.inv_braille_vocab = vocab_data['inv_braille_vocab']
            self.braille_vocab_size = len(self.braille_vocab)
        else:
            raise FileNotFoundError("braille_vocab.pt not found; please train to create it first.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        txt = str(row['input_text'])
        brl = str(row['output_text'])

        enc = self.tokenizer(
            txt,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        sos = self.braille_vocab['<sos>']
        eos = self.braille_vocab['<eos>']
        pad = self.braille_vocab['<pad>']
        unk = self.braille_vocab['<unk>']

        braille_ids = [sos] + [self.braille_vocab.get(ch, unk) for ch in brl] + [eos]
        if len(braille_ids) > self.max_tgt_len:
            braille_ids = braille_ids[:self.max_tgt_len - 1] + [eos]
        else:
            braille_ids += [pad] * (self.max_tgt_len - len(braille_ids))

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'braille_ids': torch.tensor(braille_ids, dtype=torch.long)
        }


# -----------------------------
# MoE
# -----------------------------
class MoEGate(nn.Module):
    def __init__(self, d_model: int, n_experts: int, noisy: bool = True):
        super().__init__()
        self.proj = nn.Linear(d_model, n_experts, bias=False)
        self.noisy = noisy

    def forward(self, x):
        scores = self.proj(x)
        if self.noisy and self.training:
            scores = scores + torch.randn_like(scores) * 1e-2
        prob = F.softmax(scores, dim=-1)
        top1_idx = prob.argmax(-1)
        top1_prob = prob.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)
        return top1_idx, top1_prob, prob


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class MoEFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, aux_loss_coef: float = 1e-2):
        super().__init__()
        self.experts = nn.ModuleList([ExpertFFN(d_model, d_ff) for _ in range(n_experts)])
        self.gate = MoEGate(d_model, n_experts, noisy=True)
        self.aux_loss_coef = aux_loss_coef
        self.last_aux_loss = torch.tensor(0.0)

    def forward(self, x):
        S, N, D = x.shape
        top1_idx, top1_prob, prob = self.gate(x)
        usage = prob.mean(dim=(0, 1))
        self.last_aux_loss = self.aux_loss_coef * (usage * usage.numel()).var()

        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            mask = (top1_idx == e)
            if mask.any():
                xe = x[mask]
                ye = expert(xe)
                y[mask] = ye * top1_prob[mask].unsqueeze(-1)
        return y


# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        T = x.size(0)
        return x + self.pe[:T]


# -----------------------------
# Model
# -----------------------------
class BrailleTranslatorMoE(nn.Module):
    def __init__(self, braille_vocab_size: int, n_experts: int = 4):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.moe = MoEFFN(d_model=config.d_model, d_ff=config.dim_feedforward, n_experts=n_experts, aux_loss_coef=config.moe_aux_coef)

        self.braille_embedding = nn.Embedding(braille_vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, max_len=config.max_tgt_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.num_decoder_layers)
        self.output_layer = nn.Linear(config.d_model, braille_vocab_size)

        self._init_new_weights()
        self.braille_vocab = None
        self.inv_braille_vocab = None

    def _init_new_weights(self):
        nn.init.xavier_uniform_(self.braille_embedding.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.moe.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def generate_square_subsequent_mask(self, sz: int):
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, braille_ids: torch.Tensor):
        enc = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (N, S, D)
        enc = enc.transpose(0, 1)  # (S, N, D)
        enc = self.moe(enc)        # (S, N, D)

        tgt = self.braille_embedding(braille_ids).transpose(0, 1)  # (T, N, D)
        tgt = self.positional_encoding(tgt)

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
        assert self.braille_vocab is not None, "braille_vocab not injected"
        pad_id = self.braille_vocab['<pad>']

        tgt_key_padding_mask = (braille_ids == pad_id)      # (N, T)
        mem_key_padding_mask = ~attention_mask.bool()       # (N, S)

        dec = self.decoder(
            tgt=tgt,
            memory=enc,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=mem_key_padding_mask
        )  # (T, N, D)

        out = self.output_layer(dec).transpose(0, 1)  # (N, T, V)
        return out

    @torch.no_grad()
    def translate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, max_length: int = 128):
        self.eval()
        enc = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        enc = enc.transpose(0, 1)
        enc = self.moe(enc)

        sos_id = self.braille_vocab['<sos>']
        eos_id = self.braille_vocab['<eos>']
        pad_id = self.braille_vocab['<pad>']

        ys = torch.full((input_ids.size(0), 1), sos_id, dtype=torch.long, device=input_ids.device)

        for _ in range(max_length - 1):
            tgt = self.braille_embedding(ys).transpose(0, 1)
            tgt = self.positional_encoding(tgt)
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0))
            tgt_key_padding_mask = (ys == pad_id)

            dec = self.decoder(
                tgt=tgt,
                memory=enc,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=~attention_mask.bool()
            )

            logits = self.output_layer(dec[-1])  # (N, V)
            next_token = logits.argmax(dim=-1)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)

            if (next_token == eos_id).all():
                break
        return ys


# -----------------------------
# Train / Eval
# -----------------------------
def validate_model(model: BrailleTranslatorMoE, val_loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            braille_ids = batch['braille_ids'].to(device)

            out = model(input_ids, attention_mask, braille_ids[:, :-1])
            loss = criterion(out.reshape(-1, out.size(-1)),
                             braille_ids[:, 1:].reshape(-1))

            if hasattr(model.moe, "last_aux_loss") and torch.is_tensor(model.moe.last_aux_loss):
                loss = loss + model.moe.last_aux_loss.to(loss.device)
            val_loss += loss.item()
    return val_loss / max(len(val_loader), 1)


def _corpus_bleu_smooth(refs: List[str], hyps: List[str], max_n: int = 4, smooth: float = 1.0) -> float:
    from collections import Counter

    def ngrams(s: str, n: int):
        return [tuple(s[i:i+n]) for i in range(len(s)-n+1)] if len(s) >= n else []

    precisions = []
    for n in range(1, max_n + 1):
        match, total = 0, 0
        for r, h in zip(refs, hyps):
            r_ng = Counter(ngrams(r, n))
            h_ng = Counter(ngrams(h, n))
            total += sum(h_ng.values())
            for g in h_ng:
                match += min(h_ng[g], r_ng.get(g, 0))
        precisions.append((match + smooth) / (total + smooth))

    ref_len = sum(len(r) for r in refs)
    hyp_len = sum(len(h) for h in hyps)
    if hyp_len == 0:
        return 0.0
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    return bp * geo


def evaluate_model(model: BrailleTranslatorMoE, file_path: str, tokenizer: BertTokenizer):
    dataset = BrailleDataset(file_path, tokenizer, config.max_src_len, config.max_tgt_len, is_train=False)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.braille_vocab['<pad>'])

    model.eval()
    total_loss, total_tokens, correct_tokens = 0.0, 0, 0
    predictions, targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            braille_ids = batch['braille_ids'].to(device)

            out = model(input_ids, attention_mask, braille_ids[:, :-1])
            loss = criterion(out.reshape(-1, out.size(-1)),
                             braille_ids[:, 1:].reshape(-1))
            if hasattr(model.moe, "last_aux_loss") and torch.is_tensor(model.moe.last_aux_loss):
                loss = loss + model.moe.last_aux_loss.to(loss.device)
            total_loss += loss.item()

            decoded_ids = model.translate(input_ids, attention_mask, max_length=config.max_tgt_len)
            inv_vocab = model.inv_braille_vocab

            for b in range(decoded_ids.size(0)):
                tgt_chars = []
                for idx in braille_ids[b].tolist():
                    ch = inv_vocab.get(idx, '')
                    if ch == '<eos>': break
                    if ch not in ['<sos>', '<pad>']: tgt_chars.append(ch)
                tgt = ''.join(tgt_chars)

                pred_chars = []
                for idx in decoded_ids[b].tolist():
                    ch = inv_vocab.get(idx, '')
                    if ch == '<eos>': break
                    if ch not in ['<sos>', '<pad>']: pred_chars.append(ch)
                pred = ''.join(pred_chars)

                predictions.append(pred)
                targets.append(tgt)

                L = min(len(pred), len(tgt))
                correct_tokens += sum(1 for x, y in zip(pred[:L], tgt[:L]) if x == y)
                total_tokens += max(L, 1)

    avg_loss = total_loss / max(len(loader), 1)
    char_acc = correct_tokens / max(total_tokens, 1)
    bleu = _corpus_bleu_smooth(targets, predictions, max_n=4, smooth=1.0)
    return avg_loss, char_acc, bleu, predictions, targets


def train_model_moe() -> BrailleTranslatorMoE:
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    train_dataset = BrailleDataset(config.train_file, tokenizer, config.max_src_len, config.max_tgt_len, is_train=True)
    val_dataset = BrailleDataset(config.val_file, tokenizer, config.max_src_len, config.max_tgt_len, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = BrailleTranslatorMoE(train_dataset.braille_vocab_size, n_experts=config.moe_n_experts)
    model.braille_vocab = train_dataset.braille_vocab
    model.inv_braille_vocab = train_dataset.inv_braille_vocab
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.braille_vocab['<pad>'])

    total_steps = len(train_loader) * config.epochs
    warmup_steps = max(1, int(config.warmup_ratio * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    metrics_path = os.path.join(config.log_dir, "training_metrics.json")

    for epoch in range(config.epochs):
        epoch_start = time.time()
        lr_before = optimizer.param_groups[0]['lr']

        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{config.epochs}")

        optimizer.zero_grad()
        for step, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            braille_ids = batch['braille_ids'].to(device)

            out = model(input_ids, attention_mask, braille_ids[:, :-1])
            loss = criterion(out.reshape(-1, out.size(-1)),
                             braille_ids[:, 1:].reshape(-1))

            if hasattr(model.moe, "last_aux_loss") and torch.is_tensor(model.moe.last_aux_loss):
                loss = loss + model.moe.last_aux_loss.to(loss.device)

            loss.backward()

            if (step + 1) % config.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_val_loss = validate_model(model, val_loader, criterion)

        try:
            _, char_acc, bleu, preds, tgts = evaluate_model(model, config.val_file, tokenizer)
            bleu_pct = float(bleu * 100.0)
            em = float(np.mean([p == t for p, t in zip(preds, tgts)]))
            cer = float(1.0 - char_acc)
        except Exception as e:
            print("[WARN] validation evaluation failed:", e)
            bleu_pct, char_acc, em, cer = 0.0, 0.0, 0.0, 1.0

        lr_after = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "valid_loss": float(avg_val_loss),
            "CER": float(cer),
            "EM": float(em),
            "BLEU": float(bleu_pct),
            "lr_before": float(lr_before),
            "lr_after": float(lr_after),
            "epoch_time_sec": float(epoch_time),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        _append_metrics(metrics_path, record)

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | "
            f"BLEU {bleu_pct:.2f} | CharAcc {char_acc:.4f} | EM {em:.4f} | "
            f"lr {lr_before:.6f}->{lr_after:.6f} | {epoch_time:.1f}s"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'braille_vocab': model.braille_vocab,
                'inv_braille_vocab': model.inv_braille_vocab,
                'config': vars(config)
            }, os.path.join(config.model_dir, 'best_model.pt'))
            print(f"Saved best model: val loss = {best_val_loss:.4f}")

    best_ckpt = torch.load(os.path.join(config.model_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.braille_vocab = best_ckpt['braille_vocab']
    model.inv_braille_vocab = best_ckpt['inv_braille_vocab']
    model.to(device)

    print("Metrics appended to:", os.path.join(config.log_dir, "training_metrics.json"))
    return model


# -----------------------------
# Inference utils
# -----------------------------
def translate_text(model: BrailleTranslatorMoE, text: str, tokenizer: BertTokenizer, max_length: int = 128) -> str:
    model.eval()
    enc = tokenizer(
        text,
        max_length=config.max_src_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    braille_ids = model.translate(input_ids, attention_mask, max_length)
    seq = []
    for idx in braille_ids[0].tolist():
        ch = model.inv_braille_vocab.get(idx, '')
        if ch == '<eos>':
            break
        if ch not in ['<sos>', '<pad>']:
            seq.append(ch)
    return ''.join(seq)


def interactive_translation(model: BrailleTranslatorMoE, tokenizer: BertTokenizer):
    print("Interactive mode. Type text; type 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
            if text.lower() in ['quit', 'exit']:
                print("Bye.")
                break
            if not text:
                continue
            out = translate_text(model, text, tokenizer, max_length=config.max_tgt_len)
            print("Braille:", out)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print("Error:", e)


# -----------------------------
# Main
# -----------------------------
def main():
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    model_path = os.path.join(config.model_dir, 'best_model.pt')

    if os.path.exists(model_path):
        print("Loading trained model ...")
        checkpoint = torch.load(model_path, map_location=device)
        braille_vocab = checkpoint['braille_vocab']
        braille_vocab_size = len(braille_vocab)

        model = BrailleTranslatorMoE(braille_vocab_size, n_experts=config.moe_n_experts).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.braille_vocab = braille_vocab
        model.inv_braille_vocab = checkpoint['inv_braille_vocab']

        print(f"Loaded epoch {checkpoint['epoch']} with best val loss {checkpoint['loss']:.4f}")
    else:
        print("No model found. Start training ...")
        model = train_model_moe()

    print("Evaluate on test set ...")
    try:
        test_loss, accuracy, bleu_score, predictions, targets = evaluate_model(model, config.test_file, tokenizer)

        results_df = pd.DataFrame({
            'target': targets,
            'prediction': predictions,
            'correct': [p == t for p, t in zip(predictions, targets)]
        })
        results_csv = os.path.join(config.log_dir, 'evaluation_results.csv')
        # Use utf-8-sig to be safe when opening in Excel on Windows
        results_df.to_csv(results_csv, index=False, encoding="utf-8-sig")
        print(f"Test Loss: {test_loss:.4f} | Char Acc: {accuracy:.4f} | BLEU-4: {bleu_score:.4f}")
        print("Saved test details to:", results_csv)

        test_metrics_record = {
            "tag": "test_summary",
            "Test_Loss": float(test_loss),
            "Char_Acc": float(accuracy),
            "BLEU": float(bleu_score * 100.0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        _append_metrics(os.path.join(config.log_dir, "training_metrics.json"), test_metrics_record)
        print("Test summary appended to:", os.path.join(config.log_dir, "training_metrics.json"))

        with open(os.path.join(config.log_dir, "used_config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(config), f, indent=2, ensure_ascii=True)
        print("Saved config snapshot to:", os.path.join(config.log_dir, "used_config.json"))

    except Exception as e:
        print("Evaluation failed:", e)

    interactive_translation(model, tokenizer)


if __name__ == "__main__":
    main()
