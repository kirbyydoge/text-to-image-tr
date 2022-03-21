from turtle import forward
import torch
import torch.nn as nn

from collections import OrderedDict
from torch.nn.modules import Conv2d

class DecoderBlockConv(nn.Module):
	def __init__(self, channel_in, channel_out, padding = 0):
		super().__init__()
		self.ch_in = channel_in
		self.ch_out = channel_out
		self.ch_hid = self.ch_out // 4
		self.path = nn.Sequential(OrderedDict([
			("deconv1", Conv2d(self.ch_in, self.ch_hid, 3, padding=padding)),
			("deconv2", Conv2d(self.ch_hid, self.ch_out, 3, padding=padding))
		]))
		
	def forward(self, x):
		return self.path(x)

class DecoderBlockDense(nn.Module):
	def __init__(self, features_in, features_out):
		super().__init__()
		self.feat_in = features_in
		self.feat_out = features_out
		self.feat_hid = self.feat_out // 2
		self.path = nn.Sequential(OrderedDict([
			("fc1", nn.Linear(self.feat_in, self.feat_hid, False)),
			(f"relu_1", nn.ReLU()),
			("fc2", nn.Linear(self.feat_hid, self.feat_out, False)),
			(f"relu_2", nn.ReLU()),
			("norm", nn.LayerNorm((self.feat_out)))
		]))
		
	def forward(self, x):
		return self.path(x)

#https://buomsoo-kim.github.io/attention/2020/04/21/Attention-mechanism-19.md/
class PositionalEncoding(nn.Module):
	def __init__(self, d_model, dropout=0.1, max_len=768):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		pe = torch.zeros(max_len, d_model)
		pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(pos * div)
		pe[:, 1::2] = torch.cos(pos * div)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer("pe", pe)

	def forward(self ,x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)