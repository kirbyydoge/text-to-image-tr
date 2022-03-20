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
