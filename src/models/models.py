from models.layers import *

import torch.nn as nn

from collections import OrderedDict
from torch.nn.modules import Conv2d

class Decoder(nn.Module):
	def __init__(self, features_in=768, features_out=256, features_hid=512, vocab_size=16384, blocks_per_group=2):
		super().__init__()
		self.feat_in = features_in
		self.feat_hid = features_hid
		self.ch_out = features_out
		self.ch_hid = features_out // 4
		self.p_drop = 0.2
		self.dense_blocks = nn.Sequential(OrderedDict([
			("group_1", nn.Sequential(OrderedDict([
				*[
					(f"block_{i}", DecoderBlockDense(self.feat_in if i == 0 else 2 * self.feat_hid, 2 * self.feat_hid))
					for i in range(blocks_per_group)
				],
				("dropout_1", nn.Dropout(self.p_drop))
			]))),
			("group_2", nn.Sequential(OrderedDict([
				*[
					(f"block_{i}", DecoderBlockDense(2 * self.feat_hid if i == 0 else 1 * self.feat_hid, 1 * self.feat_hid))
					for i in range(blocks_per_group)
				],
				("dropout_2", nn.Dropout(self.p_drop))
			]))),
			("output", nn.Linear(self.feat_hid, features_out))
		]))
		
	def forward(self, x):
		return self.dense_blocks(x)