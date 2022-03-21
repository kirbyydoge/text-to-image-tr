from turtle import hideturtle
from models.layers import *

import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

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

class DecoderTransform(nn.Module):
	def __init__(self, embedding_dim, hidden_size, n_heads=16,
				n_layers=4, max_len_out=256, vocab_out=16384, dropout=0.5):
		super().__init__()
		self.embedding = nn.Embedding(vocab_out, embedding_dim)
		self.pe = PositionalEncoding(embedding_dim, max_len=max_len_out)
		layer = nn.TransformerDecoderLayer(embedding_dim, n_heads, hidden_size, dropout)
		self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
		self.dense = nn.Linear(embedding_dim, vocab_out)
		self.log_softmax = nn.LogSoftmax()

	def forward(self, target, memory):
		target = self.embedding(target)
		target = self.pe(target)
		decode = self.decoder(target, memory)
		dense = self.dense(decode)
		return self.log_softmax(dense)

class DecoderAttnRNN(nn.Module):
	def __init__(self, output_size, hidden_size=768, dropout=0.1, max_length=50):
		super().__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.max_length = max_length

		self.embedding = nn.Embedding(self.output_size, self.hidden_size)
		self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
		self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(self.hidden_size, self.hidden_size)
		self.out = nn.Linear(self.hidden_size, self.output_size)

	def forward(self, input, hidden, encoder_outputs):
		embedded = self.embedding(input).view(1, 1, -1)
		embedded = self.dropout(embedded)
		attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
		output = torch.cat((embedded[0], attn_applied[0]), 1)
		output = self.attn_combine(output).unsqueeze(0)
		output = F.relu(output)
		output, hidden = self.gru(output, hidden)
		output = F.log_softmax(self.out(output[0]), dim=1)
		return output, hidden, attn_weights

	def hidden_ones(self):
		return torch.zeros(1, 1, self.hidden_size)