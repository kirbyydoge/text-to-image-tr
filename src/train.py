import random
import struct
import gc
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import io
import traceback

from preencode_dataset import load_tensor

from models.models import *
from transformers import BertTokenizer, BertModel
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

NUM_LAYERS = 4
PATCH_SIZE = 16
ENCODE_PATH = "D:/C12M/cc12m_tr_eccd.bin"
TRANSLATE_PATH = "D:/C12M/cc12m_tr.tsv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEACHER_FORCE_RATIO = 1
TEST_IMG_URL = "https://www.coinkolik.com/wp-content/uploads/2021/12/shiba-inu-dog.jpg"
PRETRAIN_DIR = "pretrained"
MODEL_DIR = "vqgan_f16_1024"
ENC_MAX_LEN = 50
DEC_MAX_LEN = 256
HID_LEN = 768
TARGET_LEN = DEC_MAX_LEN
VOCAB_IN = 2048
VOCAB_OUT = 1024

def load_config(config_path, display=False):
	config = OmegaConf.load(config_path)
	if display:
		print(yaml.dump(OmegaConf.to_container(config)))
	return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
	model = VQModel(**config.model.params)
	if ckpt_path is not None:
		sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
		missing, unexpected = model.load_state_dict(sd, strict=False)
	return model.eval()

def preprocess_vqgan(x):
	x = 2.*x - 1.
	return x

def custom_to_pil(x):
	x = x.detach().cpu()
	x = torch.clamp(x, -1., 1.)
	x = (x + 1.)/2.
	x = x.permute(1,2,0).numpy()
	x = (255*x).astype(np.uint8)
	x = Image.fromarray(x)
	if not x.mode == "RGB":
		x = x.convert("RGB")
	return x

def vqgan_encode(x, model):
	with torch.no_grad():
		z, _, [_, _, indices] = model.encode(x)
		return z, indices

def vqgan_from_token(token, model, image_size):
	entry = model.quantize.get_codebook_entry(token.flatten(), (1, image_size, image_size, 256))
	return vqgan_decode(preprocess_vqgan(entry), model)

def vqgan_decode(x, model):
	with torch.no_grad():
		return model.decode(x)

def vqgan_reconstruct(x, model):
	with torch.no_grad():
		z, _, [_, _, indices] = model.encode(x)
		xrec = model.decode(z)
		return xrec

def download_image(url):
	resp = requests.get(url)
	resp.raise_for_status()
	return Image.open(io.BytesIO(resp.content))

def preprocess(img, target_image_size=TARGET_LEN):
	s = min(img.size)
	r = target_image_size / s
	s = (round(r * img.size[1]), round(r * img.size[0]))
	img = TF.resize(img, s, interpolation=Image.LANCZOS)
	img = TF.center_crop(img, output_size=2 * [target_image_size])
	img = torch.unsqueeze(T.ToTensor()(img), 0)
	return img

def stack_images(originals, constuctions=[], titles=[]):
	w, h = originals[0].size[0], originals[0].size[1]
	if len(constuctions) > 0:
		img = Image.new("RGB", (len(originals)*w, 2*h))
	else:
		img = Image.new("RGB", (len(originals)*w, h))
	for i, cur_img in enumerate(originals):
		img.paste(cur_img, (i*w, 0))
	for i, cur_img in enumerate(constuctions):
		img.paste(cur_img, (i*w, h))
	font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 11)
	for i, title in enumerate(titles):
		if len(title) > 35:
			title = f"{title[:35]}..."
		ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font)
	return img


def get_data(data, index, dim=16, device=DEVICE, empty_input=torch.tensor([[0]])):
	top_right = empty_input
	top = empty_input
	top_left = empty_input
	left = empty_input
	if index - dim > 0 and index % dim < dim - 1:
		top_right = data[index - dim + 1]
	if index - dim > 0:
		top = data[index - dim]
	if index - dim > 0 and index % dim > 0:
		top_left = data[index - dim - 1]
	if index % dim > 0:
		left = data[index-1]
	return [top_right, top, top_left, left]

def train(image_text, image_tokens, bert_encoder, bert_tokenizer, decoder, optim, criterion, max_length=DEC_MAX_LEN, sentence_max_len=ENC_MAX_LEN):
	optim.zero_grad()
	tokens_flatten = image_tokens.flatten()
	with torch.no_grad():
		cut_idx = min(sentence_max_len, len(image_text))
		image_text = image_text[:cut_idx]
		text_tokens = bert_tokenizer(image_text, max_length=sentence_max_len, padding="max_length", return_tensors="pt").to(DEVICE)
		text_decode = bert_encoder(**text_tokens)
		text_embed = torch.stack(text_decode[2], dim=0).permute(1, 0, 2, 3)[0][-1]
	encoder_out = torch.zeros(sentence_max_len, HID_LEN).to(DEVICE)
	for i in range(len(text_embed)):
		encoder_out[i] = text_embed[i]
	decoder_hidden = encoder_out[0].repeat(NUM_LAYERS, 1, 1)
	decoded_hiddens = []
	use_teacher_forcing = True
	loss = 0
	if use_teacher_forcing:
		for i in range(max_length):
			decoder_input = [torch.tensor([[data]]).to(DEVICE) for data in get_data(tokens_flatten, i, empty_input=0)]
			decoder_hidden = get_data(decoded_hiddens, i, empty_input=encoder_out[0].repeat(NUM_LAYERS, 1, 1))
			out, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_out)
			loss += criterion(out, image_tokens[i])
			decoded_hiddens.append(decoder_hidden)
	else:
		for i in range(max_length):
			out, decoder_hidden = decoder(decoder_input, decoder_hidden)
			_, topi = out.data.topk(1)
			decoder_input = topi.squeeze().detach()
			loss += criterion(out, image_tokens[i])
	loss.backward()
	optim.step()
	return loss.item() / len(image_tokens)

def evaluate(image_text, bert_encoder, bert_tokenizer, decoder, max_length=DEC_MAX_LEN, sentence_max_len=ENC_MAX_LEN):
	with torch.no_grad():
		cut_idx = min(sentence_max_len, len(image_text))
		image_text = image_text[:cut_idx]
		text_tokens = bert_tokenizer(image_text, max_length=sentence_max_len, padding="max_length", return_tensors="pt").to(DEVICE)
		text_decode = bert_encoder(**text_tokens)
		text_embed = torch.stack(text_decode[2], dim=0).permute(1, 0, 2, 3)[0][-1]
		encoder_out = torch.zeros(sentence_max_len, HID_LEN).to(DEVICE)
		for i in range(len(text_embed)):
			encoder_out[i] = text_embed[i]
		decoder_hidden = encoder_out[0].repeat(NUM_LAYERS, 1, 1)
		decoder_input = torch.tensor([[0]]).to(DEVICE)
		decoded_words = torch.zeros(max_length, dtype=torch.int64).to(DEVICE)
		decoded_hiddens = []
		for i in range(DEC_MAX_LEN):
			decoder_input = [torch.tensor([[data]]).to(DEVICE) for data in get_data(decoded_words, i, empty_input=0)]
			decoder_hidden = get_data(decoded_hiddens, i, empty_input=encoder_out[0].repeat(NUM_LAYERS, 1, 1))
			out, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_out)
			_, topi = out.data.topk(1)
			decoded_words[i] = topi.item()
			decoder_input = topi.squeeze().detach()
			decoded_hiddens.append(decoder_hidden)
	return decoded_words
		
def load_dataset(encoding_path, translation_path, device=DEVICE, max_data=20):
	dataset = {}
	encoding_file = open(encoding_path, "rb")
	patches = struct.unpack("i", encoding_file.read(4))[0]
	print(f"Patches: {patches}")
	index, data = load_tensor(encoding_file, patches)
	data_count = 0
	while index != -1 and data_count != max_data:
		try:
			dataset[index] = {
				"url": "",
				"en": "",
				"tr": "",
				"vector": data.view(patches, 1).to(device)
			}
			data_count += 1
			index, data = load_tensor(encoding_file, patches)
		except Exception as e:
			index = -1
	print(f"Loaded {data_count} encoding tensors.")
	with open(translation_path, "r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			if i in dataset:
				url, en, tr = line.strip().split("\t")
				dataset[i]["url"] = url
				dataset[i]["en"] = en
				dataset[i]["tr"] = tr
	return dataset

def peek_data(peek_len, model_vqgan):
	dataset = load_dataset(ENCODE_PATH, TRANSLATE_PATH, DEVICE, peek_len)
	originals = []
	constructions = []
	for data in dataset.values():
		originals.append(custom_to_pil(preprocess_vqgan(
					preprocess(
						download_image(data["url"]), target_image_size=TARGET_LEN
					)[0]
		)))
		constructions.append(custom_to_pil(vqgan_from_token(data["vector"].to(DEVICE), model_vqgan, TARGET_LEN // 16)[0]))
	stacked = stack_images(originals, constructions)
	stacked.show()

def train_iters(epochs, dataset, optimizer, criterion, model_decoder, tokenizer_bert, model_bert, last_index=0, start_epoch=0, start_percent=0, info_steps=1000, eval_len=1, show_all=True):
	data_list = list(dataset.values())[:-eval_len]
	eval_list = list(dataset.values()) if show_all else list(dataset.values())[-eval_len:]
	dataset_len = len(data_list)
	percent_step = max(dataset_len // 100, 1)
	info_step = max(dataset_len // info_steps, 1)
	images = []
	last_img = None
	for eval_data in eval_list:
		try:
			img = download_image(eval_data["url"])
			img = preprocess(img, target_image_size=TARGET_LEN)[0]
			img = custom_to_pil(preprocess_vqgan(img))
			last_img = img
			images.append(img)
		except:
			images.append(last_img)
	print(f"Starting at index: {last_index} epoch: {start_epoch}")
	for epoch in range(start_epoch, epochs):
		epoch_loss = 0
		progress = last_index * info_step if epoch == start_epoch else 0
		percent = last_index * percent_step if epoch == start_epoch else 0
		start_idx = last_index if epoch == start_epoch else 0
		for i, data in enumerate(data_list[start_idx:]):
			loss = train(data["tr"], data["vector"], model_bert, tokenizer_bert, model_decoder, optimizer, criterion, max_length=DEC_MAX_LEN)
			epoch_loss += loss
			"""if (i+1) % info_step == 0:
				progress += 1
				avg_loss = epoch_loss / i if i > 0 else epoch_loss
				print(f"Progress {progress}/{info_steps}  AVG Loss: {avg_loss}")"""
		try:
			originals = []
			guesses = []
			titles = []
			for i, eval_data in enumerate(eval_list):
				tokens = evaluate(eval_data["tr"], model_bert, tokenizer_bert, model_decoder)
				model_reconstruction = custom_to_pil(vqgan_from_token(tokens, model_vqgan, TARGET_LEN // 16)[0])
				guesses.append(model_reconstruction)
				originals.append(images[i])
				titles.append(eval_data["tr"])
			stack = stack_images(originals, guesses, titles)
			stack.save(f"./eval_images/{epoch}_{progress}.png", "PNG")
		except Exception as e:
			traceback.print_exc()
		print(f"Epoch-{epoch+1} AVG Loss: {epoch_loss / dataset_len}")
if __name__ == "__main__":
	LOAD_VER = "after_train"
	LOAD, TRAIN, EVAL = False, True, False
	cfg_vqgan = load_config(f"./{PRETRAIN_DIR}/{MODEL_DIR}/configs/model.yaml", display=False)
	model_vqgan = load_vqgan(cfg_vqgan, ckpt_path=f"{PRETRAIN_DIR}/{MODEL_DIR}/checkpoints/last.ckpt").to(DEVICE)	
	tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
	model_bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(DEVICE)
	model_vqgan.eval()
	model_bert.eval()
	epoch = 0
	start_percent = 0
	last_idx = 0
	model_decoder = DecoderAttnRNNBert(VOCAB_OUT, max_length=ENC_MAX_LEN, n_layers=NUM_LAYERS).to(DEVICE)
	optimizer = optim.Adam(model_decoder.parameters(), lr=0.001)
	criterion = nn.NLLLoss()
	if LOAD:
		try:
			chkpoint = torch.load(f"{PRETRAIN_DIR}/model_epochs/{LOAD_VER}.pt")
			model_decoder.load_state_dict(chkpoint["state_dict"])
			optimizer.load_state_dict(chkpoint["optimizer"])
			epoch = chkpoint["epoch"]
			last_idx = chkpoint["index"]
		except Exception as e:
			print(e)
	if TRAIN:
		model_decoder.train()
		dataset = load_dataset(ENCODE_PATH, TRANSLATE_PATH, DEVICE, 51)
		try:
			train_iters(1000, dataset, optimizer, criterion, model_decoder, tokenizer_bert, model_bert, last_index=last_idx, eval_len=1, show_all=True)
		except KeyboardInterrupt:
			pass
		torch.save(model_decoder.state_dict(), f"./{PRETRAIN_DIR}/model_epochs/after_train.pt")
	if EVAL:
		model_decoder.eval()
		image_1 = "https://kriptokoin.com/wp-content/uploads/2022/02/shiba-inu-dog.jpg"
		image_2 = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQiEwrAb4GrYaD4LbHFxmjNqqhf9IVizK06uA&usqp=CAU"
		image_3 = "https://imageio.forbes.com/blogs-images/jimgorzelany/files/2017/10/2017-Chevrolet-Camaro-087-e1508353961251-1200x829.jpg?fit=bounds&format=jpg&width=960"
		img1 = preprocess(download_image(image_1)).to(DEVICE)
		dcd1 = vqgan_encode(img1, model_vqgan)[1]
		originals = [
			custom_to_pil(vqgan_from_token(dcd1, model_vqgan, TARGET_LEN // 16)[0]),
		]
		def spec_flip(img, dimensions=[16, 16]):
			copy = torch.clone(img)
			rows = dimensions[0]
			cols = dimensions[1]
			for i in range(rows):
				for j in range(cols):
					copy[i * cols + j][0] = img[(rows - i - 1) * cols + j][0]
			return copy
		dcd1 = spec_flip(dcd1)
		tests = [
			custom_to_pil(vqgan_from_token(dcd1, model_vqgan, TARGET_LEN // 16)[0]),
		]
		stack = stack_images(originals, tests)
		stack.show()