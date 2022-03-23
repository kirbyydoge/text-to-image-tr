import struct
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import io

from preencode_dataset import load_tensor

from models.models import DecoderAttnRNN
from transformers import BertTokenizer, BertModel
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

PATCH_SIZE = 16
ENCODE_PATH = "D:/C12M/cc12m_tr_eccd.bin"
TRANSLATE_PATH = "D:/C12M/cc12m_tr.tsv"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEACHER_FORCE_RATIO = 1
TEST_IMG_URL = "https://www.coinkolik.com/wp-content/uploads/2021/12/shiba-inu-dog.jpg"
PRETRAIN_DIR = "pretrained"
MODEL_DIR = "vqgan_f16_1024"
ENC_MAX_LEN = 512
DEC_MAX_LEN = 256
HID_LEN = 768
TARGET_LEN = DEC_MAX_LEN
VOCAB_IN = 30522
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
	return vqgan_decode(preprocess_vqgan(
			model.quantize.get_codebook_entry(token.flatten(), (1, image_size, image_size, 256))
		), model)

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

def stack_images(originals, constuctions, titles=[]):
	w, h = originals[0].size[0], originals[0].size[1]
	img = Image.new("RGB", (len(originals)*w, 2*h))
	for i, cur_img in enumerate(originals):
		img.paste(cur_img, (i*w, 0))
	for i, cur_img in enumerate(constuctions):
		img.paste(cur_img, (i*w, h))
	"""
	font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)
	for i, title in enumerate(titles):
		ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font)
	"""
	return img

def train(image_text, image_tokens, bert_encoder, bert_tokenizer, decoder, optim, criterion, max_length=DEC_MAX_LEN):
	optim.zero_grad()
	with torch.no_grad():
		text_tokens = bert_tokenizer(image_text, max_length=ENC_MAX_LEN, padding="max_length", return_tensors="pt").to(DEVICE)
		text_decode = bert_encoder(**text_tokens)[2]
		text_embed = torch.stack(text_decode, dim=0).permute(1, 0, 2, 3)[0][-1]
	encoder_out = torch.zeros(ENC_MAX_LEN, HID_LEN).to(DEVICE)
	for i in range(len(text_embed)):
		encoder_out[i] = text_embed[i]
	decoder_hidden = decoder.hidden_ones().to(DEVICE)
	decoder_input = torch.tensor([[0]]).to(DEVICE)
	use_teacher_forcing = True
	loss = 0
	if use_teacher_forcing:
		for i in range(DEC_MAX_LEN):
			out, decoder_hidden, atten = decoder(decoder_input, decoder_hidden, encoder_out)
			decoder_input = image_tokens[i]
			loss += criterion(out, image_tokens[i])
	else:
		for i in range(DEC_MAX_LEN):
			out, decoder_hidden, atten = decoder(decoder_input, decoder_hidden, encoder_out)
			_, topi = out.data.topk(1)
			decoder_input = topi.squeeze().detach()
			loss += criterion(out, image_tokens[i])
	loss.backward()
	optim.step()
	return loss.item() / len(image_tokens)

def evaluate(image_text, bert_encoder, bert_tokenizer, decoder, max_length=DEC_MAX_LEN):
	with torch.no_grad():
		text_tokens = bert_tokenizer(image_text, max_length=ENC_MAX_LEN, padding="max_length", return_tensors="pt").to(DEVICE)
		text_decode = bert_encoder(**text_tokens)[2]
		text_embed = torch.stack(text_decode, dim=0).permute(1, 0, 2, 3)[0][-1]
		encoder_out = torch.zeros(max_length, HID_LEN).to(DEVICE)
		for i in range(len(text_embed)):
			encoder_out[i] = text_embed[i]
		decoder_hidden = decoder.hidden_ones().to(DEVICE)
		decoder_input = torch.tensor([[0]]).to(DEVICE)
		decoded_words = torch.zeros(max_length, dtype=torch.int64).to(DEVICE)
		for i in range(DEC_MAX_LEN):
			out, decoder_hidden, atten = decoder(decoder_input, decoder_hidden, encoder_out)
			_, topi = out.data.topk(1)
			decoded_words[i] = topi.item()
			decoder_input = topi.squeeze().detach()
	return decoded_words
		
def load_dataset(encoding_path, translation_path, device=DEVICE, max_data=20):
	dataset = {}
	encoding_file = open(encoding_path, "rb")
	patches = struct.unpack("i", encoding_file.read(4))[0]
	index, data = load_tensor(encoding_file, patches)
	data_count = 0
	while index != -1 and data_count != max_data:
		try:
			index, data = load_tensor(encoding_file, patches)
			dataset[index] = {
				"url": "",
				"en": "",
				"tr": "",
				"vector": data.view(patches, 1).to(device)
			}
			data_count += 1
		except Exception as e:
			index = -1
	with open(translation_path, "r", encoding="utf-8") as f:
		for i, line in enumerate(f):
			if i in dataset:
				url, en, tr = line.strip().split("\t")
				dataset[i]["url"] = url
				dataset[i]["en"] = en
				dataset[i]["tr"] = tr
	return dataset

def peek_data(peek_len, model_vqgan):
	dataset = load_dataset(ENCODE_PATH, TRANSLATE_PATH, peek_len)
	originals = []
	constructions = []
	for data in dataset.values():
		originals.append(custom_to_pil(preprocess_vqgan(
					preprocess(
						download_image(data["url"])
					)[0]
		)))
		constructions.append(custom_to_pil(vqgan_from_token(data["vector"].to(DEVICE), model_vqgan, TARGET_LEN // 16)[0]))

	stacked = stack_images(originals, constructions)
	stacked.show()

def train_iters(epochs, dataset, optimizer, criterion, model_decoder, tokenizer_bert, model_bert):
	dataset_len = len(dataset)
	info_steps = dataset_len // 10
	torch.save({
		"epoch": -1,
		"state_dict": model_decoder.state_dict(),
		"optimizer": optimizer.state_dict()
	}, f"./{PRETRAIN_DIR}/model_epochs/inital.pt")
	for i in range(epochs):
		epoch_loss = 0
		for i, data in enumerate(dataset.values()):
			loss = train(data["tr"], data["vector"], model_bert, tokenizer_bert, model_decoder, optimizer, criterion, max_length=DEC_MAX_LEN)
			epoch_loss += loss
			if (i+1) % info_steps == 0:
				print(f"Data-{i+1} AVG Loss: {epoch_loss / i}")
		print(f"Epoch-{i+1} AVG Loss: {epoch_loss / dataset_len}")
		torch.save({
			"epoch": i+1,
			"state_dict": model_decoder.state_dict(),
			"optimizer": optimizer.state_dict()
		}, f"./{PRETRAIN_DIR}/model_epochs/{i}.pt")

if __name__ == "__main__":
	cfg_vqgan = load_config(f"./{PRETRAIN_DIR}/{MODEL_DIR}/configs/model.yaml", display=False)
	tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
	model_bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(DEVICE)
	model_bert.eval()
	model_decoder = DecoderAttnRNN(VOCAB_OUT, max_length=ENC_MAX_LEN).to(DEVICE)
	model_decoder.train()
	#cfg_vqgan = load_config(f"./{PRETRAIN_DIR}/{MODEL_DIR}/configs/model.yaml", display=False)
	#model_vqgan = load_vqgan(cfg_vqgan, ckpt_path=f"{PRETRAIN_DIR}/{MODEL_DIR}/checkpoints/last.ckpt").to(DEVICE)
	criterion = nn.NLLLoss()
	optimizer = optim.SGD(model_decoder.parameters(), lr=0.001)
	dataset = load_dataset(ENCODE_PATH, TRANSLATE_PATH, max_data=-1)
	train_iters(100, dataset, optimizer, criterion, model_decoder, tokenizer_bert, model_bert)




