import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import requests
import io

from models.models import DecoderAttnRNN
from transformers import BertTokenizer, BertModel
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TEACHER_FORCE_RATIO = 1
TEST_IMG_URL = "https://www.coinkolik.com/wp-content/uploads/2021/12/shiba-inu-dog.jpg"
PRETRAIN_DIR = "pretrained"
MODEL_DIR = "vqgan_f16_16384"
ENC_MAX_LEN = 50
DEC_MAX_LEN = 256
HID_LEN = 768
TARGET_LEN = 256
VOCAB_IN = 30522
VOCAB_OUT = 16384

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

def preprocess(img, target_image_size=256):
    s = min(img.size)
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img

def stack_images(images, titles=[]):
    w, h = images[0].size[0], images[0].size[1]
    img = Image.new("RGB", (len(images)*w, h))
    for i, cur_img in enumerate(images):
        img.paste(cur_img, (i*w, 0))
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
    encoder_out = torch.zeros(max_length, HID_LEN).to(DEVICE)
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
        

cfg_vqgan = load_config(f"./{PRETRAIN_DIR}/{MODEL_DIR}/configs/model.yaml", display=False)
tokenizer_bert = BertTokenizer.from_pretrained("bert-base-uncased")
model_vqgan = load_vqgan(cfg_vqgan, ckpt_path=f"{PRETRAIN_DIR}/{MODEL_DIR}/checkpoints/last.ckpt").to(DEVICE)
model_bert = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True).to(DEVICE)
model_decoder = DecoderAttnRNN(VOCAB_OUT, max_length=DEC_MAX_LEN).to(DEVICE)
model_vqgan.eval()
model_bert.eval()

image = preprocess(download_image(TEST_IMG_URL)).to(DEVICE)
image_encode, image_tokens = vqgan_encode(image, model_vqgan)
data = ("A picture of shiba-inu breed dog", image_tokens)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model_decoder.parameters(), lr=0.001)

for steps in range(10):
    for i in range(100):
        loss = train(data[0], data[1], model_bert, tokenizer_bert, model_decoder, optimizer, criterion, max_length=DEC_MAX_LEN)
        print(f"Step-{steps} Epoch-{i+1} Loss: {loss}")

    decoded_tokens = evaluate(data[0], model_bert, tokenizer_bert, model_decoder, max_length=DEC_MAX_LEN)

    with torch.no_grad():
        image_from_text = vqgan_decode(preprocess_vqgan(
            model_vqgan.quantize.get_codebook_entry(decoded_tokens, (1, 16, 16, 256))
        ), model_vqgan)
        image_from_image = vqgan_decode(preprocess_vqgan(
            model_vqgan.quantize.get_codebook_entry(image_tokens.flatten(), (1, 16, 16, 256))
        ), model_vqgan)

    images = [ image, image_from_text, image_from_image ]
    images = [ custom_to_pil(img[0]) for img in images ]

    stacked = stack_images(images, ["Original", "Ours" ,"VQGAN"])
    stacked.show()


