import requests
import torch
import time
import yaml
import io
import os
import numpy as np
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import struct
from queue import Queue
from threading import Thread

HEADERS = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
DATA_PATH = "D:/C12M/cc12m_tr.tsv"
OUT_PATH = "D:/C12M/cc12m_tr_encoded.bin"
ECC_PATH = "D:/C12M/cc12m_tr_eccd.bin"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PRETRAIN_DIR = "pretrained"
MODEL_DIR = "vqgan_f16_16384"
NUM_PATCHES = 256
NUM_DOWNLOADERS = 100
NUM_PREENCODERS = 1
INFO_FREQ = 100

print(f"DEVICE: {DEVICE}")

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

def download_image(url, headers=HEADERS):
    resp = requests.get(url, headers=headers)
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

def download_scheduler(data_path, url_queue:Queue, start_idx, max_download):
	scheduled_count = 0
	with open(data_path, "r", encoding="utf-8") as src:
		for i, line in enumerate(src):
			if i < start_idx:
				continue
			url, _, _ = line.strip().split("\t")
			url_queue.put((i, url))
			scheduled_count += 1
			if scheduled_count == max_download:
				break
		url_queue.put(None)
	print(f"Scheduler: Exitting.")

def download_worker(url_queue:Queue, image_queue:Queue, worker_id):
	running = True
	while running:
		task = url_queue.get()
		if task is None:
			url_queue.put(None)
			running = False
			break
		try:
			idx = task[0]
			url = task[1]
			img = download_image(url)
			image_queue.put((idx, img))
		except Exception as e:
			pass
	print(f"WORKER-{worker_id}: Exitting.")
	image_queue.put(None)

def preencode_worker(image_queue:Queue, token_queue:Queue, worker_id, num_patches, device, info_freq, num_downloaders):
	time.sleep(worker_id * 3.5) # delay so models are not loaded in parallel
	cfg_vqgan = load_config(f"{PRETRAIN_DIR}/{MODEL_DIR}/configs/model.yaml", display=False)
	model_vqgan = load_vqgan(cfg_vqgan, ckpt_path=f"{PRETRAIN_DIR}/{MODEL_DIR}/checkpoints/last.ckpt").to(DEVICE)
	model_vqgan.eval()
	running = True
	waiting_time = 0
	process_time = 0
	profile_count = 0
	fallback = 1
	retired_count = 0
	while running:
		wait_start = time.time()
		task = image_queue.get()
		wait_end = time.time()
		waiting_time += wait_end - wait_start
		if task is None:
			image_queue.put(None)
			retired_count += 1
			running = retired_count < num_downloaders
			continue
		process_start = time.time()
		idx = task[0]
		try:
			img = preprocess(task[1], num_patches).to(device)
			with torch.no_grad():
				_, tokens = vqgan_encode(img, model_vqgan)
			tokens = tokens.flatten().type(torch.int32).to("cpu")
			token_queue.put((idx, tokens))
		except Exception as e:
			pass
		process_end = time.time()
		process_time += process_end - process_start
		profile_count += 1
		if profile_count == info_freq:
			avg_wait = waiting_time / info_freq
			avg_proc = process_time / info_freq
			print(f"ENCODER-{worker_id}: AVG Wait - {avg_wait} AVG Process - {avg_proc}")
			waiting_time = 0
			process_time = 0
			profile_count = 0
			if worker_id > 0 and avg_wait > avg_proc: # Too many workers
				sleep_amt = worker_id * fallback * 15
				print(f"ENCODER-{worker_id}: AVG Wait > AVG Process. Sleeping for {sleep_amt} with fallback {fallback}.")
				time.sleep(sleep_amt)
				fallback += 1
			else:
				fallback = max(1, fallback - 1)
	print(f"ENCODER-{worker_id}: Exitting.")
	token_queue.put(None)

def token_combiner(out_path, token_queue:Queue, num_patches, num_preencoders, info_freq):
	running = True
	retired_preencoders = 0
	written_lines = 0
	if os.path.exists(out_path):
		file = open(out_path, "ab")
	else:
		file = open(out_path, "wb")
		file.write(struct.pack("i", num_patches))
	start = time.time()
	while running:
		task = token_queue.get()
		if task is None:
			retired_preencoders += 1
			running = retired_preencoders < num_preencoders
			continue
		idx = task[0]
		tokens = task[1]
		if file.closed:
			file = open(out_path, "ab")
		try:
			assert len(tokens) == num_patches
			file.write(struct.pack("i", idx))
			file.write(struct.pack(f"{num_patches}i", *tokens))
			written_lines += 1
			if written_lines % info_freq == 0:
				print(f"COMBINER: Encoded {written_lines} images. EPS: {info_freq / (time.time() - start)}")
				start = time.time()
		except Exception as e:
			print(f"COMBINER: {e}")
	print(f"COMBINER: Exitting.")
	file.flush()
	file.close()

def encode_pipelined(data_path, out_path, start_idx, max_download=-1):
	url_queue = Queue(NUM_DOWNLOADERS*10)
	image_queue = Queue()
	token_queue = Queue()
	scheduler = Thread(target=download_scheduler, args=(data_path, url_queue, start_idx, max_download))
	downloaders = []
	preencoders = []
	for i in range(NUM_DOWNLOADERS):
		thread = Thread(target=download_worker, args=(url_queue, image_queue, i))
		downloaders.append(thread)
	for i in range(NUM_PREENCODERS):
		thread = Thread(target=preencode_worker, args=(image_queue, token_queue, i, NUM_PATCHES, DEVICE, INFO_FREQ, NUM_DOWNLOADERS))
		preencoders.append(thread)
	combiner = Thread(target=token_combiner, args=(out_path, token_queue, NUM_PATCHES, NUM_PREENCODERS, INFO_FREQ * NUM_PREENCODERS))
	scheduler.start()
	for i in range(NUM_DOWNLOADERS):
		downloaders[i].start()
	for i in range(NUM_PREENCODERS):
		preencoders[i].start()
	combiner.start()
	scheduler.join()
	for i in range(NUM_DOWNLOADERS):
		downloaders[i].join()
	for i in range(NUM_PREENCODERS):
		preencoders[i].join()
	combiner.join()

def load_tensor(f, patches):
	idx_bytes = f.read(4)
	if not idx_bytes:
		return -1, None
	index = struct.unpack("i", idx_bytes)[0]
	data = torch.zeros(patches, dtype=torch.int32)
	for i in range(patches):
		data[i] = struct.unpack("i", f.read(4))[0]
	return index, data

def check_existing(path):
	if not os.path.exists(path):
		return -1
	file = open(path, "rb")
	patches = struct.unpack("i", file.read(4))[0]
	index = 0
	max_index = -1
	while index != -1:
		index, data = load_tensor(file, patches)
		if index > max_index:
			max_index = index
	return max_index

def ecc_existing(out_path, ecc_path):
	if not os.path.exists(out_path):
		return
	f_data = open(out_path, "rb")
	f_ecc = open(ecc_path, "wb")
	patches = struct.unpack("i", f_data.read(4))[0]
	f_ecc.write(struct.pack("i", patches))
	index, data = load_tensor(f_data, patches)
	while index != -1:
		f_ecc.write(struct.pack("i", index))
		f_ecc.write(struct.pack(f"{patches}i", *data))
		try:
			index, data = load_tensor(f_data, patches)
		except Exception as e:
			print(e)
			index = -1
	f_data.close()
	f_ecc.close()

if __name__ == "__main__":
	ECC = False
	if ECC:
		ecc_existing(OUT_PATH, ECC_PATH)
	else:
		max_index = check_existing(OUT_PATH)
		print(f"Last indexed tensor: {max_index}")
		encode_pipelined(DATA_PATH, OUT_PATH, max_index + 1)