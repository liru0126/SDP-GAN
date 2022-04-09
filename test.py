import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from models import gan_net
from models import s_net
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default = './test_imgs', help='paths to load the test images')
parser.add_argument('--load_size', default = 450, help='size of test images')
parser.add_argument('--model_path', default = './pretrained_models', help='paths to load the pretrained models')
parser.add_argument('--g_model', default='generator_vangogh', help='name of the generator model')
parser.add_argument('--s_model', default='saliency', help='name of the saliency model')
parser.add_argument('--output_dir', default = 'test_outputs', help='paths to save the style results and the saliency results')
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--nb', type=int, default=8, help='the number of resnet block layer for generator')

opt = parser.parse_args()

valid_ext = ['.jpg', '.png']

if not os.path.exists(opt.output_dir): 
	os.mkdir(opt.output_dir)

# load pretrained generator model
g_model = gan_net.Generator(opt.in_ngc, opt.out_ngc, opt.ngf, opt.nb)
g_model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.g_model)))
g_model.eval()

# load pretrained saliency model
s_model = s_net.SaliencyNet(opt.in_ngc, opt.out_ngc, opt.ngf, opt.nb)
s_model.load_state_dict(torch.load(os.path.join(opt.model_path, opt.s_model)))
s_model.eval()

# pdb.set_trace()

if opt.gpu > -1:
	print('GPU mode')
	g_model.cuda()
	s_model.cuda()
else:
	print('CPU mode')
	g_model.float()
	s_model.float()

print('Start testing')
for files in os.listdir(opt.input_dir):
	ext = os.path.splitext(files)[1]
	if ext not in valid_ext:
		continue
	# load image
	input_image = Image.open(os.path.join(opt.input_dir, files)).convert("RGB")
	# resize image, keep aspect ratio
	h = input_image.size[0]
	w = input_image.size[1]
	ratio = h * 1.0 / w
	if ratio > 1:
		h = opt.load_size
		w = int(h * 1.0 / ratio)
	else:
		w = opt.load_size
		h = int(w * ratio)
	input_image = input_image.resize((h, w), Image.BICUBIC)
	input_image = np.asarray(input_image)
	# RGB -> BGR
	input_image = input_image[:, :, [2, 1, 0]]
	input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	# preprocess, (-1, 1)
	input_image = -1 + 2 * input_image 
	if opt.gpu > -1:
		input_image = Variable(input_image, volatile=True).cuda()
	else:
		input_image = Variable(input_image, volatile=True).float()
	# forward
	output_image = g_model(input_image)
	output_image = output_image[0]
	s_output_image = s_model(input_image)[3]
	s_output_image = s_output_image[0]
	# BGR -> RGB
	output_image = output_image[[2, 1, 0], :, :]
	s_output_image = s_output_image[[2, 1, 0], :, :]
	# deprocess, (0, 1)
	output_image = output_image.data.cpu().float() * 0.5 + 0.5
	s_output_image = s_output_image.data.cpu().float() * 0.5 + 0.5
	# save
	vutils.save_image(output_image, os.path.join(opt.output_dir, files[:-4] + '_style.jpg'))
	vutils.save_image(s_output_image, os.path.join(opt.output_dir, files[:-4] + '_saliency.jpg'))

print('Done')
