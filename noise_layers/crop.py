import torch
import torch.nn as nn
import numpy as np

def get_random_rectangle_inside(image_shape, height_ratio, width_ratio):
	image_height = image_shape[2]
	image_width = image_shape[3]

	remaining_height = int(height_ratio * image_height)
	remaining_width = int(width_ratio * image_width)

	if remaining_height == image_height:
		height_start = 0
	else:
		height_start = (64-remaining_height)//2

	if remaining_width == image_width:
		width_start = 0
	else:
		width_start = (64-remaining_width)//2

	return height_start, height_start + remaining_height, width_start, width_start + remaining_width


class Crop(nn.Module):

	def __init__(self, ratio):
		super(Crop, self).__init__()
		self.height_ratio = ratio
		self.width_ratio = ratio
		self.mask = None

	def forward(self, image_and_cover):
		image = image_and_cover
		if self.mask == None:
			h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																	 self.width_ratio)
			mask = torch.zeros_like(image)
			mask[:, :, h_start: h_end, w_start: w_end] = 1
			self.mask = mask
		return image * self.mask


class Cropout(nn.Module):

	def __init__(self, ratio):
		super(Cropout, self).__init__()
		self.height_ratio = ratio
		self.width_ratio = ratio
		self.mask = None

	def forward(self, image_and_cover):
		image = image_and_cover

		if self.mask == None:
			h_start, h_end, w_start, w_end = get_random_rectangle_inside(image.shape, self.height_ratio,
																		 self.width_ratio)
			mask = torch.ones_like(image)
			mask[:, :, h_start: h_end, w_start: w_end] = 0
			self.mask = mask
		return image * self.mask


class Dropout(nn.Module):

	def __init__(self, prob):
		super(Dropout, self).__init__()
		self.prob = prob
		self.mask = None
		self.rdn = None

	def forward(self, image_and_cover):
		image = image_and_cover
		if self.rdn == None:
			self.mask = torch.zeros_like(image)
			rdn = torch.rand(image.shape).to(image.device)
			self.rdn =rdn
		output = torch.where(self.rdn > self.prob * 1., image, self.mask)
		return output
