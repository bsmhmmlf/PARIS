import torch.nn as nn
import torch
from kornia import geometry


class RS(nn.Module):

	def __init__(self, size):
		super(RS, self).__init__()
		self.size = size

	def resize(self, image):
		temp = geometry.transform.resize(image,(self.size,self.size))
		output = geometry.transform.resize(temp,(64,64))
		return output

	def forward(self, image_and_cover):
		image = image_and_cover
		return self.resize(image)

class RT(nn.Module):
	def __init__(self, angle):
		super(RT, self).__init__()
		self.angle = torch.tensor(angle,dtype=torch.float32).cuda()

	def rotation(self, image):
		output = geometry.transform.rotate(image,self.angle)
		return output

	def forward(self, image_and_cover):
		image = image_and_cover
		return self.rotation(image)

class ES(nn.Module):
	def __init__(self,sigma):
		super(ES, self).__init__()
		self.sigma = sigma

	def elastic(self, image):
		noise = torch.rand(1, 2, 64, 64).cuda()
		output = geometry.transform.elastic_transform2d(image, noise,(self.sigma,self.sigma))
		return output

	def forward(self, image_and_cover):
		image = image_and_cover
		return self.elastic(image)