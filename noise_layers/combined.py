from . import Identity
from . import Jpeg
from . import Jpeg_DCT
from . import JpegMask
from . import Jpeg_GET
import torch.nn as nn
from . import get_random_int


class Combined(nn.Module):

	def __init__(self, list=None):
		super(Combined, self).__init__()
		if list is None:
			list = [Jpeg(95),JpegMask(95)]
		self.list = list

	def forward(self, image_and_cover):
		id = get_random_int([0, len(self.list) - 1])
		return self.list[id](image_and_cover)
