import random


def get_random_float(float_range: [float]):
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
	return random.randint(int_range[0], int_range[1])


from .identity import Identity
from .jpeg import Jpeg, JpegSS, JpegMask, JpegTest, Jpeg_DCT,Jpeg_GET,Jpeg_255, JpegSS_255, JpegMask_255
from .combined import Combined
from .gaussian_noise import GN
from .middle_filter import MF
from .gaussian_filter import GF
from .salt_pepper_noise import SP
from .jpeg import Jpeg, JpegSS, JpegMask, JpegTest
from .combined import Combined
from .resize import RS,RT,ES
from .crop import Crop,Cropout,Dropout