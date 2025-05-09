import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.crnn import CRNN
from utils.image_utils import preprocess_for_crnn, crop_image, save_cropped_image
from utils.text_utils import strLabelConverter

# ... existing code ... 