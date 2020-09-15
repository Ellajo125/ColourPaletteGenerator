from PIL import Image as Im
import numpy as np


class UploadedImage:
    """A class to handle the uploaded images. """

    def __init__(self, image_path, type='rgb'):
        """Initializing information for the UploadedImage class"""
        self.image_path = image_path
        self.type = type


    def img_pixels(self):
        """ Function to turn an image into an array of pixels."""
        self.image = Im.open("test1.png")

        " Setting up a matrix to store the rgb values"
        num_pix = self.image.size[0]*self.image.size[1]
        px = np.zeros((num_pix, 3)) #This is currently only set up for rgb.

        for band in range(0, 2):
            px[:, band] = list(self.image.getdata(band))
        return px