from PIL import Image as Im
import numpy as np


class UploadedImage:
    """A class to handle the uploaded images. """

    def __init__(self, image_path, type='rgb'):
        """Initializing information for the UploadedImage class"""
        self.source = str(image_path)
        self.type = type


    def img_pixels(self, ideal_size =(480,270)):
        """ Function to turn an image into an array of pixels."""

        with Im.open(self.source) as img:
            " Setting up a matrix to store the rgb values"
            ratio = max(img.size[0]/ideal_size[0], img.size[1]/ideal_size[1])
            img_resized = img.resize((int(img.width//ratio), int(img.height//ratio)))
            num_pix = img_resized.size[0]*img_resized.size[1]
            print(num_pix)
            data = np.asarray(img_resized)
            px = data.reshape(img_resized.size[0]*img_resized.size[1], 3)
        return px