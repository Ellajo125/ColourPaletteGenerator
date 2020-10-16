from PIL import Image as Im
import numpy as np


class UploadedImage:
    """A class to handle the uploaded image in colour palette generator. """

    def __init__(self, image_path):
        """Initializing information for the UploadedImage class"""
        self.source = str(image_path)
        self.colours = np.zeros((1,3))
        self.colours_path = 'colourgrad.png'

    def img_pixels(self, ideal_size=(480, 270)):
        """ Function to turn an image into an array of pixels."""

        with Im.open(self.source) as img:
            # Creating a matrix of RGB values with each row consisting of a pixel in the image. Images are resized
            # to allow for quicker processing.
            ratio = max(img.size[0]/ideal_size[0], img.size[1]/ideal_size[1])
            img_resized = img.resize((int(img.width//ratio), int(img.height//ratio)))

        data = np.asarray(img_resized)
        self.size = data.shape
        px = data.reshape(img_resized.size[0]*img_resized.size[1], 3)
        return px

    def create_colour_grad(self):
        """Function to create a colour gradient of values found in k_means"""

        square_size = (int(self.size[1]/5), int(self.size[0]/self.colours.shape[0]))
        colour_a = np.zeros((square_size[0],square_size[1]*self.colours.shape[0],3))
        i = 0

        # Making a pretty square box image
        for colour in self.colours:
            c_start = (i*square_size[1])
            c_end = ((i*square_size[1])+square_size[1]-1)
            colour_a[:, c_start:c_end, :] = colour
            i += 1

        # Creating the image and saving it to the colour grad image path
        self.c_grad = Im.fromarray(colour_a.astype('uint8'), 'RGB')
        self.c_grad.save(self.colours_path)


