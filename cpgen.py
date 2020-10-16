from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty, StringProperty
from kivy.cache import Cache

import os
from kmeans import run_k_means
from upload_image import UploadedImage
from PIL import UnidentifiedImageError


class NewImageDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    cancel = ObjectProperty(None)

    # Default properties for widgets
    current_img_source = StringProperty('default.png')
    img_colours = StringProperty('no_colour.png')
    k_labels = StringProperty('')

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = NewImageDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="New Image", content=content, size_hint=(0.9,0.9))
        self._popup.open()

    def load(self, path, filename):
        """ Function for when an object is loaded through the file loader."""

        try:
            self.current_img_source = os.path.join(path, filename[0])
            self.current_image = UploadedImage(self.current_img_source)
            self.img_colours = 'no_colour.png'
            self.current_image.px = self.current_image.img_pixels()

        except UnidentifiedImageError or ValueError or IndexError:
            """This is for is a non-image is uploaded"""
            self.current_img_source = 'default.png'

        self.dismiss_popup()

    def calculate(self):
        """To map the k-means algorithm to the calculate button"""
        Cache.remove('kv.image')
        Cache.remove('kv.texture')
        self.img_colours = 'no_colour.png'

        try:
            k, num_points = run_k_means(self.current_image.px)
            self.current_image.colours = k
            self.current_image.create_colour_grad()
            self.img_colours = self.current_image.colours_path
            self.k_labels = f'Your RGB colour codes are: \n{k}'

        except AttributeError:
            pass


Factory.register('Root', cls=Root)
Factory.register('NewImageDialog', cls=NewImageDialog)

class CPGenApp(App):

     def build(self):
         pass


if __name__ == '__main__':
    CPGenApp().run()