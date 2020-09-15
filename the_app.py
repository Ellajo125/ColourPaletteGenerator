from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.uix.widget import Widget
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.properties import  ObjectProperty, StringProperty

import os
from kmeans import run_kmeans

from upload_image import UploadedImage


class NewImageDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout):
    cancel = ObjectProperty(None)
    current_image = ObjectProperty(None)
    img_path = ObjectProperty(None)

    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = NewImageDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="New Image", content=content, size_hint=(0.9,0.9))
        self._popup.open()

    def load(self, path, filename):
        img_path = filename[0]
        self.current_image = UploadedImage(img_path)
        self.dismiss_popup()

    def calculate(self):
        """To map the k-means algorithm to the calculate button"""
        self.current_image.px = self.current_image.img_pixels()
        print(self.current_image.px)
        k, err_list, same_k= run_kmeans(3, self.current_image.px)

class ImageSpot(Widget):
    """Widget to define the spot the holds the images on the app."""
    source = StringProperty(None)


class CPGenApp(App):

     def build(self):
         pass

Factory.register('Root', cls=Root)
Factory.register('NewImageDialog', cls=NewImageDialog)

if __name__ == '__main__':
    CPGenApp().run()