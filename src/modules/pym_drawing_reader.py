import numpy as np
from ipywidgets import Image
from ipywidgets import ColorPicker, IntSlider, link, AppLayout, HBox, Button, SliderStyle, FloatSlider, VBox, HTML, Layout
from ipycanvas import RoughCanvas, hold_canvas, Canvas
import tensorflow as tf
import cv2
from modules.pym_encoding import greedy_decoder

class PymDrawingReader:
    
    def __init__(self, model, width = 800, height = 250):
        canvas = Canvas(width=width, height=height, sync_image_data=True)

        drawing = False
        position = None
        shape = []

        def on_mouse_down(x, y):
            global drawing
            global position
            global shape

            drawing = True
            position = (x, y)
            shape = [position]

        def on_mouse_move(x, y):
            global drawing
            global position
            global shape

            if not drawing:
                return

            with hold_canvas():
                canvas.stroke_line(position[0], position[1], x, y)
                canvas.fill_circle(x, y, slider.value)

                position = (x, y)

            shape.append(position)

        def on_mouse_up(x, y):
            global drawing
            global position
            global shape

            drawing = False

            with hold_canvas(canvas):
                canvas.stroke_line(position[0], position[1], x, y)
                canvas.fill_circle(x, y, slider.value)


            shape = []

        def change_thickness(x):
            canvas.line_width = slider.value*2

        canvas.on_mouse_down(on_mouse_down)
        canvas.on_mouse_move(on_mouse_move)
        canvas.on_mouse_up(on_mouse_up)

        canvas.stroke_style = '#749cb8'

        picker = ColorPicker(description='Color:', value='#749cb8')
        link((picker, 'value'), (canvas, 'stroke_style'))
        link((picker, 'value'), (canvas, 'fill_style'))
        
        slider = FloatSlider(value=3, min=1, max=15, step=1, description='Width')
        slider.observe(change_thickness)

        clear_button = Button(description='clear')
        clear_button.on_click(self._on_clear_click)
        
        predict_button = Button(description='predict')
        predict_button.on_click(self._on_predict_click)
        
        layout = Layout(margin='auto 10px auto 35px')
        label = HTML(value = "Prediction: ", layout=layout)
        prediction = HTML(value="")
        
        change_thickness('')    
        
        self.model = model
        self.canvas = canvas
        self.prediction = prediction
        
        first_box = HBox((clear_button, picker,slider))
        second_box = HBox(( predict_button, label, prediction))        
        self.widget = VBox((self.canvas, first_box, second_box))

    def show_widget(self):
        return self.widget
    
    def _on_clear_click(self, change):
        self.canvas.clear()
        self.prediction.value = '';
    
    def _on_predict_click(self, change):
        img = self.canvas.get_image_data()
        img_resized = np.max(img[...,:3], axis=-1)
        img_resized = 1-cv2.resize(img_resized, (128,32))/255 # pixels values are inverted (white : 0, black :255) so we revert them back while putting values between [0;1]
        predicted_text = greedy_decoder(self.model(tf.expand_dims([img_resized], -1)))[0]
        self.prediction.value = "<b>" + predicted_text +"</b>"