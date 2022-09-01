from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.graphics import *
from kivy.core.window import Window

from screeninfo import get_monitors
# for m in get_monitors():
monitor = get_monitors()[0]
print(monitor.width,monitor.height)
print(Window.system_size)
# Window.size = (300, 200)
class MainWindow(BoxLayout):
    def __init__(self):
        super().__init__()
        self.button = Button(text="Hello, World?")
        self.button.bind(on_press=self.handle_button_clicked)
        self.add_widget(self.button)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        # self.canvas.before.clear()
        with self.canvas:
            Color(1, 0, 0)
            Rectangle(pos=(50,50), size=(600,600))
            Color(1, 1, 1)
            BorderImage(
                size=( 600,  600),
                pos=( 50,  50),
                # borders: (5, 'solid', (1,0,0,1)),
                source='./pictures/818_heatmap.png')
        
    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        print(keycode[1],text)
        # if keycode[1] == 'w':
        #     self.player1.center_y += 10
        # elif keycode[1] == 's':
        #     self.player1.center_y -= 10
        # elif keycode[1] == 'up':
        #     self.player2.center_y += 10
        # elif keycode[1] == 'down':
        #     self.player2.center_y -= 10
        return True

    def handle_button_clicked(self, event):
        self.button.text = "Hello, World!"


class MyApp(App):
    def build(self):
        self.title = "Hello, World!"
        return MainWindow()


app = MyApp()
app.run()