from nets.yolo3 import yolo_body
from keras.layers import Input
from yolo import YOLO
from PIL import Image

yolo = YOLO()

r_image = Image.open("./img/ttt.jpg")
r_image = yolo.detect_image(r_image)
r_image.show()
r_image.save("test.jpg",quality=95)
yolo.close_session()
