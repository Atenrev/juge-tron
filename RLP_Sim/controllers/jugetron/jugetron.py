from torch._C import device
from robot import Sphero

controller = Sphero(yolo=True, device='0')  # use gpu
controller.run()
