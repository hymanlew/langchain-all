'''
目标检测，是在图像中找出检测对象的位置和大小，在自动驾驶，机器人和无人机方面极具研究价值。
使用 MobileNet-SSD 进行目标检测，它可以检测飞机、自行车、人、沙发等20种物体课1种背景

pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python

第一步：加载 MobileNet-SSD 目标检测网络模型
第二步：读入待检测图像，并将其转化为 blob 数据包
第三步：将 blob 数据包传入目标检测网络，并进行前向传播
第四步：根据返回结果标注图像中被检测出的对象
'''
import cv2

class ImageOcr:
	def __init__(self):
	
	
		
		
		