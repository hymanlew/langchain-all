'''
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-contrib-python


'''
import cv2

class ImageOcr:
	def __init__(self):
	
	def putText(img_path: str):
		#从文件中读取图像（numpy数组）
		img = cv2.imread(img_path)

		#设置文件的位置、字体、字体大小、字体粗细、颜色等参数
		pos = (10, 50)
		font_style = cv2.FONT_HERSHEY_SIMPLEX
		font_size = 2
		font_width = 2
		color = (255, 0, 0)

		#在图像中显示文本 "hello, world"
		cv2.putText(img, 'hello, world', pos, font, font_size, color, font_width)

		#显示图像窗口
		cv2.imshow('Image', img)

		#按任意键退出
		cv2.waitKey(0)

		#销毁所有窗口
		cv2.destroyAllWindows()


	# 人脸检测就是从一个图像中寻找出人脸所在的位置和大小
	def human_img_ocr(img_path: str):
		#从文件读取图像并转为灰度图像（人脸检测通常在灰度图上进行）
		img = cv2.imread(img_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		#创建人脸检测器，Haar级联分类器模型文件（是OpenCV提供的传统人脸检测方法），并加载这个文件
		file = 'file/haarcascade_frontalface_default.xml'
		face_cascade = cv2.CascadeClassifier(file)
		
		#检测人脸区域
		'''
		#第1个参数表示灰度图像，
		#第2个参数表示比例因子，是类似于从远到近观察图片，每次移动时把模板放大指定倍数（这里是 1.3 放大 30%）。每一级都用相同大小的窗口扫描，相当于检测不同大小的人脸
		#第3个参数，防止误检现象：在真实人脸周围，算法可能会产生多个重叠的检测框。
			5 表示只有当此区域至少有 5 个相互重叠的检测框时，才确认为人脸。
			值越大，检测要求越严格，结果越可靠但可能漏检
			默认为 3
		'''
		faces = face_cascade.detectMultiScale(gray, 1.3, 5)
		
		#标注人脸区域，faces 是检测到的人脸矩形列表，每个元素是(x, y, w, h)格式的元组
		for (x, y, w, h) in faces:
			'''
			img: 要绘制矩形的图像
			(x, y): 矩形左上角坐标
			(x+w, y+h): 矩形右下角坐标
			(255, 0, 0): 矩形颜色（BGR格式，这里是纯蓝色）
			3: 矩形边框的线宽
			'''
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
			
		#显示检测结果到窗口
		cv2.imshow('Image', img)

		#按任意键退出
		cv2.waitKey(0)

		#销毁所有窗口
		cv2.destroyAllWindows()
		
		
	def human_video_ocr(video_path: str):
		#创建人脸检测器
		file = 'file/haarcascade_frontalface_default.xml'
		face_cascade = cv2.CascadeClassifier(file)

		#加载视频文件，创建视频捕获对象
		if video_path:
			vc = cv2.VideoCapture(video_path)
		else:
			#打开摄像头，设置画面大小
			vc = cv2.VideoCapture(0)
			vc.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
			vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
		
		#处理视频流
		while True:
			#读取视频帧，是一帧帧的读取
			'''
			retval：布尔值，表示是否成功读取帧（True=成功，False=失败/视频结束）
			frame：读取到的视频帧（numpy 数组，BGR 格式的图像数据）
			'''
			retval, frame = vc.read()
			
			#waitKey(16)：等待 16 毫秒（约等于 60FPS），并返回按键值
			#按Q键退出
			if not retval or cv2.waitKey(16) & 0xFF == ord('q'):
				break
			
			#将彩色帧转换为灰度图像
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			#检测人脸区域，同图片检测逻辑
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			
			#标注人脸区域，同图片检测逻辑
			for (x, y, w, h) in faces:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
				
				'''
				#检测眼睛区域，在原图的基础上缩小检测范围
				eye_cascade = cv2.CascadeClassifier('file/haarcascade_eye.xml')
				roi_gray = gray[y:y+h, x:x+w]
				eyes = eye_cascade.detectMultiScale(roi_gray)

				#检测眼睛区域
				roi_color = face_img[y:y+h, x:x+w]
				for (ex,ey,ew,eh) in eyes:
					cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
				'''
			#显示视频帧到窗口
			cv2.imshow('Video', frame)

		#关闭视频
		vc.release()

		#销毁所有窗口
		cv2.destroyAllWindows()
	
	
	def car_number(img_path: str):
		img = cv2.imread(img_path)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		
		#创建车牌检测器
		file = 'haarcascade_russian_plate_number.xml'
		face_cascade = cv2.CascadeClassifier(file)
		faces = face_cascade.detectMultiScale(img, 1.2, 5)
		
		for (x, y, w, h) in faces:
			#标注车牌区域
			cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)
			#将车牌区域的图像写入文件
			number_img = img[y:y+h, x:x+w]
			cv2.imwrite('file/images/car_number.jpg', number_img)
			
		cv2.imshow('Image', img) #显示检测结果到窗口
		cv2.waitKey(0) #按任意键退出
		cv2.destroyAllWindows() #销毁所有窗口

		
		
		