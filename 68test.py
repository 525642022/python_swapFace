# 作者 ljc
# 68-points
# 2017-12-28
# By TimeStamp
# #cnblogs: http://www.cnblogs.com/AdaminXie/
import dlib                     #人脸识别的库dlib
import numpy as np              #数据处理的库numpy
import cv2                      #图像处理的库OpenCv

# dlib预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# cv2读取图像
img=cv2.imread("NEW.jpg")

# 取灰度
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 人脸数rects
rects = detector(img_gray, 0)

for i in range(len(rects)):
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img, rects[i]).parts()])

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])

        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 5, color=(0, 255, 0))

        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(img, str(idx)+str(), pos, font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

cv2.namedWindow("img", 2)
cv2.imshow("img", img)
# cv2.imwrite("lytest1.jpg",img)//
cv2.waitKey(0)