# 作者 ljc
import cv2
import dlib
import numpy
import sys

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
# 代表各个区域的关键点标号
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


# 获取关键点坐标位置，只获取一张人脸
# input：代表一张图片的numpy array
# output：68*2的关键点坐标位置matrix
def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s


# 注解关键点
def annotate_landmarks(im, landmarks):
    # 数组切片是原始数组的视图，这意味着数据不会被复制，视图上的任何修改都会被直接反映到源数组上.
    # 若想要得到的是ndarray切片的一份副本而非视图，就需要显式的进行复制操作函数copy()。
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.2,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 1, color=(0, 255, 255))
        cv2.imwrite("landmak.jpg", im)
    return im


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)  # 检测凸包函数
    cv2.fillConvexPoly(im, points, color=color)  # 绘制好多边形后并填充     点的顺序不同绘制出来的凸包也不同


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    # for group in OVERLAY_POINTS:
    #     draw_convex_hull(im,landmarks[group],color=1)

    # 11. 下面这行代码用来替代上面两行代码
    draw_convex_hull(im, landmarks, color=1)
    im = numpy.array([im, im, im]).transpose((1, 2, 0))  # 得到一个类似于3通道的图片

    # 22. 高斯滤波，注释掉效果更好
    # im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    # im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    return im


# 用普氏分析(Procrustes analysis)调整脸部
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:返回一个仿射变换矩阵

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # 通过减去中心id，通过标准偏差进行缩放，然后使用SVD来计算旋转，从而解决了普是问题
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    # 计算标准差
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    # 通过奇异值分解求得旋转矩阵R
    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T  # 维度:2*2
    # 仿射变换矩阵3*3 #  numpy.hstack用来在第1个维度上拼接tup  numpy.vstack在第0个维度上拼接tup
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    # cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue ]]]])-->dst
    cv2.warpAffine(im, M[:2], (dshape[1], dshape[0]), dst=output_im, borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


# 颜色校正
def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) / im2_blur.astype(numpy.float64))


im1, landmarks1 = read_im_and_landmarks("111.jpg")
im2, landmarks2 = read_im_and_landmarks("333.png")
# 44. 参数landmarks1[ALIGN_POINTS]-->landmarks1
M = transformation_from_points(landmarks1, landmarks2)  # [ALIGN_POINTS]

# get_face_mask()的定义是为一张图像和一个标记矩阵生成一个掩膜
mask = get_face_mask(im2, landmarks2)
warped_mask = warp_im(mask, M, im1.shape)
# 33. 用min函数取掩膜区域效果更好
combined_mask = numpy.min([get_face_mask(im1, landmarks1), warped_mask], axis=0)
# 将图像2的掩膜转换到图像1的坐标空间
warped_im2 = warp_im(im2, M, im1.shape)
warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
cv2.imwrite('output.jpg', output_im)