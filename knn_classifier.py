# USAGE
# python knn_classifier.py --dataset firearm

# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
    #＃將圖像調整為固定大小，然後將圖像壓平
    #原始像素強度列表
	return cv2.resize(image, size).flatten()

def extract_color_histogram(image, bins=(8, 8, 8)):
    #使用HSV顏色空間提取3D顏色直方圖
    #每個頻道提供的“bins”數量
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])

    #處理規範化直方圖如果我們使用的是OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)

    #否則，在OpenCV 3中進行“就地”歸化   
	else:
		cv2.normalize(hist, hist)

    #將展平的直方圖作為特徵向量返回
	return hist.flatten()

#構造參數解析並解析參數
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

#抓住我們將要描述的圖像列表
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))

#初始化原始像素強度矩陣，特徵矩陣，和標籤列表
rawImages = []
features = []
labels = []

#循環輸入圖像
for (i, imagePath) in enumerate(imagePaths):
    #加載圖像並提取類標籤(假設我們的路徑為格式:/path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]

    #提取原始像素強度“特徵”，然後是顏色直方圖，以表徵圖像中像素的顏色分佈
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)

    #分別更新原始圖像，功能和標籤
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)

    #每1000張圖片顯示一次更新
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

#顯示原始圖像消耗的內存的一些信息
#矩陣和特徵矩陣
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))

#使用75％將數據劃分為訓練和測試分割
#其餘25％的訓練數據，並進行測試
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)

#在原始像素強度上訓練和評估k-NN分類器
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

#在直方圖表示上訓練和評估k-NN分類器
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
	n_jobs=args["jobs"])
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))