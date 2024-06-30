import cv2
import os
model_path = "C:/Users/boltc/OneDrive/桌面/face_LBPH1.yml"
if os.path.exists(model_path):
    print(f"Model file exists at: {model_path}")
else:
    print(f"Model file not found at: {model_path}")

# 嘗試載入模型
me = cv2.face.LBPHFaceRecognizer_create()


# 加載要讀取的團體照
img_path = "C:/Users/boltc/Downloads/IMG_0472.jpg"


img = cv2.imread(img_path)
img = cv2.resize(img, (800, 600))

# 將影像轉換為灰階以進行人臉偵測
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 載入預訓練的 Haar Cascade 進行人臉偵測
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 偵測灰階影像中的人臉
faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 在偵測到的臉部上繪製黑色矩形
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), -1)

# 顯示處理過的圖像
cv2.imshow("si2", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
