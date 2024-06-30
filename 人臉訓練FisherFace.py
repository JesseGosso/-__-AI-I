import cv2
import os
import numpy as np
import glob

# 函數：儲存影像
def saveImg(image, index, name):
    filename = 'images/' + name + '/face{:03d}.jpg'.format(index)
    cv2.imwrite(filename, image)

# 初始化參數
index = 1
total = 1000

# 取得使用者名稱
name = input('Enter your name (in English): ')
if not os.path.exists('images'):
    os.mkdir('images')

if os.path.isdir('images/' + name):
    print("This name already exists")
else:
    os.mkdir('images/' + name)
    casc_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(casc_path)

    # 開啟預設相機（通常索引為0）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
    else:
        while index > 0:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                image = cv2.resize(gray[y:y + h, x:x + w], (400, 400))
                saveImg(image, index, name)
                index += 1
                if index > total:
                    print('Sampling completed')
                    index = -1
                    break
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# 訓練 Fisherfaces 模型
images = []
labels = []
labelstr = []
count = 0

dirs = os.listdir("images")
for d in dirs:
    files = glob.glob('images/' + d + '/*.jpg')
    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        
        # 確保影像資料是連續的並調整大小為200x200
        if img is not None:
            img = cv2.resize(img, (200, 200))
            img = np.ascontiguousarray(img)
            images.append(img)
            labels.append(count)
    labelstr.append(d)
    count += 1

with open('member.txt', 'w') as f:
    f.write(','.join(labelstr))

print('開始建立模型...')
model = cv2.face.FisherFaceRecognizer_create()
model.train(np.asarray(images), np.asarray(labels))
model.save('face_Fisher.yml')
print('建立模型完成')
