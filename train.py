import cv2
import copy
import numpy as np 
import os
import random
from sklearn import svm
from sklearn.externals import joblib
from sklearn import metrics

def cal_pixel(src, black_pixel = True):
    black = 0
    white = 0
    for i in range(src.shape[0]):
        for j in range(src.shape[1]): 
            if src[i,j] == 0:
                black += 1
            else:
                white += 1
    if (black_pixel):
        return black
    else:
        return white

def extract_feature(src):
    gray = copy.copy(src)
    if src.ndim == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        th, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    else:
        th, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    feature = []
    gray = cv2.resize(gray, (40, 40))
    h = int(gray.shape[0]/4)
    w = int(gray.shape[1]/4)
    S = cal_pixel(gray)

    for i in range(0, int(gray.shape[0]), w):
        for j in range(0, int(gray.shape[1]), h):
            roi = gray[i:i+h, j:j+w]
            s = cal_pixel(roi)
            f = float(s)/S
            feature.append(f)
    
    for i in range(0, 16, 4):
        f = feature[i] + feature[i+1] + feature[i+2] + feature[i+3]
        feature.append(f)

    for i in range(4):
        f = feature[i] + feature[i+4] + feature[i+8] + feature[i+12]
        feature.append(f)

    feature.append(feature[0] + feature[5] + feature[10] + feature[15])
    feature.append(feature[3] + feature[6] + feature[9] + feature[12])
    feature.append(feature[0] + feature[1] + feature[4] + feature[5])
    feature.append(feature[2] + feature[3] + feature[6] + feature[7])
    feature.append(feature[8] + feature[9] + feature[12] + feature[13])
    feature.append(feature[10] + feature[11] + feature[14] + feature[15])
    feature.append(feature[5] + feature[6] + feature[9] + feature[10])
    feature.append(feature[0] + feature[1] + feature[2] + feature[3] + feature[4] + feature[7] + feature[8] + feature[11] + feature[12] + feature[13] + feature[14] + feature[15])
    
    return feature

def load_data(data_dir):
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]
        for f in file_names:
            images.append(extract_feature(cv2.imread(f)))
            labels.append(int(d))
    return np.array(images), np.array(labels)

def save_model(model, name, db):
    if not os.path.exists(db):
        os.makedirs(db)
    os.chdir(db)
    file_name = name + ".joblib"
    print("[+] Saving model to file : " ,file_name)
    joblib.dump(model, file_name)

def linear_svm(X_train, y_train, X_test, y_test):
    print("[!] SVM data...")
    clf = svm.SVC(kernel='linear').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    print("[+] Finished")
    return clf

ROOT_PATH = "data"
PATH_WEIGHT = "weight"

train_data_dir = os.path.join(ROOT_PATH)
X_train, y_train = load_data(train_data_dir)
clf = linear_svm(X_train, y_train, X_train, y_train)
save_model(clf, "model", PATH_WEIGHT)

# print(images.shape)
# print(labels.shape)

# if __name__=="__main__":
    