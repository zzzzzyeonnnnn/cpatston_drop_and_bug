import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50

def getPrediction(filename):
    #클래스 레이블 정의
    classes = ['담배가루이','상추균핵병','상추노균병','복숭아혹진딧물']
    #classes = ['Bemisia tabaci', 'Sclerotinia minor', 'Bremia lactucae', 'Myzus persicae']
    num_classes = len(classes)

    idx=0
    label = [0 for i in range(num_classes)]
    label[idx] = 1
    one_hot_label = np.array(label)


    #load model
    my_model=load_model("models/mobile_1.h5")

    SIZE = 224 #크기 정의
    img_path = 'static/images/' + filename #이미지 경우, 사용자가 업로드 하는 경로
    img = Image.open(img_path).resize((SIZE, SIZE)) #넘파이 배열로 반환
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) #치수를 오른쪽으로 확장
    img = resnet50.preprocess_input(img)

    pred = my_model.predict(img) #모델을 사용해서 이미지 진단 예측
    pred_num = np.argmax(pred)
    pred_class = classes[pred_num] #예측 결과를 역변환해서 클래스를 가져옴

    print('Predict is: ', pred_class) #에측 진단값
    return pred_class

#test = getPrediction('KakaoTalk_20231117_234018173.jpg')