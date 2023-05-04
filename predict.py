import tensorflow as tf
import numpy as np
import cv2

def predict_(img_path):
    model_path = './model/resnet.h5'
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    classes = ["hoodies", "hoodies-female", "longsleeve", "shirt", "sweatshirt","sweatshirt-female"]

    prediction = model.predict(img)
    predict_cla = np.argmax(prediction, axis=1)[0]
    return classes[predict_cla], prediction[0][predict_cla]

if __name__ == '__main__':
    img_path = '/Users/zhuhongyun/PycharmProjects/pythonProject1/Project/data/zalando/sweatshirt/0DB22S004-C11@8.jpg'
    net = predict_(img_path)
    print(net)
