import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Load a model from a .h5 model file
model = tf.keras.models.load_model('model/resnet.h5')
dir = 'data/zalando'
img_width, img_height = 224, 224

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255.,
                                                           validation_split = 0.2
                                                           )
test_data1 = test_gen.flow_from_directory(dir,
                                         target_size=(img_width,img_height),
                                         class_mode="categorical",
                                         classes=["hoodies", "hoodies-female", "longsleeve", "shirt", "sweatshirt",
                                                  "sweatshirt-female"],
                                         seed=42,
                                         subset="validation"
                                         )
# Load test set data
test_labels = test_data1.labels

# Make predictions on the test set
predictions = model.predict(test_data1)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Calculate the confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

classes=['hoodies','hoodies-female','longsleeve','shirt','sweatshirt','sweatshirt-female']

with tf.Session() as sess:
    conf = tf.confusion_matrix(test_labels, predicted_labels, num_classes=6)  # 计算混淆矩阵
    print(conf.eval())

    conf_numpy = conf.eval()
# plot confusion matrix
plt.figure(figsize=(7, 7))
plt.imshow(conf_numpy, cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks,classes, rotation=45)
plt.yticks(tick_marks,classes, rotation=45)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()
confusion_matrix_graph_path='checkpointer/confusion_matrix'
plt.savefig(confusion_matrix_graph_path)



