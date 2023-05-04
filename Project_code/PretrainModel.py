import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.optimizers import Adam, SGD

dir = 'data/zalando'

img_width, img_height = 224, 224


train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255.,
                                                           rotation_range=0.2,
                                                           width_shift_range=0.2,
                                                           height_shift_range=0.2,
                                                           zoom_range = 0.2,
                                                           horizontal_flip=True,
                                                           validation_split = 0.02
                                                            )

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255.,
                                                           validation_split = 0.2
                                                           )



train_data = train_gen.flow_from_directory(dir,
                                           target_size=(img_width,img_height),
                                           class_mode="categorical",
                                           seed=42,
                                           classes=["hoodies", "hoodies-female", "longsleeve", "shirt", "sweatshirt",
                                                    "sweatshirt-female"],
                                           subset="training"
                                           )

test_data = test_gen.flow_from_directory(dir,
                                         target_size=(img_width,img_height),
                                         class_mode="categorical",
                                         classes=["hoodies", "hoodies-female", "longsleeve", "shirt", "sweatshirt",
                                                  "sweatshirt-female"],
                                         seed=42,
                                         subset="validation"
                                         )

labels = list(train_data.class_indices.keys())

base_model = tf.keras.applications.ResNet101V2(weights='imagenet',include_top= False)


inputs = tf.keras.Input(shape=(224,224,3))

x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(6, activation="softmax")(x)

# for layer in base_model.layers:
#     layer.trainable = False

res_model = tf.keras.Model(inputs, outputs)

res_model.compile(
    loss = tf.keras.losses.categorical_crossentropy,
    optimizer = Adam(learning_rate=0.0001),
    # optimizer = SGD(learning_rate=0.0001),
    metrics = ["accuracy"]
    )

history=res_model.fit(
    train_data,
    batch_size = 128,
    epochs = 50,
    steps_per_epoch = len(train_data),
    validation_data = test_data,
    validation_steps = len(test_data)
)

res_model.save('/Users/zhuhongyun/PycharmProjects/pythonProject1/Project/model/resnet.h5')

loss_trend_graph_path = "checkpointer/WW_loss.jpg"
acc_trend_graph_path = "checkpointer/WW_acc.jpg"
fig = plt.figure(1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(acc_trend_graph_path)
plt.close(1)
# summarize history for loss
fig = plt.figure(2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.savefig(loss_trend_graph_path)
plt.close(2)



