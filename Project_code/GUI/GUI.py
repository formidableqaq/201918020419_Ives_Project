from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(673, 595)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        Dialog.setFont(font)

        # Set the button style
        self.selectImage_Btn = QtWidgets.QPushButton(Dialog)
        self.selectImage_Btn.setGeometry(QtCore.QRect(100, 290, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.selectImage_Btn.setFont(font)
        self.selectImage_Btn.setObjectName("selectImage_Btn")

        # Set the button style
        self.run_Btn = QtWidgets.QPushButton(Dialog)
        self.run_Btn.setGeometry(QtCore.QRect(440, 290, 131, 51))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.run_Btn.setFont(font)
        self.run_Btn.setObjectName("run_Btn")

        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(100, 380, 91, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(100, 440, 121, 31))
        self.label_2.setObjectName("label_2")
        self.display_result = QtWidgets.QLabel(Dialog)
        self.display_result.setGeometry(QtCore.QRect(240, 390, 221, 16))
        self.display_result.setText("")
        self.display_result.setObjectName("display_result")
        self.disply_acc = QtWidgets.QLabel(Dialog)
        self.disply_acc.setGeometry(QtCore.QRect(240, 450, 191, 16))
        self.disply_acc.setText("")
        self.disply_acc.setObjectName("disply_acc")
        self.gridLayoutWidget = QtWidgets.QWidget(Dialog)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(89, 10, 481, 251))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_image = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_image.setText("")
        self.label_image.setObjectName("label_image")
        self.gridLayout.addWidget(self.label_image, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.selectImage_Btn.clicked.connect(Dialog.openImage)
        self.run_Btn.clicked.connect(Dialog.run)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Image Classifier"))
        self.selectImage_Btn.setText(_translate("Dialog", "Choose picture"))
        self.run_Btn.setText(_translate("Dialog", "Run"))
        self.label.setText(_translate("Dialog", "Result："))
        self.label_2.setText(_translate("Dialog", "Accuracy："))
