from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import sys
from Project.predict import predict_
from Project.GUI import GUI


class MainDialog(QDialog):
    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)
        self.ui = GUI.Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle("Cloth Image Prediction")
        self.setWindowIcon(QIcon('1.png'))

    def openImage(self):
        global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "choose picture", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QPixmap(imgName).scaled(self.ui.label_image.width(), self.ui.label_image.height())
        self.ui.label_image.setPixmap(jpg)
        self.img_path = imgName

    def run(self):
        # global fname
        # file_name = str(fname)
        # img = Image.open(self.img_path)

        a, b = predict_(self.img_path)
        self.ui.display_result.setText(a)
        self.ui.disply_acc.setText(str(b))

if __name__ == '__main__':
    myapp = QApplication(sys.argv)
    myDlg =MainDialog()
    myDlg.show()
    sys.exit(myapp.exec_())
