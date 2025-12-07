import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage

class ImageClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crab Classification - CNN")
        self.setGeometry(100, 100, 400, 500)

        self.model = tf.keras.models.load_model("model.h5")
        self.img_path = None
        self.img_size = 128

        # ============================
        #  MAP ID → CLASS NAME
        # ============================
        self.class_names = [
            "Blue Crab",
            "Coconut Crab",
            "King Crab",
            "Mud Crab",
            "Stone Crab",
            "Vampire Crab"
        ]

        # Layout
        self.layout = QVBoxLayout()

        # Image preview
        self.image_label = QLabel("Chưa chọn ảnh")
        self.image_label.setFixedSize(350, 350)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setScaledContents(True)
        self.layout.addWidget(self.image_label)

        # Buttons
        self.btn_choose = QPushButton("Chọn ảnh")
        self.btn_choose.clicked.connect(self.choose_image)
        self.layout.addWidget(self.btn_choose)

        self.btn_predict = QPushButton("Dự đoán")
        self.btn_predict.clicked.connect(self.predict_image)
        self.layout.addWidget(self.btn_predict)

        # Output
        self.result_label = QLabel("Kết quả: ")
        self.layout.addWidget(self.result_label)

        self.setLayout(self.layout)

    def choose_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Image Files (*.jpg *.png *.jpeg)"
        )
        if file_path:
            self.img_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)

    def predict_image(self):
        if not self.img_path:
            self.result_label.setText("Chưa chọn ảnh!")
            return

        img = cv2.imread(self.img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

        img_input = img_resized.astype("float32") / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        # Predict
        pred = self.model.predict(img_input)
        class_id = np.argmax(pred)
        confidence = float(np.max(pred))

        class_name = self.class_names[class_id]

        self.result_label.setText(
            f"Kết quả: {class_name}  (Độ tin cậy: {confidence:.2f})"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifier()
    window.show()
    sys.exit(app.exec_())
