from PyQt5 import QtWidgets, QtGui
import sys
import cv2
import numpy as np

class AppWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth Map Enhancer")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(10, 10, 600, 400)

        self.load_button = QtWidgets.QPushButton("Load Image", self)
        self.load_button.setGeometry(620, 10, 150, 30)
        self.load_button.clicked.connect(self.load_image)

        self.range_label = QtWidgets.QLabel("Select Distance Range:", self)
        self.range_label.setGeometry(620, 50, 150, 30)

        self.min_distance = QtWidgets.QSpinBox(self)
        self.min_distance.setGeometry(620, 90, 70, 30)
        self.min_distance.setRange(0, 255)

        self.max_distance = QtWidgets.QSpinBox(self)
        self.max_distance.setGeometry(700, 90, 70, 30)
        self.max_distance.setRange(0, 255)

        self.enhance_button = QtWidgets.QPushButton("Enhance Contrast", self)
        self.enhance_button.setGeometry(620, 130, 150, 30)
        self.enhance_button.clicked.connect(self.enhance_contrast)

        self.image = None

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if file_name:
            self.image = cv2.imread(file_name)
            self.display_image(self.image)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(q_img))

    def enhance_contrast(self):
        if self.image is not None:
            min_val = self.min_distance.value()
            max_val = self.max_distance.value()
            depth_map = self.compute_depth_map(self.image)
            enhanced_map = self.apply_contrast(depth_map, min_val, max_val)
            self.display_image(enhanced_map)

    def compute_depth_map(self, img):
        # Placeholder for depth map computation
        # In a real application, this would call the depth generator
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Simulating a depth map

    def apply_contrast(self, depth_map, min_val, max_val):
        depth_map = np.clip(depth_map, min_val, max_val)
        depth_map = (depth_map - min_val) / (max_val - min_val) * 255
        return depth_map.astype(np.uint8)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = AppWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()