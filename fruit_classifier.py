import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget,
    QHBoxLayout, QGraphicsDropShadowEffect, QFrame
)
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtCore import Qt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Tên lớp và ánh xạ sang tiếng Việt
class_names = ['freshapples', 'freshbanana', 'freshoranges', 'rottenapples', 'rottenbanana', 'rottenoranges']
class_names_vi = {
    'freshapples': 'Táo tươi',
    'freshbanana': 'Chuối tươi',
    'freshoranges': 'Cam tươi',
    'rottenapples': 'Táo hỏng',
    'rottenbanana': 'Chuối hỏng',
    'rottenoranges': 'Cam hỏng'
}

# Load model
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(torch.load('fruit_classifier_resnet18.pth', map_location='cpu'))
model.eval()

# Transform giống lúc test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FruitClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Phân Loại Hoa Quả | Version 1.0")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: #f6f8fa;")

        # ======= Tiêu đề lớn và mô tả nhỏ =======
        self.header_title = QLabel("PHÂN LOẠI HOA QUẢ BẰNG AI", self)
        self.header_title.setAlignment(Qt.AlignCenter)
        self.header_title.setStyleSheet("""
            QLabel {
                font-size: 32px;
                font-weight: bold;
                color: white;
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38b6ff, stop:1 #4fdc94);
                padding: 18px 0 0 0;
            }
        """)

        self.header_desc = QLabel("Upload ảnh hoa quả để nhận diện bằng trí tuệ nhân tạo", self)
        self.header_desc.setAlignment(Qt.AlignCenter)
        self.header_desc.setStyleSheet("""
            QLabel {
                font-size: 16px;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38b6ff, stop:1 #4fdc94);
                border-bottom-left-radius: 20px;
                border-bottom-right-radius: 20px;
                padding-bottom: 12px;
            }
        """)

        # ======= Footer trạng thái =======
        self.footer = QLabel("Mô hình đã được tải thành công!", self)
        self.footer.setStyleSheet("background-color: #1b3556; color: #fff; font-size: 16px; padding: 6px;")
        self.footer.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.footer.setFixedHeight(32)

        # ======= Ảnh hoa quả =======
        self.image_label = QLabel("Ảnh Hoa Quả", self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(320, 240)
        self.image_label.setStyleSheet("""
            QLabel {
                background: #fff;
                border-radius: 16px;
                border: 1px solid #eee;
                font-size: 16px;
            }
        """)
        self.add_shadow(self.image_label)

        # ======= Nút tải ảnh lên =======
        self.btn_load = QPushButton("Tải Ảnh Lên", self)
        self.btn_load.setFixedWidth(150)
        self.btn_load.setStyleSheet("""
            QPushButton {
                background: #38b6ff;
                color: #fff;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background: #4fdc94;
            }
        """)
        self.btn_load.clicked.connect(self.load_image)
        self.add_shadow(self.btn_load)

        # ======= Nút xóa ảnh =======
        self.btn_clear = QPushButton("Xóa", self)
        self.btn_clear.setFixedWidth(100)
        self.btn_clear.setStyleSheet("""
            QPushButton {
                background: #b0b0b0;
                color: #fff;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background: #ff5c5c;
            }
        """)
        self.btn_clear.clicked.connect(self.clear_image)
        self.add_shadow(self.btn_clear)

        # ======= Layout cho nút, căn giữa dưới ảnh =======
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_load)
        btn_layout.addSpacing(20)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addStretch()

        # ======= Layout trái (ảnh + nút) =======
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Ảnh Hoa Quả:"))
        left_layout.addWidget(self.image_label)
        left_layout.addSpacing(10)
        left_layout.addLayout(btn_layout)
        left_layout.addStretch()

        # ======= Kết quả phân tích =======
        self.result_label = QLabel("Chưa có kết quả", self)
        self.result_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #38b6ff;")
        self.confidence_label = QLabel("", self)
        self.confidence_label.setStyleSheet("font-size: 18px; color: #222;")

        # ======= Biểu đồ xác suất =======
        self.figure, self.ax = plt.subplots(figsize=(4, 2))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background: transparent;")
        self.figure.patch.set_alpha(0.0)

        # ======= Layout phải (kết quả + biểu đồ) =======
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Kết quả phân tích:"))
        right_layout.addWidget(self.result_label)
        right_layout.addWidget(self.confidence_label)
        right_layout.addWidget(self.canvas)
        right_layout.addStretch()

        # ======= Khung chứa 2 panel =======
        panel_layout = QHBoxLayout()
        left_panel = QFrame()
        left_panel.setStyleSheet("background: #fff; border-radius: 18px;")
        left_panel.setLayout(left_layout)
        right_panel = QFrame()
        right_panel.setStyleSheet("background: #fff; border-radius: 18px;")
        right_panel.setLayout(right_layout)
        panel_layout.addWidget(left_panel)
        panel_layout.addWidget(right_panel)

        # ======= Layout tiêu đề =======
        header_layout = QVBoxLayout()
        header_layout.addWidget(self.header_title)
        header_layout.addWidget(self.header_desc)

        # ======= Layout tổng =======
        main_layout = QVBoxLayout()
        main_layout.addLayout(header_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(panel_layout)
        main_layout.addStretch()
        main_layout.addWidget(self.footer)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.clear_image()

    def add_shadow(self, widget):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(2)
        shadow.setColor(QColor(0, 0, 0, 60))
        widget.setGraphicsEffect(shadow)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg *.webp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.classify_image(file_name)

    def clear_image(self):
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("Ảnh Hoa Quả")
        self.result_label.setText("Chưa có kết quả")
        self.confidence_label.setText("")
        self.ax.clear()
        self.canvas.draw()

    def classify_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy()[0]
            _, preds = torch.max(outputs, 1)
            class_en = class_names[preds.item()]
            class_vi = class_names_vi[class_en]
            confidence = probabilities[preds.item()] * 100

        self.result_label.setText(f"{class_vi}")
        self.confidence_label.setText(f"Độ tin cậy: {confidence:.2f}%")

        # Vẽ biểu đồ xác suất
        self.ax.clear()
        y_pos = np.arange(len(class_names))
        colors = ['#b0e0e6' if i != preds.item() else '#38b6ff' for i in range(len(class_names))]
        self.ax.barh(y_pos, probabilities * 100, align='center', color=colors)
        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels([class_names_vi[c] for c in class_names], fontsize=12)
        self.ax.invert_yaxis()
        self.ax.set_xlabel('Xác suất (%)', fontsize=12)
        self.ax.set_xlim(0, 100)
        self.ax.set_title('')
        for i, v in enumerate(probabilities * 100):
            self.ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=10)
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FruitClassifierGUI()
    window.show()
    sys.exit(app.exec_())
