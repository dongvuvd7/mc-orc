from efficientnet_pytorch import EfficientNet #dòng này để
import torch
import torch.nn as nn #nn là neural network
import cv2
import matplotlib.pyplot as plt
import numpy as np
# def transform():
#     return A.Compose(
#         [
#             A.Resize(height=32, width=128,interpolation=cv2.INTER_AREA , p=1.0),
#             ToTensor(),
#         ]
#     )   

# Predictor_image là class để dự đoán ảnh
class Predictor_image(nn.Module): 
  # hàm khởi tạo, đầu vào là model_name, model_path, device
  def __init__(self,path_model="weights/cls_invoice.pth"): #path_model là đường dẫn đến tập huấn luyện
    super().__init__() #kế thừa từ class nn.Module
    self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2) #load model, 2 lớp ảnh ngược và ảnh xuôi
    self.model.load_state_dict(torch.load(path_model)) #load weight, load model 2 lớp ảnh ngược và ảnh xuôi

  #Hàm này để xử lý ảnh đầu vào
  def process_input(self,img): #img là ảnh đầu vào dưới dạng nhị phân
    img = cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA) #resize ảnh, đầu vào là: ảnh, kích thước ảnh đích (w, h), phương pháp resize (INTER_AREA là thu nhỏ), đầu ra: ảnh đích
    return torch.FloatTensor(img).permute(2,0,1).unsqueeze(0) #trả về ảnh dưới dạng tensor (tensor là kiểu biểu diễn dữ liệu nhiều chiều), FloatTensor: chuyển ảnh về dạng float, permute: chuyển đổi vị trí các chiều của tensor, unsqueeze: thêm một chiều vào tensor, đầu ra: ảnh đích với thức tự các chiều là [batch_size, channel, height, width] trong đó batch_size là số lượng ảnh, channel là số kênh màu, height là chiều cao, width là chiều rộng

  #Hàm này để dự đoán ảnh đầu vào
  def forward(self,img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) #chuyển ảnh về dạng RGB, astype(np.float32): chuyển ảnh về dạng float
    img /= 255.0 # img /= 255 để chuyển ảnh về dạng 0-1
    img = self.process_input(img) #xử lý ảnh đầu vào
    result = self.model(img) #self.model(img) là đưa ảnh vào model để dự đoán, đầu ra là 2 giá trị là xác suất ảnh ngược và ảnh xuôi, result có dạng tensor
    result = torch.softmax(result,-1) # chuyển đổi giá trị đầu ra thành xác suất, chuyển từ tensor sang numpy, softmax là hàm chuyển đổi giá trị thành xác suất
    return torch.argmax(result).item() #trả về giá trị dự đoán, argmax: trả về giá trị lớn nhất, item: trả về giá trị dự đoán

