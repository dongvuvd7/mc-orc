import re
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
from utils import rotate_box,align_box,get_idx
from modules.text_detect.predict import test_net,net,refine_net,poly
from modules.image_clf.predict import Predictor_image
from PIL import Image
# from modules.text_recognition.predict import text_recognizer
from modules.text_clf.svm import predict_svm
from modules.text_clf.phoBert import predict_phoBert
from modules.text_clf.regex import date_finder
import torch

from modules.image_segmentation.predict import  segment_single_images

import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import joblib

def test(image):
    image = segment_single_images(image) #sử dụng model image segmentation để cắt nền ảnh
    image_copy = image.copy() #copy ảnh để nhận dạng ra các bounding box, tính toán độ lệch để xác định ảnh nằm ngang hay ngược rồi xoay về thẳng
    bboxes, polys, score_text = test_net(net, image_copy, 0.7, 0.4, 0.4, True, poly, refine_net) #tiến hành xác định chữ (text detection), bboxes là tọa độ các khung chứa text, polys là tọa độ các đường viền của các khung chứa text, score_text là ảnh heatmap(?chỗ này đặt tên score_text không hợp lí), tham số thứ 3,4,5 là ngưỡng của score_text, score_link, low_text, tham số thứ 6 là cuda, tham số thứ 7 là poly, tham số thứ 8 là refine_net
    if bboxes!=[]: #Nếu có các khung chứa text
        
        bboxes_xxyy = [] #tọa độ các khung chứa text theo dạng xmin, xmax, ymin, ymax trong đó xmin, ymin là tọa độ góc duới bên trái, xmax, ymax là tọa độ góc trên bên phải
        ratios = [] #tỉ lệ chiều dài chiều rộng của các khung chứa text
        degrees = [] #góc xoay của các khung chứa text
        for box in bboxes:#duyệt qua các khung chứa text
            x_min = min(box, key=lambda x: x[0])[0] #tìm tọa độ x nhỏ nhất của khung chứa text, key=lambda x: x[0] là tìm tọa độ x của các điểm trong khung chứa text
            x_max = max(box, key=lambda x: x[0])[0] #tìm tọa độ x lớn nhất của khung chứa text
            y_min = min(box, key=lambda x: x[1])[1] #tìm tọa độ y nhỏ nhất của khung chứa text, key=lambda x: x[1] là tìm tọa độ y của các điểm trong khung chứa text
            y_max = max(box, key=lambda x: x[1])[1] #tìm tọa độ y lớn nhất của khung chứa text
            if (x_max-x_min) > 20: #nếu chiều dài khung chứa text lớn hơn 20 thì mới tính toán
                ratio = (y_max-y_min)/(x_max-x_min) #tính tỉ lệ chiều rông / chiều dài của khung chứa text
                ratios.append(ratio) #thêm tỉ lệ vào list ratios
            

        mean_ratio = np.mean(ratios) #tính trung bình cộng của các tỉ lệ trong list ratios
        if mean_ratio>=1:   #nếu trung bình cộng lớn hơn 1 thì là text dọc (ảnh nằm ngang)
            image,bboxes = rotate_box(image,bboxes,None,True,False)  
        
        if predict_image(image) == 1: #sau phân loại ảnh, nếu ảnh ngược thì xoay ảnh 180 độ
            image,bboxes = rotate_box(image,bboxes,None,False,True) 

        bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, True, poly, refine_net) #tiến hành xác định chữ (text detection), thu được các khung chứa text, các khung chứa chữ liên kết với nhau, ảnh heatmap

        image,check = align_box(image,bboxes,skew_threshold=0.9) #tiến hành căn chỉnh ảnh nếu ảnh bị lệch, check là biến kiểm tra xem có căn chỉnh được không, skew_threshold là ngưỡng của độ lệch của ảnh, lấy ngưỡng = 0,9 radian = 51,5 độ (nếu độ lệch > 51,5 độ thì căn chỉnh ảnh)
        #Tới đây là xong bước tiền xử lý

        if check:
            bboxes, polys, score_text = test_net(net, image, 0.7, 0.4, 0.4, True, poly, refine_net)
        h,w,c = image.shape #lấy chiều cao, chiều rộng, số kênh màu của ảnh

        for box in bboxes: #duyệt qua các khung chứa text
            x_min = max(int(min(box, key=lambda x: x[0])[0]),1) #tìm tọa độ x nhỏ nhất của khung chứa text int(min(box, key=lambda x: x[0])[0]),1 là tìm tọa độ x của các điểm trong khung chứa text, sau đó ép kiểu về int, nếu tọa độ x nhỏ nhất nhỏ hơn 1 thì gán bằng 1
            x_max = min(int(max(box, key=lambda x: x[0])[0]),w-1) #tìm tọa độ x lớn nhất của khung chứa text int(max(box, key=lambda x: x[0])[0]),w-1 là tìm tọa độ x của các điểm trong khung chứa text, sau đó ép kiểu về int, nếu tọa độ x lớn nhất lớn hơn w-1 thì gán bằng w-1
            y_min = max(int(min(box, key=lambda x: x[1])[1]),3) #tìm tọa độ y nhỏ nhất của khung chứa text int(min(box, key=lambda x: x[1])[1]),3 là tìm tọa độ y của các điểm trong khung chứa text, sau đó ép kiểu về int, nếu tọa độ y nhỏ nhất nhỏ hơn 3 thì gán bằng 3
            y_max = min(int(max(box, key=lambda x: x[1])[1]),h-2) #tìm tọa độ y lớn nhất của khung chứa text int(max(box, key=lambda x: x[1])[1]),h-2 là tìm tọa độ y của các điểm trong khung chứa text, sau đó ép kiểu về int, nếu tọa độ y lớn nhất lớn hơn h-2 thì gán bằng h-2 
            bboxes_xxyy.append([x_min-1,x_max,y_min-1,y_max]) #thêm tọa độ của khung chứa text vào list bboxes_xxyy, tại sao lại trừ 1 ở đây? vì tọa độ của khung chứa text được tính từ 0, còn tọa độ của ảnh được tính từ 1, nên phải trừ 1 để tọa độ của khung chứa text và ảnh bằng nhau

        img_copy = image.copy() #tạo bản sao của ảnh
        for b in bboxes_xxyy: #duyệt qua các khung chứa text
            cv2.rectangle(img_copy, (b[0],b[2]),(b[1],b[3]),(255,0,0),1) #vẽ các khung chứa text lên ảnh bản sao
        plt.figure(figsize=(10,10)) #tạo 1 figure có kích thước 10x10, figure là 1 cửa sổ chứa các hình ảnh
        plt.imshow(img_copy) #hiển thị ảnh bản sao lên figure
        texts = [] #khởi tạo list chứa các text
        probs = [] #khởi tạo list chứa các xác suất của các text
        for box in bboxes_xxyy: #duyệt qua các khung chứa text
            x_min,x_max,y_min,y_max = box #lấy tọa độ của khung chứa text
            img = image[y_min:y_max,x_min:x_max,:]  #cắt ảnh theo tọa độ của khung chứa text, y_min:y_max là lấy tất cả các hàng từ y_min đến y_max, x_min:x_max là lấy tất cả các cột từ x_min đến x_max, : là lấy tất cả các kênh màu, sau đó gán cho biến img
            img = Image.fromarray(img) #chuyển ảnh từ dạng numpy array sang dạng Image, img là ảnh chứa 1 text
            s,prob = text_recognizer.predict(img,return_prob = True) #nhận dạng text trong ảnh img, s là text nhận dạng được, prob là xác suất nhận dạng có đúng là text hay không
            texts.append(s) #thêm text nhận dạng được vào list texts
            probs.append(prob) #thêm xác suất nhận dạng được vào list probs
            
        out,score = predict_svm(texts[10:]) #gọi hàm predict_svm để sử dụng kỹ thuật tf-idf kết hợp svm để gán nhãn cho các text, out là list chứa các nhãn của các text, score là list chứa các xác suất gán nhãn đúng, texts[10:] là lấy các text từ vị trí thứ 10 trở đi, lấy từ vị trí thứ 10 trở đi vì ?
        out_bert = predict_phoBert(texts[:10]) 

        rs_text = ""
        t_seller = None

        if len(np.where(out_bert==0)[0])!=0:
            seller_idx = np.where(out_bert==0)[0][0].item()
            text_seller = texts[seller_idx]
            rs_text += text_seller+"|||"
        else:
            rs_text += "|||"
        if len(np.where(out_bert==1)[0])!=0:
            add_idx = np.where(out_bert==1)[0]
            txt_address=""
            for idx in range(len(add_idx)):
                txt_address += texts[add_idx[idx].item()]
                if idx < len(add_idx)-1:
                    txt_address += " "
            rs_text += txt_address+ "|||"
        else:
            rs_text += "|||"
        date_str = None
        for idx,string in enumerate(texts):
    #                     if re.search("Ngày",string):
    #                         start_idx = re.search("Ngày",string).start() 
    #                         date_str = string[start_idx:]
                if date_finder(string):
                    date_str = string
                    date_idx = idx
                    if len(list(string)) > 30:
                        if re.search("Ngày",string):
                            start_idx = re.search("Ngày",string).start() 
                            date_str = date_str[start_idx:]
        for idx,string in enumerate(texts):
            if re.search("Ngay",string):
                start_idx = re.search("Ngay",string).start() 
                date_str = string[start_idx:]
        for idx,string in enumerate(texts):
            if re.search("Ngày",string):
                start_idx = re.search("Ngày",string).start() 
                date_str = string[start_idx:]
        if date_str:

            rs_text += date_str+"|||"
        else:
            rs_text += "|||"

        cost_idx = get_idx(out,score,0)

        if cost_idx!=None:
            txt_cst = texts[cost_idx]                  
            rs_text += txt_cst

            cst1_xmin,cst1_xmax,cst1_ymin,cst1_ymax = bboxes_xxyy[cost_idx]
            cst1_ycenter = (cst1_ymin+cst1_ymax)/2

            for box in bboxes_xxyy:
                if box == bboxes_xxyy[cost_idx]:
                    continue
                x_min,x_max,y_min,y_max = box
                if abs(cst1_ycenter-(y_max+y_min)/2)<13:
                    img = image[y_min:y_max,x_min:x_max,:]
                    img = Image.fromarray(img)
                    s,prob = text_recognizer.predict(img,return_prob = True)
                    rs_text +=" "+ s
        inp_task1 = []
        for prob in probs:
            if np.isnan(prob):
                continue
            inp_task1.append(prob)
        inp_task1 = sorted(inp_task1)
        if len(inp_task1)>100:
            inp1 = np.array(inp_task1[:100])
        else:
            inp1 = np.concatenate((inp_task1,np.zeros(100 - len(inp_task1), dtype=np.float32)))
            
        out_task1 = model_task1.predict([inp1])  
        out_task1 = out_task1.item()
    else:
        out_task1 = 0.2
        rs_text = "|||||||||"
    return rs_text,out_task1

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='MC-OCR inference')
    parser.add_argument('--folder_test', type=str, help='path to folder')
    args = parser.parse_args()


    predict_svm = predict_svm()
    predict_phoBert = predict_phoBert()
    predict_image = Predictor_image()
    #recognition
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = 'weights/transformerocr.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False
    text_recognizer = Predictor(config)
    model_task1 = joblib.load("weights/model_task1.sav")
    with open('results.csv', mode='w') as csv_file:
        fieldnames = ['img_id', 'anno_image_quality', 'anno_texts']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
        writer.writeheader()
        for file_name in os.listdir(args.folder_test):
            image = cv2.imread(os.path.join(args.folder_test,file_name))
            rs_text,out_task1 = test(image)
            writer.writerow({'img_id': file_name, 'anno_image_quality': out_task1, 'anno_texts': rs_text})




