import  cv2
import numpy as np
import imutils	
import math

#Hàm align_box dùng để căn chỉnh ảnh nếu ảnh bị lệch
def align_box(image, bboxes, skew_threshold=5, top_box=3): #đầu vào bao gồm: ảnh, bouding box, ngưỡng lệch, số lượng box lấy ra để tính trung bình
    vertical_vector = [0, -1] #vector chiều dọc
    top_box = np.argpartition([box[1][0]- box[0][0] for box in bboxes], -top_box)[-top_box:] #lấy ra 3 box có chiều dài lớn nhất rồi lấy chỉ số của chúng gán vào top_box
    avg_angle = 0 #tính trung bình góc lệch
    for idx in top_box: #duyệt qua các box
        skew_vector = bboxes[idx][0] - bboxes[idx][3] #tính vector lệch bằng cách lấy điểm góc trên bên trái trừ đi điểm góc dưới bên trái
        angle = np.math.atan2(np.linalg.det([vertical_vector,skew_vector]),np.dot(vertical_vector,skew_vector)) #tính góc lệch, trả về angle có đơn vị là radian, 1 radian = 180/np.pi = 57.2958 độ, 0,9 radian = 51.5 độ, tức là gốc lệnh > 51.5 độ thì ảnh bị lệch
        avg_angle += math.degrees(angle)/3 #tính trung bình góc lệch, /3 là vì chỉ lấy 3 box
    #Nếu trung bình góc lệch lớn hơn ngưỡng lệch thì sẽ căn chỉnh ảnh, ngược lại thì không
    if abs(avg_angle) < skew_threshold: 
        return image,0
    return imutils.rotate(image, avg_angle),1 #Xoay ảnh một góc bằng trung bình góc lệch, avg_angle > 0 thì xoay ngược chiều kim đồng hồ

#Hàm rotate_box dùng để xoay ảnh và bounding box nếu ảnh nằm ngang hoặc ngược  
def rotate_box(img, bboxes,degree,rotate_90, flip): #đầu vào bao gồm: ảnh, bouding box, góc xoay,nếu xoay 90 độ, nếu xoay 180 độ
    h,w = img.shape[:2] #lấy chiều cao và chiều rộng của ảnh
    if degree: #nếu có góc xoay
        new_bboxes = [[[h - i[1], i[0]] for i in bbox] for bbox in bboxes] #xoay các điểm của bouding box
        new_img = cv2.rotate(img, degree) #xoay ảnh
        return new_img, np.array(new_bboxes) #trả về ảnh và bouding box mới, np.array(new_bboxes) để chuyển list thành numpy array
    if rotate_90: #nếu xoay 90 độ (ảnh nằm ngang)
        new_bboxes = [[[h - i[1], i[0]] for i in bbox] for bbox in bboxes] #xoay các điểm của bouding box
        new_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) # xoay ảnh 90 độ theo chiều kim đồng hồ
        return new_img, np.array(new_bboxes) #trả về ảnh và bouding box mới, np.array(new_bboxes) để chuyển list thành numpy array
    if flip: #nếu xoay 180 độ (ảnh nằm ngược)
        new_bboxes = [[[w-i[0], h-i[1]] for i in bbox] for bbox in bboxes] #xoay các điểm của bouding box
        new_img = cv2.rotate(img, cv2.ROTATE_180) # xoay ảnh 180 độ
        return new_img, np.array(new_bboxes) #trả về ảnh và bouding box mới, np.array(new_bboxes) để chuyển list thành numpy array
    return img, bboxes #nếu không xoay thì trả về ảnh và bouding box cũ

#Hàm get_idx để lấy ra các chỉ số của các bouding box có chiều dài lớn hơn chiều rộng
def get_idx(out,score,label): #đầu vào bao gồm: output của mô hình, ngưỡng score, label
    rs_idx = None #khởi tạo rs_idx
    m = max(score, key=lambda x: x[0])[0] #lấy ra score lớn nhất, key=lambda x: x[0] để lấy ra phần tử đầu tiên của mỗi phần tử trong list
    for idx in range(len(out)): #duyệt qua các bouding box
        if  out[idx] == label and score[idx][0] >= m: #nếu bouding box có label là label và score lớn hơn score lớn nhất
            rs_idx = idx #lấy ra chỉ số của bouding box
    return rs_idx #trả về chỉ số của bouding box