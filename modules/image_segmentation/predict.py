from detectron2.engine import DefaultPredictor 
from detectron2.config import get_cfg 
import cv2
import numpy as np
from detectron2 import model_zoo
#load config file và pre-trained model weights
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = 'weights/model_segmentation.pth'  # path to the model we just trained: load model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95   # set a custom testing threshold: đặt ngưỡng cho confidence score của các output đầu ra của mô hình là 0.95, ngưỡng này được sử dụng để lọc ra các bounding box có confidence score cao hơn ngưỡng này, các bounding box có confidence score thấp hơn ngưỡng này sẽ bị loại bỏ
predictor = DefaultPredictor(cfg) #tạo một predictor từ config file và model weights

#Hàm cắt ảnh dựa trên ảnh mặt nạ nhị phân (img là ảnh gốc, mask là ảnh mặt nạ nhị phân, tolerance là độ chênh lệch giữa các giá trị RGB của ảnh gốc và ảnh mặt nạ nhị phân)
def get_segment_crop(img, tol=0, mask=None):
    if mask is None: #nếu không có ảnh mặt nạ nhị phân thì sẽ sinh ra ảnh mặt nạ nhị phân từ ảnh gốc
        mask = img > tol #ảnh mask được tạo ra bằng cách so sánh giá trị của ảnh gốc với giá trị ngưỡng tol, nếu giá trị của ảnh gốc lớn hơn tol thì giá trị của ảnh mask là 1, ngược lại là 0
    return img[np.ix_(mask.any(1), mask.any(0))] #mask.any(1) trả về một mảng các giá trị True/False, True nếu có ít nhất một giá trị khác 0 trong hàng, False nếu tất cả các giá trị trong hàng đều bằng 0, mask.any(0) tương tự như mask.any(1) nhưng trả về mảng các giá trị True/False theo cột, sau đó dùng hàm np.ix_ để lấy các hàng và cột có giá trị True, (True chứng tỏ phần tử tại đó là thuộc vật thể) cuối cùng trả về ảnh đã cắt

# 
def segment_single_images(image, save_img=False):
    error_ims = [] #
    segmen_info = [] #

#     image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #chuyển ảnh về dạng BGR
    output_predictor = predictor(image) #output_predictor là kết quả dự đoán xác định vị trí của các vật thể trong ảnh, output_predictor là một dictionary chứa các thông tin về các vật thể trong ảnh, bao gồm: bounding box, mask, class, score, 
    if output_predictor['instances'].pred_masks.shape[0] > 1: #Nếu có nhiều hơn 1 vật thể trong ảnh thì sẽ lấy vật thể có confidence score cao nhất, .pred_masks.shape[0] là số lượng vật thể trong ảnh 
        mask_check = output_predictor['instances'].pred_masks.cpu().numpy() #.pred_masks là một tensor chứa các ảnh mặt nạ nhị phân của các vật thể trong ảnh, .cpu().numpy() chuyển tensor sang numpy array, (tensor là một kiểu dữ liệu của pytorch, numpy array là một kiểu dữ liệu của numpy), thu mask_check là một numpy array chứa các ảnh mặt nạ nhị phân của các vật thể trong ảnh
        masks = output_predictor['instances'].pred_masks.cpu().numpy() # masks là một numpy array chứa các ảnh mặt nạ nhị phân của các vật thể trong ảnh (như mask_check)
        mask_binary = masks[np.argmax(np.sum(masks, axis=(1, 2))) ,:,:] #mask_binary là ảnh mặt nạ nhị phân của vật thể có confidence score cao nhất, np.sum(masks, axis=(1, 2)) trả về một mảng các giá trị tổng của các ảnh mặt nạ nhị phân theo hàng và cột, np.argmax(np.sum(masks, axis=(1, 2))) trả về chỉ số của vật thể có confidence score cao nhất, confidence score được tính bằng tổng các giá trị của ảnh mặt nạ nhị phân của vật thể đó

    else: #Nếu chỉ có 1 vật thể trong ảnh thì mask_binary là ảnh mặt nạ nhị phân của vật thể đó
        mask_binary = np.squeeze(output_predictor['instances'].pred_masks.permute(1, 2, 0).cpu().numpy()) #.permute để chuyển vị trí các chiều của tensor (chưa hiểu) (trong đó 1 là chiều hàng, 2 là chiều cột, 0 là chiều kênh màu), np.squeeze() loại bỏ các chiều có kích thước là 1 (nếu không loại bỏ thì ảnh mặt nạ nhị phân sẽ có 3 chiều, chiều thứ 3 là chiều kênh màu, trong khi ảnh mặt nạ nhị phân chỉ có 2 chiều), .cpu().numpy() chuyển tensor sang numpy array, thu mask_binary là một numpy array chứa ảnh mặt nạ nhị phân của vật thể trong ảnh
    try: #Sau khi lấy được ma trận ảnh mặt nạ nhị phân thì cắt ảnh theo ma trận ảnh mặt nạ nhị phân đó
        crop_mask = get_segment_crop(img = image, mask = mask_binary) 
        
    except ValueError: #nếu không có mask nào được tìm thấy thì sẽ báo lỗi
        print("error")
    origin_mask = cv2.cvtColor(np.float32(mask_binary) * 255.0, cv2.COLOR_GRAY2RGB) #chuyển ảnh mặt nạ nhị phân sang ảnh màu

    #Hiển thị ảnh gốc, ảnh mặt nạ nhị phân, ảnh sau khi cắt
    for j in range(image.shape[2]): # image.shape[2] là số kênh màu của ảnh, ảnh màu có 3 kênh màu
        image[:,:,j] = image[:,:,j] * origin_mask[:,:,j] * 255 #image[:,:,j] là ảnh màu của kênh màu thứ j, origin_mask[:,:,j] là ảnh mặt nạ nhị phân của kênh màu thứ j, ảnh mặt nạ nhị phân có giá trị 0 hoặc 1, nếu ảnh mặt nạ nhị phân có giá trị 0 thì ảnh màu sẽ bị mờ đi, nếu ảnh mặt nạ nhị phân có giá trị 1 thì ảnh màu sẽ không bị mờ đi (ảnh hóa đơn thì phần nền màu đen, chỉ giữ lạiphần hóa đơn chính)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # chuyển ảnh màu từ BGR sang RGB, trả về ảnh màu sau khi cắt
