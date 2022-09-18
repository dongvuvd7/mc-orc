import torch
import torch.backends.cudnn as cudnn 
from collections import OrderedDict #collections là một module trong python, nó được sử dụng để tạo các container data types như list, dict, set, tuple, OrderedDict là một container data type trong collections, nó giống như dict nhưng nó có thứ tự, nếu không có OrderedDict() thì trọng số sẽ bị đảo ngược
from torch.autograd import Variable #Variable là một class trong torch.autograd, nó được sử dụng để lưu trữ và tính toán gradient của tensor
import time #time là một module trong python, nó được sử dụng để đo thời gian
import cv2
import numpy as npư
from modules_craft import * #import các hàm trong file modules_craft.py

pretrained = 'weights/craft_mlt_25k.pth' # path to pretrained model
refiner_model = 'weights/craft_refiner_CTW1500.pth' 

#copyStateDict là hàm copy trọng số từ model đã train sang model mới, state_dict là trọng số của model
def copyStateDict(state_dict): 
    if list(state_dict.keys())[0].startswith("module"): # state_dict.keys()[0] trả về key đầu tiên trong list, .startswith("module") kiểm tra xem key đó có bắt đầu bằng "module" không, nếu có chứng tỏ model đã được load từ file .pth, nếu không thì model chưa được load từ file .pth
        start_idx = 1 # nếu model đã được load từ file .pth thì start_idx = 1
    else:
        start_idx = 0 # nếu model chưa được load từ file .pth thì start_idx = 0
    new_state_dict = OrderedDict() # OrderedDict() là một dictionary có thứ tự, nếu không có OrderedDict() thì trọng số sẽ bị đảo ngược
    for k, v in state_dict.items(): # k là key, v là value
        name = ".".join(k.split(".")[start_idx:]) # name là tên của trọng số, k.split(".")[start_idx:] là tách chuỗi k theo dấu chấm, sau đó lấy từ start_idx đến hết chuỗi
        new_state_dict[name] = v # thêm trọng số vào new_state_dict
    return new_state_dict # trả về new_state_dict

# Hàm test_net dùng để test model, đầu vào là ảnh, đầu ra là ảnh đã được dự đoán chứa các bounding box của text, tham số đầu vào gồm: net là model, image là ảnh, text_threshold là ngưỡng của text, link_threshold là ngưỡng của link, low_text là ngưỡng của text, cuda là sử dụng GPU hay không, poly là sử dụng poly hay không, refine_net là model refiner, canvas_size là kích thước của ảnh, mag_ratio là tỉ lệ phóng to của ảnh
def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None): 
    t0 = time.time() # t0 là thời gian bắt đầu chạy hàm test_net, (một số phần tính time để tùy chọn hiển thị thời gian xử lý)

    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(image, 1536, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5) # img_resized là ảnh đã được resize, target_ratio là tỉ lệ phóng to của ảnh, site_heatmap, resize_aspect_ratio là hàm resize ảnh theo tỉ lệ phóng to, đầu vào gồm: image là ảnh, 1536 tương ứng square_size, interpolation là phương pháp resize, mag_ratio là tỉ lệ phóng to của ảnh
    ratio_h = ratio_w = 1 / target_ratio  # ratio_h, ratio_w là tỉ lệ phóng to của ảnh theo chiều dọc và chiều ngang

    # preprocessing là tiền xử lý ảnh
    x = normalizeMeanVariance(img_resized) # x là ảnh đã được chuẩn hóa, normalizeMeanVariance là hàm chuẩn hóa ảnh theo mean và variance, đầu vào là ảnh, đầu ra là ảnh đã được chuẩn hóa, cần chuẩn hóa để model có thể dự đoán chính xác hơn
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w] # x là ảnh đã được chuyển đổi từ numpy sang tensor, permute là hàm chuyển đổi thứ tự của tensor, đầu vào là thứ tự mới, đầu ra là tensor đã được chuyển đổi
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w] # x là ảnh đã được thêm một chiều, unsqueeze là hàm thêm một chiều vào tensor, đầu vào là vị trí của chiều mới, đầu ra là tensor đã được thêm một chiều, trong đó b là số lượng ảnh, c là số kênh màu, h là chiều cao, w là chiều rộng
    if cuda: # nếu sử dụng GPU
        x = x.cuda() # chuyển x sang GPU

    # forward pass tức là chạy model
    # không tính toán gradient vì không cần tính gradient trong quá trình test, đây là một cách để tiết kiệm bộ nhớ, đồng thời cũng là một cách để tăng tốc độ, vì khi tính gradient thì cần lưu lại các giá trị của các biến trong quá trình tính gradient, còn khi không tính gradient thì không cần lưu lại các giá trị của các biến trong quá trình tính gradient, gradient được tính toán theo công thức: gradient = (f(x + h) - f(x)) / h, với h là một số nhỏ, h càng nhỏ thì độ chính xác của gradient càng cao, nhưng cũng càng tốn thời gian tính toán, nên khi không cần tính gradient thì không cần lưu lại các giá trị của các biến trong quá trình tính gradient
    with torch.no_grad(): #(gradient nói chung là liên quan đến tính độ nghiêng)
        y, feature = net(x) # y là output của model, feature là feature map của model, net là model, x là ảnh đã được chuẩn hóa

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy() #score text dùng để phân biệt vùng chứa chữ và vùng không chứa chữ, score_text là score của vùng chứa chữ, score_text là một ma trận 2 chiều, score_text.shape = (h, w), h là chiều cao, w là chiều rộng, score_text[i][j] là score của pixel ở vị trí (i, j), score_text[i][j] càng lớn thì pixel ở vị trí (i, j) càng có khả năng là pixel của vùng chứa chữ, score_text[i][j] càng nhỏ thì pixel ở vị trí (i, j) càng có khả năng là pixel của vùng không chứa chữ, .cpu.data.numpy() là hàm chuyển đổi tensor sang numpy, đầu vào là tensor, đầu ra là numpy
    score_link = y[0,:,:,1].cpu().data.numpy() #score link dùng để phân biệt các vùng chứa chữ có liên kết với nhau và các vùng chứa chữ không liên kết với nhau, score_link là score của các vùng chứa chữ có liên kết với nhau, score_link là một ma trận 2 chiều, score_link.shape = (h, w), h là chiều cao, w là chiều rộng, score_link[i][j] là score của pixel ở vị trí (i, j), score_link[i][j] càng lớn thì pixel ở vị trí (i, j) càng có khả năng là pixel của vùng chứa chữ có liên kết với nhau, score_link[i][j] càng nhỏ thì pixel ở vị trí (i, j) càng có khả năng là pixel của vùng chứa chữ không liên kết với nhau (các chữ liên kết với nhau có thể tạo ra từ)

    # refine link
    if refine_net is not None: # nếu có refine_net
        with torch.no_grad(): # không tính toán gradient
            y_refiner = refine_net(y, feature) # refine_net là hàm dùng để cải thiện kết quả của model, đầu vào là output của model và feature map của model, đầu ra là kết quả cải thiện của model, y_refiner là kết quả cải thiện của model
        score_link = y_refiner[0,:,:,0].cpu().data.numpy() # tính lại score_link

    t0 = time.time() - t0 # thời gian chạy hàm detect
    t1 = time.time() # t1 là thời gian hiện tại

    # Post-processing là bước xử lý sau khi đã có kết quả của model, bước này dùng để tìm các vùng chứa chữ và các vùng chứa chữ có liên kết với nhau
    boxes, polys = getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly) # hàm getDetBoxes dùng để tìm các vùng chứa chữ và các vùng chứa chữ có liên kết với nhau, đầu vào là score_text, score_link, text_threshold, link_threshold, low_text, poly, đầu ra là các vùng chứa chữ và các vùng chứa chữ có liên kết với nhau, boxes là các vùng chứa chữ, polys là các vùng chứa chữ có liên kết với nhau, text_threshold là ngưỡng score_text, link_threshold là ngưỡng score_link, low_text là ngưỡng score_text, poly là kiểu dữ liệu của các vùng chứa chữ có liên kết với nhau, nếu poly = True thì các vùng chứa chữ có liên kết với nhau là các đa giác, nếu poly = False thì các vùng chứa chữ có liên kết với nhau là các hình chữ nhật

    # coordinate adjustment là bước điều chỉnh tọa độ các vùng chứa chữ và các vùng chứa chữ có liên kết với nhau 
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h) # hàm adjustResultCoordinates dùng để điều chỉnh tọa độ các vùng chứa chữ, đầu vào là boxes, ratio_w, ratio_h, đầu ra là các vùng chứa chữ sau khi điều chỉnh tọa độ, ratio_w là tỉ lệ giữa chiều rộng của ảnh đầu vào và chiều rộng của ảnh sau khi resize, ratio_h là tỉ lệ giữa chiều cao của ảnh đầu vào và chiều cao của ảnh sau khi resize
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h) # hàm adjustResultCoordinates dùng để điều chỉnh tọa độ các vùng chứa chữ có liên kết với nhau, đầu vào là polys, ratio_w, ratio_h, đầu ra là các vùng chứa chữ có liên kết với nhau sau khi điều chỉnh tọa độ, ratio_w là tỉ lệ giữa chiều rộng của ảnh đầu vào và chiều rộng của ảnh sau khi resize, ratio_h là tỉ lệ giữa chiều cao của ảnh đầu vào và chiều cao của ảnh sau khi resize
    for k in range(len(polys)): # duyệt qua các vùng chứa chữ có liên kết với nhau
        if polys[k] is None: polys[k] = boxes[k] # nếu polys[k] = None thì polys[k] = boxes[k]

    t1 = time.time() - t1 # thời gian chạy hàm post-processing

    # render results (optional) là bước vẽ kết quả, bước này dùng để vẽ các vùng chứa chữ và các vùng chứa chữ có liên kết với nhau lên ảnh đầu vào
    render_img = score_text.copy() # tạo một bản sao của score_text
    render_img = np.hstack((render_img, score_link)) # nối score_link vào bên phải của render_img
    ret_score_text = cvt2HeatmapImg(render_img) # chuyển render_img sang ảnh heatmap

    # if args.show_time : print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text # trả về các vùng chứa chữ, các vùng chứa chữ có liên kết với nhau và ảnh heatmap, boxes có dạng [[x1, y1, x2, y2, x3, y3, x4, y4], [x1, y1, x2, y2, x3, y3, x4, y4], ...], polys có dạng [[x1, y1, x2, y2, x3, y3, x4, y4], [x1, y1, x2, y2, x3, y3, x4, y4], ...], ret_score_text là ảnh heatmap

#init
net = CRAFT() # khởi tạo một đối tượng net thuộc lớp CRAFT
net.load_state_dict(copyStateDict(torch.load(pretrained))) # load trọng số của mạng đã được train sẵn
# net = net.cuda()
net = torch.nn.DataParallel(net) # dùng để chạy song song trên nhiều GPU
cudnn.benchmark = False # dùng để tăng tốc độ chạy
net.eval() #Đánh giá mô hình

    # LinkRefiner tức là bước cải thiện kết quả, bước này dùng để cải thiện các vùng chứa chữ có liên kết với nhau
refine_net = RefineNet() # khởi tạo một đối tượng refine_net thuộc lớp RefineNet
# if args.cuda: tức là nếu có GPU thì load trọng số của mạng đã được train sẵn lên GPU
refine_net.load_state_dict(copyStateDict(torch.load(refiner_model))) # load trọng số của mạng đã được train sẵn
# refine_net = refine_net.cuda()
refine_net = torch.nn.DataParallel(refine_net) # dùng để chạy song song trên nhiều GPU
# else:
#     refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))
refine_net.eval() #Đánh giá mô hình
poly = True 
