import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
from vncorenlp import VnCoreNLP

#VnCoreNLP là thư viện dùng để tách từ tiếng Việt
#PhoBERT là một mô hình tiền huấn luyện dành riêng cho bài toán phân loại văn bản tiếng Việt
#Tokenize là quá trình mã hóa các văn bản thành các index dạng số mang thông tin của văn bản để cho máy tính có thể huấn luyện được. Khi đó mỗi một từ hoặc ký tự sẽ được đại diện bởi một index.

#class model_classify_IE 
class model_classify_IE(nn.Module): 
    #Hàm khởi tạo
    def __init__(self,num_cls): #num_cls là số lượng nhãn
        super().__init__()
        self.model = AutoModel.from_pretrained("vinai/phobert-base") #Tải mô hình phoBERT
        self.clf = nn.Sequential( #Tạo một mạng nơ-ron, nn.Sequential là một mạng nơ-ron tuần tự
            nn.Linear(768,256), #Lớp kết nối đầy đủ, 768 là số chiều của đầu vào, 256 là số chiều của đầu ra, nn.Linear là một lớp kết nối đầy đủ
            nn.ReLU(), #Hàm kích hoạt ReLU
            nn.Dropout(0.5), #Lớp dropout, giảm thiểu overfitting
            nn.Linear(256,num_cls) #Lớp kết nối đầy đủ, 256 là số chiều của đầu vào, num_cls là số chiều của đầu ra
        )
        self.activation = nn.Sigmoid() #Hàm kích hoạt Sigmoid
    #Hàm forward dùng để chạy mạng nơ-ron
    def forward(self,x): 
        x = self.model(x) #Chạy mô hình phoBERT
        x = self.clf(x[1]) #Chạy mạng nơ-ron
        return self.activation(x) #Trả về kết quả

#class MyDataset để tạo dataset cho mạng nơ-ron
class MyDataset(Dataset):
    #Hàm khởi tạo
    def __init__(self,data):
        self.data = data #Dữ liệu đầu vào
        self.rdrsegmenter = VnCoreNLP("weights/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') #Tạo đối tượng tách từ, đối tượng này sẽ được sử dụng để tách từ cho các câu trong dữ liệu đầu vào, annotators="wseg" là chỉ tách từ, max_heap_size='-Xmx500m' là dung lượng tối đa của bộ nhớ được sử dụng để tách từ
    #Hàm __getitem__ dùng để lấy dữ liệu từ dataset
    def __getitem__(self,idx): #idx là chỉ số của dữ liệu cần lấy
        x = self.data[idx]  #Lấy dữ liệu
        x = self.tokenize(x) #mã hóa dữ liệu
        sample ={ #Tạo một sample, sample là một từ điển, key là tên của dữ liệu, value là dữ liệu
            "input":x
        }
        return sample #Trả về sample
    #Hàm __len__ dùng để lấy số lượng dữ liệu trong dataset
    def __len__(self):
        return len(self.data)
    #Hàm tokenize 
    def tokenize(self,text): #đầu vào là một câu
        try : #Cố gắng thực hiện
            sents = self.rdrsegmenter.tokenize(text) #Tách từ, sents là một list các từ được tách từ câu đầu vào,rdrsegmenter.tokenize(text) là hàm tách từ, thu được sent là một list các từ được tách dưới dạng đã mã hóa
            text_token = ' '.join([' '.join(sent) for sent in sents]) #Ghép các từ lại thành một câu, text_token là câu sau khi ghép dưới dạng đã mã hóa
        except: #Nếu có lỗi thì thực hiện
            print(text)
            text_token = ''
            print("fail")
        return text_token

#Hàm AlignCollate dùng để tạo batch cho mạng nơ-ron, batch là một tập các dữ liệu được lấy từ dataset
class AlignCollate(object):
    #Hàm khởi tạo
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base") #Tạo đối tượng mã hóa câu, đối tượng này sẽ được sử dụng để mã hóa các câu trong dữ liệu đầu vào
    #Hàm __call__ dùng để tạo batch
    def __call__(self, batch): #batch là một tập các dữ liệu được lấy từ dataset
        x = [sample['input'] for sample in batch] #Lấy các câu trong batch
        
        x = self.tokenizer(x,padding=True)["input_ids"] #Mã hóa các câu trong batch, padding=True là thêm các ký tự đặc biệt vào các câu để đưa các câu về cùng độ dài, sau khi mã hóa x là một list các câu đã được mã hóa, mỗi câu là một list các số nguyên, input_ids là key của dữ liệu đã mã hóa
        
        return torch.tensor(x) #Trả về batch

#Hàm predict_phoBert dùng để dự đoán
class predict_phoBert(nn.Module):
    #Hàm khởi tạo
    def __init__(self,path_model='weights/text_classification.pth'):
        super().__init__()
        self.model = model_classify_IE(3) #Tạo một mạng nơ-ron, 3 là số lượng nhãn
        self.model.load_state_dict(torch.load(path_model)) #Tải trọng số của mạng nơ-ron, path_model là đường dẫn đến file chứa trọng số của mạng nơ-ron
        self.collate_fn = AlignCollate() #Tạo đối tượng AlignCollate để tạo batch
    #Hàm forward dùng để dự đoán
    def forward(self,texts): #texts là một list các câu cần dự đoán
        test_gen = DataLoader(MyDataset(texts),batch_size =32,shuffle = False, num_workers = 0,collate_fn=self.collate_fn) #Tạo một đối tượng DataLoader, DataLoader dùng để tạo batch cho mạng nơ-ron, MyDataset(texts) là một dataset, batch_size =32 là kích thước của một batch, shuffle = False là không xáo trộn dữ liệu, num_workers = 0 là số lượng tiến trình xử lý dữ liệu, collate_fn=self.collate_fn là đối tượng tạo batch
        x = next(iter(test_gen)) #Lấy một batch từ DataLoader
        out = self.model(x) #Dự đoán
        out = torch.softmax(out,-1) #Chuyển đổi các giá trị dự đoán thành xác suất
        return torch.argmax(out,-1) #Trả về nhãn dự đoán
                          
