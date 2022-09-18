import pickle #pickle là một thư viện của python dùng để lưu trữ dữ liệu dưới dạng file

#tf-idf là một kỹ thuật để đánh giá mức độ quan trọng của một từ trong một văn bản, nó được tính bằng cách chia số lần xuất hiện của từ đó trong văn bản cho tổng số lần xuất hiện của tất cả các từ trong văn bản đó kèm theo ước lượng mức độ quan trọng của từ đó như thế nào, khi tính tần số xuất hiện thì các từ đều quan trọng như nhau nhưng có một ssố từ thường sử dụng nhiều nhưng không quan trọng như: và, tuy, nhưng, trong, ...
#(https://viblo.asia/p/tf-idf-term-frequency-inverse-document-frequency-JQVkVZgKkyd)
#svm là một kỹ thuật phân lớp
#(https://viblo.asia/p/gioi-thieu-ve-support-vector-machine-svm-6J3ZgPVElmB)

#Class preidct_svm dùng để phân loại nhãn
class predict_svm(object):
    #Hàm khởi tạo
    def __init__(self,path_svm='weights/svm_model_v3.pkl',path_tf_idf='weights/tfidf_vectorization_v3.pkl'): 
        super().__init__()
        self.svm = pickle.load(open(path_svm, 'rb')) #Đọc file svm_model_v3.pkl, rb là read binary
        self.Tfidf_vect = pickle.load(open(path_tf_idf, 'rb')) #Đọc file tfidf_vectorization_v3.pkl, rb là read binary
    #Hàm __call__ dùng để gọi đối tượng (hàm xử lý)
    def __call__(self, texts): #đầu vào là một văn bản
        inp = self.Tfidf_vect.transform(texts) #Tính tf-idf cho văn bản đầu vào
        out = self.svm.predict(inp) #Dự đoán nhãn cho văn bản đầu vào
        out_proba = self.svm.predict_proba(inp) #Dự đoán xác suất gán nhãn đúng
        return out,out_proba #Trả về nhãn và xác suất gán nhãn đúng
