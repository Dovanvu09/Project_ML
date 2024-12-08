import numpy as np
import matplotlib.pyplot as plt
import joblib


class SVMClassifier:
    def __init__(self, C = 1.0, kernel = 'linear', max_iter = 1000, tol = 1e-3, learning_rate = 1e-3):
        self.C = C  # Tham số điều chỉnh mức độ phạt cho những mẫu dữ liệu bị phân loại sai
        self.kernel = kernel  # Kernel để biến đổi dữ liệu thành dạng không gian đặc trưng khác
        self.max_iter = max_iter  # Số lần lặp tối đa
        self.tol = tol  # Độ chênh lệch để dừng thuật toán khi hội tụ
        self.learning_rate = learning_rate  # Tốc độ học cho thuật toán Gradient Descent
        self.models = {}  # Lưu các mô hình SVM cho từng cặp nhãn trong phân loại đa lớp
        self.model_accuracies = {}  # Đánh giá độ chính xác của các mô hình
        self.loss_history = []  # Lưu giá trị hàm mất mát qua các epoch
        self.accuracy_history = []  # Lưu độ chính xác qua các epoch
    
    """
    hàm kernel ở đây cho phép ta sử dụng kĩ thuật kernel nếu dữ liệu của chúng ta không phân biệt tuyến tính
    thì khi cho  đi qua kernel , tập dữ liệu x sẽ được map sang một không gian khác nhiều chiều hơn
    giúp chúng ta có thể phân tích tuyến tính  ( linear separable ) ra được
    """
    def _kernel(self, x1, x2):
        if self.kernel == 'linear':
            return  np.dot(x1, x2)
        
        elif self.kernel == "polynomial":
            return (1 + np.dot(x1, x2)) *3
        
        elif self.kernel == 'rbf' :
            gamma = 1 / len(x1)
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        
        else:
            raise ValueError(f"Unspported kernel : {self.kernel} " )
    

    """
    Ở trong mô hình này với tập dữ liệu chia thành 3 class thì chúng ta cần mô hình phân loại đa lớp
    bằng các sử dụng phương pháp ONE VS ONE, mỗi cặp class_i, và class_j sẽ huấn luyện một mô hình SVM 
    sau đó huấn luyện mô hình nhị phân với các mẫu rồi lưu vào models
    """
    def fit(self, X, y):
        # Lấy các nhãn khác nhau
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Huấn luyện mô hình SVM cho từng cặp nhãn
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                print(f"Train class {i} and class {j}" )
                class_i = self.classes[i]
                class_j = self.classes[j]

                # Lọc dữ liệu cho cặp nhãn class_i và class_j
                idx = np.where((y == class_i) | (y == class_j))[0]
                X_pair = X[idx]
                y_pair = y[idx]
                y_pair = np.where(y_pair == class_i, 1, -1)  # Chuyển thành nhãn nhị phân

                # Khởi tạo và huấn luyện mô hình SVM cho cặp nhãn
                model = SVMClassifier(C=self.C, kernel=self.kernel, max_iter=self.max_iter, tol=self.tol, learning_rate=self.learning_rate)
                model._fit_binary(X_pair, y_pair)
                self.models[(class_i, class_j)] = model
                
                # đánh giá độ chính xác của mô hình
                y_pred = model._predict_binary(X_pair)
                accuracy = np.mean(y_pair == y_pred)
                self.model_accuracies[(class_i, class_j)] = accuracy

    """
    sau khi chia các tập data ở trên thành tập dữ liệu có thể phân tách tuyến tính 
    và lưu model vào trong models, ta tiếp tục chuyển đến bước huấn luyện mô hình 
    cho bài toán nhị phân
    """

    def _fit_binary(self, X, y):
        """ Huấn luyện mô hình SVM nhị phân """
        y = np.where(np.array(y) == 1, 1, -1)  # Chuyển đổi nhãn thành -1, 1
        n_samples, n_features = X.shape  # Lấy số lượng mẫu và đặc trưng

        # Khởi tạo trọng số và bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(1, self.max_iter + 1):
            weights_prev = np.copy(self.weights)
            bias_prev = self.bias

            for i in range(n_samples):
                # Sử dụng phương thức .dot() cho sparse matrix
                condition = y[i] * (X[i].dot(self.weights) + self.bias) < 1  # Điều kiện kiểm tra

                if condition:
                    # Cập nhật trọng số và bias khi điều kiện thỏa mãn
                    self.weights += self.learning_rate * (y[i] * X[i].toarray().flatten() - 2 * (1 / self.max_iter) * self.weights)
                    self.bias += self.learning_rate * y[i]
                else:
                    # Cập nhật trọng số khi điều kiện không thỏa mãn
                    self.weights += self.learning_rate * (-2 * (1 / self.max_iter) * self.weights)

            current_loss = self.loss(X, y)  # Tính loss hiện tại
            self.loss_history.append(current_loss)

            y_pred = self._predict_binary(X)  # Dự đoán
            accuracy = np.mean(y == y_pred)  # Tính độ chính xác
            self.accuracy_history.append(accuracy)

            print(f"Epoch {epoch}, Loss: {current_loss:.4f}, Accuracy: {accuracy:.4f}")

            # Kiểm tra sự thay đổi giữa các epoch để dừng sớm nếu không thay đổi đáng kể
            diff = np.linalg.norm(self.weights - weights_prev) + abs(self.bias - bias_prev)
            if diff < self.tol:
                break

        # Tính giá trị quyết định và xác định các vector hỗ trợ
        decision_values = y * (X.dot(self.weights) + self.bias)
        self.support_vectors = X[decision_values <= 1]
        self.support_labels = y[decision_values <= 1]

    
    def loss(self, X, y):
        """ Tính giá trị loss (hinge loss) cho SVM phân loại nhị phân """
        n_samples = X.shape[0]
        
        # Mảng bias cho tất cả các mẫu (tất cả đều có giá trị giống self.bias)
        bias_array = np.full(n_samples, self.bias)
        
        # Tính giá trị quyết định cho tất cả các mẫu
        decision_values = X.dot(self.weights) + bias_array
        
        # Tính giá trị hinge loss cho mỗi mẫu
        hinge_losses = np.maximum(0, 1 - y * decision_values)
        
        # Trả về giá trị loss trung bình
        return np.mean(hinge_losses)


    
    def predict (self, X):
        """
        Input [X1,
               X2,
               X3,
               X4]
        ta dùng phương thức predict_binary để dự đoán cho từng mẫu dữ liệu, giả sử ta cần phân loại 3 lớp 
        A VS B : [1, -1, 1, -1] 
        A VS C : [1, 1, -1, -1]
        B VS C : [-1, 1, -1, 1]

        A(X1) = 2
        C(X2) = 2
        C(X3) = 2
        B(X4) = 2 

        """
        """ Dự đoán nhãn cho tập dữ liệu đầu vào X """
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))

        for (class_i, class_j), model in self.models.items():
            y_pred = model._predict_binary(X)
            print(y_pred)
            accuracy = self.model_accuracies[(class_i, class_j)]
            for idx, pred in enumerate(y_pred):
                if pred == 1:
                    votes[idx, np.where(self.classes == class_i)[0][0]] += 1
                else:
                    votes[idx, np.where(self.classes == class_j)[0][0]] += 1
        
        return self.classes[np.argmax(votes, axis=1)]
    
    def save(self, filename):
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        return joblib.load(filename)
    

    def _predict_binary(self, X):
        """ Dự đoán nhãn cho mô hình SVM nhị phân """
        # Tạo mảng bias với kích thước (n_samples,)
        bias_array = np.full(X.shape[0], self.bias)

        # Dự đoán nhãn với ma trận sparse X
        decision_values = X.dot(self.weights) + bias_array

        # Sử dụng np.sign để trả về dấu của các giá trị quyết định
        return np.sign(decision_values)

    def evaluate(self, X, y):
        # Đánh giá mô hình trên tập dữ liệu X và y
        y_pred = self.predict(X)
        accuracy = np.mean(y == y_pred)
        f1 = self._f1_score(y, y_pred)
        recall = self._recall_score(y, y_pred)
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'recall': recall
        }

    def _recall_score(self, y_true, y_pred):
        # Tính recall thủ công
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred != 1))
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        return recall

    def _f1_score(self, y_true, y_pred):
        # Lấy các lớp mà mô hình dự đoán
        classes = np.unique(y_true)  # Lấy tất cả các lớp từ nhãn thực tế

        # Khởi tạo các biến để tính F1-score cho từng lớp
        f1_scores = []

        for class_label in classes:
            # Chuyển lớp thành 1 vs. các lớp còn lại để tính F1-score từng lớp
            y_true_binary = (y_true == class_label).astype(int)
            y_pred_binary = (y_pred == class_label).astype(int)

            # Tính true positives, false positives và false negatives cho từng lớp
            true_positives = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            false_positives = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            false_negatives = np.sum((y_true_binary == 1) & (y_pred_binary == 0))

            # Tính precision và recall
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

            # Tính F1-score cho lớp này
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            # Thêm F1-score của lớp này vào danh sách
            f1_scores.append(f1)

        # Tính giá trị F1-score trung bình
        f1_mean = np.mean(f1_scores)

        return f1_mean

    
    def plot_loss_accuracy(self):
        """ Vẽ đồ thị mất mát và độ chính xác """
        epochs = range(1, len(self.loss_history) + 1)

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.loss_history, label='Loss', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Hàm Mất Mát (Loss) qua các Epoch')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.accuracy_history, label='Accuracy', color='blue')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Độ Chính Xác (Accuracy) qua các Epoch')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


