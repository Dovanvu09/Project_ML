import string
import numpy as np
from collections import Counter
from math import log, exp

class NaiveBayesClassifier:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing  # Laplace Smoothing
        self.class_probs = {}  # P(C)
        self.word_probs_by_class = {}  # P(word | class)
        self.vocab = set()  # Từ điển (Vocabulary)

    # Tiền xử lý văn bản: loại bỏ dấu câu và chuyển thành chữ thường
    def clean_text(self, text):
        text = text.lower()  # Chuyển thành chữ thường
        text = text.translate(str.maketrans('', '', string.punctuation))  # Loại bỏ dấu câu
        return text

    # Tokenize văn bản và loại bỏ từ dừng (nếu có)
    def tokenize(self, text, stopwords=None):
        tokens = text.split()
        if stopwords:
            tokens = [word for word in tokens if word not in stopwords]
        return tokens

    # Tiền xử lý văn bản cho toàn bộ tập dữ liệu
    def preprocess_data(self, texts, stopwords=None):
        return [self.tokenize(self.clean_text(text), stopwords) for text in texts]

    # Tính xác suất tiên nghiệm cho các lớp P(C)
    def calculate_class_probs(self, labels):
        class_counts = Counter(labels)
        total_count = len(labels)
        self.class_probs = {class_label: count / total_count for class_label, count in class_counts.items()}

    # Tính xác suất điều kiện P(word | class)
    def calculate_word_probs(self, tokens, labels):
        word_counts_by_class = {label: Counter() for label in set(labels)}
        class_counts = Counter(labels)
        
        # Đếm tần suất từ trong từng lớp
        for text, label in zip(tokens, labels):
            word_counts_by_class[label].update(text)
        
        # Tính xác suất điều kiện cho mỗi từ trong mỗi lớp
        word_probs_by_class = {}
        vocab_size = len(self.vocab)  # Số lượng từ trong từ điển (Vocabulary)
        
        for class_label, word_counts in word_counts_by_class.items():
            total_words_in_class = sum(word_counts.values())
            word_probs = {word: (count + self.smoothing) / (total_words_in_class + self.smoothing * vocab_size)  # Laplace smoothing
                          for word, count in word_counts.items()}
            word_probs_by_class[class_label] = word_probs
        
        self.word_probs_by_class = word_probs_by_class

    # Huấn luyện mô hình Naive Bayes
    def fit(self, texts, labels, stopwords=None):
        # Tiền xử lý dữ liệu
        tokens = self.preprocess_data(texts, stopwords)
        
        # Cập nhật từ điển (vocabulary)
        self.vocab = set([word for text in tokens for word in text])
        
        # Tính xác suất tiên nghiệm cho các lớp
        self.calculate_class_probs(labels)
        
        # Tính toán xác suất điều kiện (Likelihood) cho mỗi lớp
        self.calculate_word_probs(tokens, labels)

    # Dự đoán lớp cho một văn bản mới
    def predict(self, text, stopwords=None):
        # Tiền xử lý văn bản mới
        tokens = self.tokenize(self.clean_text(text), stopwords)
        
        # Tính toán xác suất của văn bản cho mỗi lớp
        class_scores = {}
        for class_label, class_prob in self.class_probs.items():
            score = log(class_prob)  # Log của xác suất tiên nghiệm P(C)
            for word in tokens:
                # Nếu từ có trong lớp, cộng log của xác suất điều kiện P(word | class)
                if word in self.vocab:
                    word_prob = self.word_probs_by_class[class_label].get(word, self.smoothing / (sum([sum(word_counts.values()) for word_counts in self.word_probs_by_class.values()]) + len(self.vocab)))  # Nếu từ chưa có trong lớp, dùng Laplace smoothing
                    score += log(word_prob)
            class_scores[class_label] = score
        
        # Chọn lớp có xác suất cao nhất
        predicted_class = max(class_scores, key=class_scores.get)
        return predicted_class

    # Dự đoán cho toàn bộ tập dữ liệu
    def predict_all(self, texts, stopwords=None):
        return [self.predict(text, stopwords) for text in texts]

    # Tính toán hàm mất mát (Log Loss / Cross-entropy loss)
    def log_loss(self, X, y):
        loss = 0.0
        for text, true_label in zip(X, y):
            predicted_label = self.predict(text)
            predicted_prob = self.class_probs.get(predicted_label, 0.0)
            true_prob = 1 if predicted_label == true_label else 0
            loss += -true_prob * log(predicted_prob) - (1 - true_prob) * log(1 - predicted_prob)
        return loss / len(y)

# Ví dụ sử dụng Naive Bayes

# Dữ liệu ví dụ
texts = ["Tôi yêu sản phẩm này", "Dịch vụ rất tệ", "Tôi không thích bộ phim này"]
labels = ["positive", "negative", "negative"]

# Khởi tạo Naive Bayes
naive_bayes = NaiveBayesClassifier()

# Huấn luyện mô hình
naive_bayes.fit(texts, labels)

# Dự đoán cho văn bản mới
new_text = "Dịch vụ này rất tệ"
predicted_class = naive_bayes.predict(new_text)
print(f"Predicted sentiment: {predicted_class}")

# Dự đoán cho tất cả dữ liệu
predictions = naive_bayes.predict_all(texts)
print(f"Predictions: {predictions}")

# Tính toán hàm mất mát (Log Loss)
loss = naive_bayes.log_loss(texts, labels)
print(f"Log Loss: {loss}")
