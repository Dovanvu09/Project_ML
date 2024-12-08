import pandas as pd
import re

# Danh sách stopwords
stopwords = set([
    "và", "là", "thì", "mà", "nhưng", "hoặc", "của", "đã", "đang", "rằng", "một", "cái", "này",
    "nọ", "đó", "ấy", "những", "các", "tất", "cả", "vậy", "nên", "để", "cho", "với", "từ",
    "trên", "dưới", "giữa", "bên", "ngoài", "trong", "khi", "nếu", "còn", "vì", "bởi", "như",
    "ai", "gì", "nào", "thế", "sao", "không", "cũng", "chỉ", "vẫn", "được", "nữa", "này",
    "kia", "đâu", "ấy", "có", "lại", "rồi", "hơn", "về", "cả", "ấy", "còn", "dẫu", "hay",
    "dù", "nhỉ", "nhé", "ra", "vừa", "bao", "giờ", "chưa", "sẽ", "đi", "từng", "sao", "à", "ừ",
    "nhỉ", "nhé", "ừm", "ờ"
])

# Hàm chuẩn hóa văn bản
def preprocess_text(text, stopwords):
    # Chuyển về chữ thường
    text = text.lower()
    # Xóa emoji
    text = re.sub(r'[^\w\s,.!?]', '', text)
    # Xóa các biểu tượng đặc biệt như =)), =)
    text = re.sub(r'(\=\))|(\=\(|\:v)', '', text)
    # Xóa dấu câu
    text = re.sub(r'[^\w\s]', '', text)
    # Xóa số
    text = re.sub(r'\d+', '', text)
    # Loại bỏ stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    # Ghép lại thành văn bản đã xử lý
    return " ".join(filtered_words)

def replace_abbreviations(text, abbreviations):
    words = text.split()
    replace_words = [abbreviations.get(word, word) for word in words]
    return " ".join(replace_words)

def remove_random_words(text):
    text = re.sub(r'\b[a-zA-Z]{3,100}\b','',text)
    # chuẩn hóa các chữ lặp lại trong câu : vừaaaa -> vừa
    text = re.sub(r'\s+',' ',text).strip()
    text = re.sub(r'(.)\1{2,}' , r'\1', text)
    return text

# def remove_invalid_rows(row):
#     """
#     Loại bỏ các dòng chứa chuỗi ký tự ngẫu nhiên không hợp lệ
#     """
#     # Biểu thức regex: tìm các chuỗi có từ 10 ký tự trở lên liên tiếp, không chứa khoảng trắng
#     if re.fullmatch(r"[a-zA-Z0-9]{10,}", row):
#         return None  # Đánh dấu dòng này là không hợp lệ
#     return row  # Giữ lại dòng hợp lệ

# Hàm để loại bỏ các cụm từ lặp lại liên tiếp
def remove_repeated_phrases(text):
    """
    Loại bỏ các cụm từ lặp lại liên tiếp trong văn bản.
    Ví dụ: 'bền lâu trôi bền lâu trôi bền lâu trôi' -> 'bền lâu trôi'
    """
    if not text:
        return text
    # Regex để tìm các cụm từ lặp lại
    return re.sub(r'(\b\w+(?: \w+)*\b)(?: \1)+', r'\1', text)

def remove_nonsense_data(text):
    if not text:
        return text
    # Loại bỏ các chuỗi ngẫu nhiên dài hơn 8 ký tự
    if re.fullmatch(r'[a-zA-Z0-9]{8,}', text):  # Tìm chuỗi chỉ gồm chữ cái và số dài >= 8 ký tự
        return None  # Loại bỏ dòng này
    return text

def remove_invalid_rows_from_dataset(dataset):
    """
    Loại bỏ các dòng chứa chuỗi ngẫu nhiên không hợp lệ hoặc không có ý nghĩa
    """
    dataset = dataset[~dataset['comment'].str.fullmatch(r'[a-zA-Z0-9]{20,}')]  # Tìm các chuỗi chỉ có chữ cái và số dài >= 20 ký tự
    dataset = dataset.dropna(subset=['comment'])  # Loại bỏ dòng chứa None
    dataset = dataset[dataset['comment'].str.len()  >= 5]
    dataset = dataset[~dataset['comment'].str.contains(r'\bxu\b', case = False, na = False)]
    return dataset

# Hàm chuẩn hóa toàn bộ dữ liệu
def preprocess_dataset(dataset, stopwords, abbreviations):
    # chuẩn hóa từng comment

    dataset['comment'] = dataset['comment'].apply(lambda x : replace_abbreviations(x, abbreviations))
    dataset['comment'] = dataset['comment'].apply(lambda x : remove_random_words(x))
    # dataset['comment'] = dataset['comment'].apply(lambda x : remove_invalid_rows(x))
    dataset['comment'] = dataset['comment'].apply(lambda x : remove_repeated_phrases(x))

    dataset["comment"] = dataset["comment"].apply(lambda x : preprocess_text(x, stopwords))
    # Loại bỏ các dòng không có ý nghĩa
    dataset = remove_invalid_rows_from_dataset(dataset)

    # loại bỏ các dòng trùng lặp
    dataset = dataset.drop_duplicates(subset = ["comment"], keep = "first")

    # loại bỏ các dòng có nội dung rỗng sau khi xử lý
    dataset = dataset[dataset["comment"].str.strip() != ""]

    return dataset

# Đọc dữ liệu từ file CSV
path_file = "D:\\NLP_CVS\\pj_ML\\data\\comment.csv"  # Đường dẫn tới file dữ liệu của bạn

dataset = pd.read_csv(path_file)

# Kiểm tra xem có cột 'comment' không
if 'comment' not in dataset.columns:
    raise ValueError("Column 'comment' not found in the file")

# Định nghĩa từ viết tắt
abbreviations = {
    "tr" : "trời",
    "ng" : "ngon",
    "Ko" : "không",
    "ưng" : "vừa lòng",
    "r" : "rồi",
    "ak" : "à",
    "cx": "cũng",
    "đc": "được",
    "dc": "được",
    "sz": "size",
    "mik": "mình",
    "mn": "mọi người",
    "vs": "với",
    "tl": "trả lời",
    "rep": "trả lời",
    "ib": "nhắn tin",
    "nt": "nhắn tin",
    "k": "không",
    "K" : "không",
    "ko": "không",
    "hok": "không",
    "kh": "không",
    "bn": "bao nhiêu",
    "m": "mình",
    "mk": "mình",
    "ck": "chuyển khoản",
    "sll": "số lượng lớn",
    "tks": "cảm ơn",
    "thank": "cảm ơn",
    "thks": "cảm ơn",
    "tk": "tài khoản",
    "ms": "mã số",
    "nv": "nhân viên",
    "vc": "vận chuyển",
    "add": "địa chỉ",
    "ad": "admin",
    "fb": "facebook",
    "face": "facebook",
    "stk": "số tài khoản",
    "hn": "hà nội",
    "hcm": "hồ chí minh",
    "sg": "sài gòn",
    "qá": "quá",
    "qa": "quá",
    "p": "phòng",
    "sđt": "số điện thoại",
    "dt": "điện thoại",
    "sp": "sản phẩm",
    "thik": "thích",
    "iu": "yêu",
    "sz": "kích thước",
    "nv": "nhân viên",
    "hdsd": "hướng dẫn sử dụng",
    "hsd": "hạn sử dụng",
    "ship": "giao hàng",
    "síp": "giao hàng",
    "auth": "chính hãng",
    "aut": "chính hãng",
    "chx": "chưa",
    "zui": "vui",
    "buon": "buồn",
    "z": "gì",
    "qtrong": "quan trọng",
    "mag": "mang",
    "form": "dáng",
    "sd": "sử dụng",
    "hk": "không",
    "thg": "thường",
    "ord": "đặt hàng",
    "order": "đặt hàng",
    "del": "giao hàng",
    "sale": "giảm giá",
    "sl": "số lượng",
    "km": "khuyến mãi",
    "freesh": "miễn phí giao hàng",
    "fs": "miễn phí giao hàng",
    "cmt": "bình luận",
    "cfs": "confession",
    "bt": "bình thường",
    "ok": "đồng ý",
    "oki": "đồng ý",
    "oke": "đồng ý",
    "okie": "đồng ý",
    "pk": "phải không",
    "phk": "phải không",
    "lm": "làm",
    "qtqđ": "quá trời quá đất",
    "vl": "vãi lúa",
    "vcl": "vãi cả lúa",
    "vll": "vãi lúa luôn",
    "đg": "đang",
    "h": "giờ",
    "mng": "mọi người",
    "qt": "quốc tế",
    "qtv": "quản trị viên",
    "mod": "người điều hành",
    "fl": "theo dõi",
    "unfl": "hủy theo dõi",
    "tt": "tiếp tục",
    "addfr": "thêm bạn",
    "avt": "ảnh đại diện",
    "acc": "tài khoản",
    "full": "đầy đủ",
    "huhu": "khóc",
    "haha": "cười",
    "g9": "ngủ ngon",
    "ckn": "chúc ngủ ngon",
    "hihi": "vui vẻ",
    "idk": "tôi không biết",
    "gg": "gì cũng được",
    "msgh": "tin nhắn",
    "ppl": "mọi người",
    "irl": "ngoài đời thực",
    "imo": "theo tôi",
    "bff": "bạn thân",
    "bro": "anh trai",
    "sis": "chị em",
    "asap": "ngay lập tức",
    "omg": "ôi trời ơi",
    "wtf": "cái quái gì",
    "brb": "quay lại ngay",
    "bbiab": "tôi sẽ trở lại ngay",
    "lmao": "cười to",
    "fyi": "để bạn biết",
    "idc": "tôi không quan tâm",
    "j/k": "chỉ đùa thôi",
    "nvm": "không sao đâu",
    "pls": "làm ơn",
    "u": "bạn",
    "ur": "của bạn",
    "ty": "cảm ơn",
    "np": "không có chi",
    "hbd": "chúc mừng sinh nhật",
    "yolo": "bạn chỉ sống một lần",
    "qc": "quảng cáo",
    "cl": "chất lượng",
    "sp" : "sản phẩm",
}

# Tiến hành tiền xử lý dữ liệu
processed_dataset = preprocess_dataset(dataset, stopwords, abbreviations)
# Xáo trộn dataset
processed_dataset = processed_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
# Lưu dữ liệu sau khi xử lý vào file CSV
processed_dataset.to_csv("comment_cleaned.csv", index=False, encoding='utf-8-sig')

print("Dữ liệu đã được xử lý và lưu tại comment_cleaned.csv")
