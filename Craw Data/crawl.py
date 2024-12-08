import os
import re
from selenium import webdriver
import time
import csv
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Nhập dictionary chứa các số sao và label từ người dùng
star_label_dict = {}
print("Nhập số sao và label tương ứng (ví dụ: 1: Negative, 2: Negative, 3: Negative, 4: Positive, 5: Positive). Gõ 'done' khi hoàn tất:")

while True:
    entry = input()
    if entry.lower() == 'done':
        break
    try:
        star, label = entry.split(':')
        star_label_dict[int(star.strip())] = label.strip()
    except ValueError:
        print("Vui lòng nhập theo định dạng: <số sao>: <label>")

# Cài đặt Chrome Options
chrome_options = Options()
chrome_options.add_argument("--disable-gpu")  # Tắt GPU acceleration
chrome_options.add_argument("--disable-software-rasterizer")

# Khởi tạo trình duyệt với các tùy chọn đã thêm
driver = webdriver.Chrome(options=chrome_options)

# Mở trang sản phẩm Lazada
driver.get("https://www.lazada.vn/products/mut-chum-ruot-trai-lon-1kg-100-khong-chat-bao-quanmau-tu-nhien-vi-chua-ngot-i476092030-s909132862.html?c=&channelLpJumpArgs=&clickTrackInfo=query%253Am%2525E1%2525BB%2525A9t%252Btr%2525C3%2525A1i%252Bc%2525C3%2525A2y%253Bnid%253A476092030%253Bsrc%253ALazadaMainSrp%253Brn%253Ab4950c95c62d18efd71539d1d372a33d%253Bregion%253Avn%253Bsku%253A476092030_VNAMZ%253Bprice%253A66000%253Bclient%253Adesktop%253Bsupplier_id%253A1000049605%253Bbiz_source%253Ah5_internal%253Bslot%253A6%253Butlog_bucket_id%253A470687%253Basc_category_id%253A10003475%253Bitem_id%253A476092030%253Bsku_id%253A909132862%253Bshop_id%253A308276%253BtemplateInfo%253A107883_D_E%2523-1_A3_C%2523&freeshipping=1&fs_ab=2&fuse_fs=&lang=en&location=Vietnam&price=6.6E%204&priceCompare=skuId%3A909132862%3Bsource%3Alazada-search-voucher%3Bsn%3Ab4950c95c62d18efd71539d1d372a33d%3BoriginPrice%3A66000%3BdisplayPrice%3A66000%3BsinglePromotionId%3A900000035036605%3BsingleToolCode%3ApromPrice%3BvoucherPricePlugin%3A0%3Btimestamp%3A1732696083713&ratingscore=4.792916666666667&request_id=b4950c95c62d18efd71539d1d372a33d&review=2400&sale=9172&search=1&source=search&spm=a2o4n.searchlist.list.6&stock=1")
time.sleep(10)
print("Trang đã tải xong.")

# Đường dẫn tới file CSV
csv_file_path = 'comment.csv'

# Kiểm tra nếu file tồn tại và có nội dung
file_exists = os.path.exists(csv_file_path)

# Mở file CSV ở chế độ ghi tiếp
with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file, quoting=csv.QUOTE_NONE, escapechar='\\')

    # Chỉ ghi header nếu file chưa tồn tại hoặc trống
    if not file_exists or os.stat(csv_file_path).st_size == 0:
        csv_writer.writerow(['comment', 'label'])

    # Cuộn xuống để tải thêm nội dung
    scroll_amount = 1200
    for _ in range(2):
        driver.execute_script(f"window.scrollTo(0, {scroll_amount})")
        scroll_amount += 1200
        time.sleep(5)

    for star, label in star_label_dict.items():
        try:
            # Click vào nút lọc đánh giá để mở danh sách các bộ lọc sao
            print(f"Đang tìm nút lọc đánh giá {star} sao...")
            filter_button = driver.find_element(By.CSS_SELECTOR, "svg.lazadaicon.lazada-icon.svgfont.oper-icon")

            # Sử dụng JavaScript để click vào phần tử SVG
            driver.execute_script("""
                var element = arguments[0];
                var event = document.createEvent('MouseEvents');
                event.initEvent('click', true, true);
                element.dispatchEvent(event);
                """, filter_button)
            time.sleep(5)
            print("Đã click vào nút lọc đánh giá.")

            # Tìm và click vào bộ lọc đánh giá tương ứng
            filters = driver.find_elements(By.CSS_SELECTOR, "li.next-menu-item[role='menuitem']")
            for filter_element in filters:
                if f"{star} Sao" in filter_element.text:
                    filter_element.click()
                    time.sleep(5)
                    print(f"Đã chọn bộ lọc {star} sao.")
                    break
        except NoSuchElementException:
            print(f"Không tìm thấy bộ lọc cho {star} sao.")
            continue

        # Lặp lại để lấy tất cả các bình luận với bộ lọc hiện tại
        while True:
            try:
                # Đợi cho đến khi các bình luận xuất hiện
                WebDriverWait(driver, 5).until(
                    EC.presence_of_all_elements_located((By.CLASS_NAME, "item"))
                )
                time.sleep(5)

                # Tìm và lấy số sao của bình luận và nội dung bình luận
                review_items = driver.find_elements(By.CLASS_NAME, "item")

                for item in review_items:
                    try:
                        # Lấy nội dung bình luận
                        comment_element = item.find_element(By.CLASS_NAME, "content")
                        if comment_element:
                            comment = comment_element.text.strip()

                            # Sử dụng biểu thức chính quy để loại bỏ dấu câu
                            clean_comment = re.sub(r'[^\w\s]', '', comment)
                            clean_comment = re.sub(r'\s+', ' ', clean_comment).strip()

                            # Kiểm tra nếu comment không rỗng
                            if clean_comment:
                                # Ghi vào file CSV
                                print(f"Comment: {clean_comment} - Label: {label}")
                                csv_writer.writerow([clean_comment, label])

                    except NoSuchElementException:
                        continue

                # Tìm nút "Trang tiếp theo" và kiểm tra trạng thái của nó
                try:
                    next_button = driver.find_element(By.CSS_SELECTOR, 'button.next-btn.next-btn-normal.next-btn-medium.next-pagination-item.next')
                    if next_button.get_attribute('disabled'):
                        # Nếu nút next bị vô hiệu hóa, có nghĩa là đã đến trang cuối cùng
                        print("Đã tới trang cuối cùng")
                        break

                    # Click vào nút "Trang tiếp theo"
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(3)  # Đợi trang tiếp theo tải xong
                    print("Đã chuyển sang trang tiếp theo.")

                except NoSuchElementException:
                    print("Không tìm thấy nút trang tiếp theo. Đã tới trang cuối cùng.")
                    break

            except Exception as e:
                print(f"Đã xảy ra lỗi: {e}")
                break

# Đóng trình duyệt
print("Đang đóng trình duyệt.")
driver.quit()
'''
1 : Negative
2 : Negative
3 : Negative
4 : Normal
5 : Positive
done
'''