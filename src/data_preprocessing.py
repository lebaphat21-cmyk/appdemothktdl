"""
Module Tiền xử lý dữ liệu cho Đề tài 4: Market Basket Analysis (>520.000 dòng).
"""

# Thư viện tương tác với hệ thống tệp tin và thư mục của hệ điều hành
import os
# Thư viện can thiệp các tham số cấu hình runtime của trình thông dịch Python
import sys
# Thư viện đọc và xuất dữ liệu định dạng JSON
import json
# Thư viện tuần tự hóa và lưu trữ đối tượng Python nhị phân
import pickle
# Thư viện xử lý và thao tác cấu trúc dữ liệu bảng DataFrame
import pandas as pd
# Thư viện hỗ trợ tính toán mảng và số học đa chiều
import numpy as np
# Bộ mã hóa danh sách giỏ hàng thành ma trận nhị phân One-hot của mlxtend
from mlxtend.preprocessing import TransactionEncoder

# Kiểm tra xem console hiện tại có phải là chuẩn mã hóa UTF-8 hay không
if sys.stdout.encoding != 'utf-8':
    try:
        # Ép buộc console xuất ký tự theo chuẩn UTF-8 để hiển thị tiếng Việt không bị lỗi font
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Khởi tạo tập hợp danh sách các mục chi phí hệ thống không phải là hàng hóa bán lẻ
NON_PRODUCT_ITEMS = {
    # Cước phí bưu điện chuyển phát thông thường
    'POSTAGE',
    # Cước phí bưu điện chuyển phát đơn hàng trực tuyến
    'DOTCOM POSTAGE',
    # Phí chiết khấu dịch vụ sàn thương mại điện tử Amazon
    'AMAZON FEE',
    # Hàng mẫu gửi kèm dùng thử cho khách hàng
    'SAMPLES',
    # Bút toán điều chỉnh hóa đơn thủ công của kế toán
    'MANUAL',
    # Mã chiết khấu khuyến mãi trực tiếp trên hóa đơn
    'DISCOUNT',
    # Chi phí phát sinh qua cổng thanh toán ngân hàng
    'BANK CHARGES',
    # Tiền hoa hồng quyên góp quỹ từ thiện
    'CRUK COMMISSION',
    # Bút toán điều chỉnh số lượng tồn kho nội bộ
    'ADJUST',
    # Bút toán xử lý xóa sổ nợ khó đòi kế toán
    'ADJUST BAD DEBT',
    # Chi phí cước vận tải đường bộ hàng cồng kềnh
    'CARRIAGE'
}

# Định nghĩa hàm thực hiện toàn bộ pipeline làm sạch và tạo giỏ hàng giao dịch
def clean_and_prepare_data(raw_csv_path='data/raw/Assignment-1_Data.csv',
                           processed_csv_path='data/processed/cleaned_retail.csv',
                           transactions_path='data/processed/transactions.pkl',
                           summary_json_path='outputs/tables/data_summary.json',
                           min_items_per_tx=2):
    
    # In thông báo đường dẫn file thô đang được tiến hành nạp vào bộ nhớ
    print(f"[*] Đang nạp dữ liệu thô từ: {raw_csv_path}...")
    
    # Đọc tệp CSV thô với phân cách chấm phẩy, số thực dùng dấu phẩy, tắt cảnh báo kiểu dữ liệu
    df_raw = pd.read_csv(raw_csv_path, sep=';', decimal=',', low_memory=False)
    
    # Đếm tổng số lượng dòng bản ghi ban đầu trong tập dữ liệu thô
    initial_rows = len(df_raw)
    
    # Đếm tổng số lượng mã hóa đơn duy nhất xuất hiện trong dữ liệu thô
    initial_bills = df_raw['BillNo'].nunique()
    
    # Đếm tổng số lượng tên mặt hàng duy nhất trong dữ liệu thô
    initial_items = df_raw['Itemname'].nunique()

    # In ra màn hình số bản ghi thô ban đầu
    print(f"    - Tổng số bản ghi thô: {initial_rows:,}")
    
    # In ra màn hình số lượng hóa đơn ban đầu
    print(f"    - Số hóa đơn ban đầu: {initial_bills:,}")
    
    # In ra màn hình số lượng mặt hàng ban đầu
    print(f"    - Số mặt hàng ban đầu: {initial_items:,}")

    # Tạo bản sao dữ liệu chỉ giữ lại các dòng có tên mặt hàng Itemname không bị rỗng
    df = df_raw[df_raw['Itemname'].notnull()].copy()
    
    # Lọc bỏ các dòng có số lượng mua nhỏ hơn hoặc bằng 0 (đơn hủy, hàng lỗi trả lại kho)
    df = df[df['Quantity'] > 0]
    
    # Lọc bỏ các dòng có đơn giá nhỏ hơn hoặc bằng 0 (hàng tặng hoặc lỗi kế toán)
    df = df[df['Price'] > 0]
    
    # Chuyển đổi mã hóa đơn sang kiểu chuỗi và loại bỏ khoảng trắng thừa ở hai đầu
    df['BillNo'] = df['BillNo'].astype(str).str.strip()
    
    # Sử dụng Regex lọc chỉ giữ lại các hóa đơn gồm toàn chữ số, loại bỏ đơn hủy 'C' và nợ 'A'
    df = df[df['BillNo'].str.match('^[0-9]+$')]
    
    # Chuyển toàn bộ tên mặt hàng thành chữ in hoa và xóa khoảng cách đầu cuối
    df['Itemname'] = df['Itemname'].astype(str).str.strip().str.upper()
    
    # Thay thế các khoảng trắng liên tiếp bên trong tên mặt hàng bằng đúng một dấu cách đơn
    df['Itemname'] = df['Itemname'].str.replace(r'\s+', ' ', regex=True)
    
    # Lọc loại bỏ tất cả các dòng có tên mặt hàng nằm trong danh mục phí phi sản phẩm
    df = df[~df['Itemname'].isin(NON_PRODUCT_ITEMS)]
    
    # Loại bỏ các dòng trùng lặp cùng một mặt hàng trong cùng một số hóa đơn
    df = df.drop_duplicates(subset=['BillNo', 'Itemname'])
    
    # Đếm số lượng dòng dữ liệu hợp lệ sau khi hoàn thành toàn bộ bước làm sạch
    cleaned_rows = len(df)
    
    # Đếm số lượng mã hóa đơn hợp lệ còn lại sau làm sạch
    cleaned_bills = df['BillNo'].nunique()
    
    # Đếm số lượng mặt hàng duy nhất còn lại sau khi chuẩn hóa
    cleaned_items = df['Itemname'].nunique()
    
    # In thông báo tổng kết kết quả làm sạch dữ liệu
    print("\n[*] Kết quả làm sạch dữ liệu:")
    
    # In số lượng bản ghi sạch và số bản ghi rác đã được loại bỏ
    print(f"    - Số bản ghi hợp lệ: {cleaned_rows:,} (giảm {initial_rows - cleaned_rows:,} bản ghi)")
    
    # In số lượng hóa đơn hợp lệ
    print(f"    - Số hóa đơn hợp lệ: {cleaned_bills:,}")
    
    # In số lượng mặt hàng sạch
    print(f"    - Số mặt hàng chuẩn hóa: {cleaned_items:,}")

    # Tự động tạo thư mục chứa file dữ liệu sạch nếu thư mục đó chưa tồn tại
    os.makedirs(os.path.dirname(processed_csv_path), exist_ok=True)
    
    # Xuất toàn bộ DataFrame dữ liệu sạch ra tệp tin CSV với chuẩn mã hóa UTF-8
    df.to_csv(processed_csv_path, index=False, encoding='utf-8')
    
    # In đường dẫn tệp tin CSV dữ liệu sạch đã được lưu thành công
    print(f"    -> Đã lưu dữ liệu sạch vào: {processed_csv_path}")

    # In thông báo bắt đầu quá trình tạo giỏ hàng giao dịch
    print("\n[*] Đang tạo tập hợp Transaction (Giỏ hàng) theo BillNo...")
    
    # Gom nhóm theo mã hóa đơn BillNo và gộp tất cả các Itemname thành một danh sách các món
    grouped = df.groupby('BillNo')['Itemname'].apply(list)
    
    # Chỉ giữ lại các giỏ hàng có từ min_items_per_tx mặt hàng trở lên (tối thiểu 2 món để sinh luật)
    valid_transactions = [tx for tx in grouped if len(tx) >= min_items_per_tx]
    
    # Tạo danh sách ghi nhận độ dài (số lượng món hàng) của từng giỏ hàng hợp lệ
    tx_lengths = [len(tx) for tx in valid_transactions]
    
    # In số lượng giỏ hàng hợp lệ đủ điều kiện khai thác luật kết hợp
    print(f"    - Tổng số giao dịch >= {min_items_per_tx} món: {len(valid_transactions):,}")
    
    # In số lượng mặt hàng trung bình có trong mỗi giỏ hàng
    print(f"    - Kích thước giỏ hàng trung bình: {np.mean(tx_lengths):.2f} món")
    
    # In số lượng mặt hàng trong giỏ hàng lớn nhất
    print(f"    - Kích thước giỏ hàng lớn nhất: {np.max(tx_lengths)} món")
    
    # In giá trị trung vị số lượng mặt hàng trong giỏ hàng
    print(f"    - Trung vị kích thước giỏ hàng: {np.median(tx_lengths):.1f} món")

    # Mở tệp tin nhị phân ở chế độ ghi để lưu trữ danh sách các giỏ hàng
    with open(transactions_path, 'wb') as f:
        # Ghi toàn bộ danh sách valid_transactions vào tệp pickle
        pickle.dump(valid_transactions, f)
        
    # In đường dẫn tệp transactions.pkl đã lưu
    print(f"    -> Đã lưu danh sách transactions vào: {transactions_path}")

    # Tạo thư mục outputs/tables nếu chưa có sẵn
    os.makedirs(os.path.dirname(summary_json_path), exist_ok=True)
    
    # Khởi tạo từ điển chứa các chỉ số thống kê phục vụ vẽ bảng trong báo cáo
    summary = {
        # Tổng số dòng bản ghi thô ban đầu
        'initial_rows': int(initial_rows),
        # Tổng số hóa đơn thô ban đầu
        'initial_bills': int(initial_bills),
        # Tổng số mặt hàng thô ban đầu
        'initial_items': int(initial_items),
        # Số dòng bản ghi sạch giữ lại
        'cleaned_rows': int(cleaned_rows),
        # Số hóa đơn sạch giữ lại
        'cleaned_bills': int(cleaned_bills),
        # Số mặt hàng chuẩn hóa giữ lại
        'cleaned_items': int(cleaned_items),
        # Tổng số giỏ hàng hợp lệ có từ 2 món trở lên
        'total_transactions_usable': int(len(valid_transactions)),
        # Kích thước giỏ hàng trung bình
        'avg_basket_size': float(np.mean(tx_lengths)),
        # Trung vị kích thước giỏ hàng
        'median_basket_size': float(np.median(tx_lengths)),
        # Kích thước giỏ hàng lớn nhất
        'max_basket_size': int(np.max(tx_lengths)),
        # Ngưỡng lọc số món tối thiểu của một giỏ
        'min_items_filter': int(min_items_per_tx)
    }
    
    # Mở tệp JSON ở chế độ ghi với mã hóa UTF-8
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        # Xuất từ điển thống kê ra file JSON với thụt đầu dòng 4 khoảng trắng
        json.dump(summary, f, indent=4, ensure_ascii=False)
        
    # In thông báo đã lưu thành công tệp tóm tắt dữ liệu
    print(f"    -> Đã lưu thông số thống kê vào: {summary_json_path}")

    # Trả về 3 đối tượng: DataFrame dữ liệu sạch, danh sách giỏ hàng và từ điển thống kê
    return df, valid_transactions, summary

# Định nghĩa hàm nạp danh sách giỏ hàng đã xử lý từ file pickle
def load_transactions(transactions_path='data/processed/transactions.pkl'):
    # Mở file pickle ở chế độ đọc nhị phân
    with open(transactions_path, 'rb') as f:
        # Đọc và giải tuần tự hóa trả về danh sách các giỏ hàng
        return pickle.load(f)

# Định nghĩa hàm chuyển đổi giỏ hàng sang ma trận thưa One-hot CSR Matrix
def get_onehot_sparse_df(transactions):
    # Khởi tạo đối tượng TransactionEncoder của thư viện mlxtend
    te = TransactionEncoder()
    
    # Quét toàn bộ tập giao dịch và chuyển đổi trực tiếp sang ma trận thưa scipy csr_matrix
    te_ary = te.fit(transactions).transform(transactions, sparse=True)
    
    # Bọc ma trận thưa dưới dạng pandas DataFrame kiểu Sparse để tiết kiệm 98% dung lượng RAM
    df_sparse = pd.DataFrame.sparse.from_spmatrix(te_ary, columns=te.columns_)
    
    # Trả về ma trận thưa và bộ mã hóa TransactionEncoder
    return df_sparse, te

# Kiểm tra nếu tệp này được chạy trực tiếp từ dòng lệnh
if __name__ == '__main__':
    # Gọi hàm thực thi toàn bộ pipeline làm sạch và xuất dữ liệu
    clean_and_prepare_data()
