"""
Module Cài đặt và Tích hợp các Thuật toán Khai thác Tập phổ biến:
1. Apriori (mlxtend)
2. FP-Growth (mlxtend)
3. IT-Tree / Eclat (Tự cài đặt tối ưu)
"""

# Thư viện hệ thống của Python
import sys
# Thư viện đo lường thời gian thực thi hiệu năng cao
import time
# Cấu trúc từ điển tự động gán giá trị mặc định cho khóa mới
from collections import defaultdict
# Thư viện xử lý cấu trúc bảng DataFrame
import pandas as pd
# Hàm chạy thuật toán Apriori từ thư viện mlxtend
from mlxtend.frequent_patterns import apriori
# Hàm chạy thuật toán FP-Growth từ thư viện mlxtend
from mlxtend.frequent_patterns import fpgrowth

# Kiểm tra nếu console Windows chưa dùng bảng mã UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        # Cấu hình lại console sang UTF-8 để in tiếng Việt không bị lỗi
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Khởi tạo lớp thuật toán IT-Tree (Itemset-Tid Tree) dựa trên kỹ thuật biểu diễn dữ liệu dọc
class ITTreeMiner:
    # Hàm khởi tạo đối tượng lớp ITTreeMiner
    def __init__(self):
        # Khởi tạo từ điển lưu trữ các thông số đo lường hiệu năng của thuật toán
        self.stats = {
            # Số lượng nút trên cây tìm kiếm mà thuật toán đã duyệt qua
            'nodes_explored': 0,
            # Tổng số lượng tập phổ biến tìm được
            'total_frequent_itemsets': 0,
            # Tổng thời gian thực thi thuật toán tính bằng giây
            'execution_time': 0.0
        }

    # Định nghĩa phương thức khai thác tập phổ biến từ danh sách giao dịch
    def mine(self, transactions, min_support=0.01, max_len=None):
        # Bắt đầu bấm giờ với độ chính xác nano-giây
        start_time = time.perf_counter()
        
        # Đếm tổng số lượng giao dịch trong tập dữ liệu
        n_trans = len(transactions)
        
        # Chuyển đổi tỷ lệ min_support thành số lượng giao dịch tối thiểu tuyệt đối
        min_count = min_support * n_trans
        
        # Đặt lại số nút đã duyệt về 0 trước khi bắt đầu
        self.stats['nodes_explored'] = 0

        # Khởi tạo từ điển ánh xạ: Tên mặt hàng -> Tập hợp các mã giao dịch (TID set)
        tid_map = defaultdict(set)
        
        # Lặp qua từng giao dịch và chỉ số định danh tid tương ứng
        for tid, tx in enumerate(transactions):
            # Lặp qua từng mặt hàng có trong giao dịch hiện tại
            for item in tx:
                # Thêm mã giao dịch tid vào tập hợp của mặt hàng đó
                tid_map[item].add(tid)

        # Lọc các 1-itemset có số lượng giao dịch xuất hiện lớn hơn hoặc bằng min_count
        freq_1_items = {item: tids for item, tids in tid_map.items() if len(tids) >= min_count}
        
        # Sắp xếp các mặt hàng theo thứ tự tần số TĂNG DẦN (Support-ascending order) để tỉa nhánh DFS sớm
        sorted_items = sorted(freq_1_items.keys(), key=lambda k: len(freq_1_items[k]))

        # Khởi tạo danh sách lưu trữ toàn bộ các tập phổ biến tìm được
        results = []

        # Định nghĩa hàm đệ quy duyệt không gian tìm kiếm cây IT-Tree theo chiều sâu DFS
        def _dfs_extend(prefix_itemset, items_list, current_tid_map, current_depth):
            # Nếu người dùng giới hạn độ dài tối đa và độ sâu hiện tại vượt quá giới hạn thì dừng nhánh
            if max_len is not None and current_depth > max_len:
                # Kết thúc nhánh đệ quy hiện tại
                return

            # Duyệt qua từng mặt hàng ứng viên trong danh sách
            for i, item in enumerate(items_list):
                # Tăng biến đếm số lượng nút cây đã khám phá
                self.stats['nodes_explored'] += 1
                
                # Tạo tập mục mới bằng phép hợp tiền tố với mặt hàng hiện tại
                new_itemset = prefix_itemset | {item}
                
                # Lấy danh sách TID của tập mục mới
                item_tids = current_tid_map[item]
                
                # Tính toán giá trị độ hỗ trợ Support bằng số TID chia cho tổng số giao dịch
                support_val = len(item_tids) / n_trans
                
                # Thêm tập phổ biến vào danh sách kết quả (dùng frozenset để cố định tập hợp)
                results.append({
                    # Giá trị độ hỗ trợ
                    'support': support_val,
                    # Tập hợp các mặt hàng phổ biến
                    'itemsets': frozenset(new_itemset)
                })

                # Kiểm tra nếu chưa vượt quá độ dài tối đa max_len thì tiếp tục sinh các nút con
                if max_len is None or (current_depth + 1) <= max_len:
                    # Khởi tạo danh sách các mặt hàng con cho tầng tiếp theo
                    child_items_list = []
                    # Khởi tạo từ điển lưu trữ danh sách TID cho các nút con
                    child_tid_map = {}
                    
                    # Thử ghép mặt hàng hiện tại với các mặt hàng đứng sau nó trong thứ tự sắp xếp
                    for other_item in items_list[i + 1:]:
                        # THỰC HIỆN PHÉP TOÁN CỐT LÕI CỦA ECLAT: Giao hai tập hợp TID bằng toán tử '&'
                        intersect_tids = item_tids & current_tid_map[other_item]
                        
                        # Kiểm tra xem số lượng giao dịch chung có đạt ngưỡng min_count hay không
                        if len(intersect_tids) >= min_count:
                            # Thêm mặt hàng thỏa mãn vào danh sách con
                            child_items_list.append(other_item)
                            # Lưu tập TID giao nhau vào từ điển nhánh con
                            child_tid_map[other_item] = intersect_tids
                    
                    # Nếu có ít nhất một nút con thỏa mãn điều kiện min_support
                    if child_items_list:
                        # Gọi đệ quy để tiếp tục đi sâu xuống tầng tiếp theo của cây IT-Tree
                        _dfs_extend(new_itemset, child_items_list, child_tid_map, current_depth + 1)

        # Bắt đầu duyệt đệ quy từ gốc cây rỗng (tiền tố rỗng, độ sâu khởi điểm = 1)
        _dfs_extend(set(), sorted_items, freq_1_items, current_depth=1)

        # Đo lường tổng thời gian thực thi của thuật toán
        elapsed = time.perf_counter() - start_time
        
        # Ghi nhận thời gian chạy vào từ điển thống kê
        self.stats['execution_time'] = elapsed
        
        # Ghi nhận tổng số tập phổ biến tìm được vào từ điển thống kê
        self.stats['total_frequent_itemsets'] = len(results)

        # Chuyển đổi danh sách kết quả thành DataFrame
        df_res = pd.DataFrame(results)
        
        # Nếu không có tập phổ biến nào được tìm thấy
        if df_res.empty:
            # Trả về DataFrame rỗng có đúng 2 cột support và itemsets
            return pd.DataFrame(columns=['support', 'itemsets'])
        
        # Sắp xếp các tập phổ biến theo Support giảm dần và đặt lại chỉ số dòng
        df_res = df_res.sort_values(by='support', ascending=False).reset_index(drop=True)
        
        # Trả về DataFrame chứa toàn bộ tập phổ biến
        return df_res

# Định nghĩa hàm bọc chạy thuật toán Apriori trên ma trận thưa
def run_apriori(df_sparse, min_support=0.01, max_len=None):
    # Bắt đầu bấm giờ đo thời gian thực thi của Apriori
    start_time = time.perf_counter()
    
    # Gọi hàm apriori của mlxtend với chế độ tiết kiệm bộ nhớ low_memory=True
    res = apriori(df_sparse, min_support=min_support, use_colnames=True, max_len=max_len, low_memory=True)
    
    # Tính toán thời gian thực thi bằng giây
    elapsed = time.perf_counter() - start_time
    
    # Trả về bảng kết quả đã sắp xếp theo support giảm dần và thời gian chạy
    return res.sort_values(by='support', ascending=False).reset_index(drop=True), elapsed

# Định nghĩa hàm bọc chạy thuật toán FP-Growth trên ma trận thưa
def run_fpgrowth(df_sparse, min_support=0.01, max_len=None):
    # Bắt đầu bấm giờ đo thời gian thực thi của FP-Growth
    start_time = time.perf_counter()
    
    # Gọi hàm fpgrowth của mlxtend sử dụng cây tiền tố nén FP-Tree
    res = fpgrowth(df_sparse, min_support=min_support, use_colnames=True, max_len=max_len)
    
    # Tính toán thời gian thực thi bằng giây
    elapsed = time.perf_counter() - start_time
    
    # Trả về bảng kết quả đã sắp xếp theo support giảm dần và thời gian chạy
    return res.sort_values(by='support', ascending=False).reset_index(drop=True), elapsed

# Định nghĩa hàm bọc chạy thuật toán IT-Tree tự cài đặt
def run_ittree(transactions, min_support=0.01, max_len=None):
    # Khởi tạo một đối tượng thực thi ITTreeMiner
    miner = ITTreeMiner()
    
    # Gọi phương thức mine để khai thác tập phổ biến từ danh sách transactions
    res = miner.mine(transactions, min_support=min_support, max_len=max_len)
    
    # Trả về bảng kết quả, thời gian thực thi và số lượng nút cây đã khám phá
    return res, miner.stats['execution_time'], miner.stats['nodes_explored']
