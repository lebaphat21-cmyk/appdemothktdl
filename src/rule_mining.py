"""
Module Khai thác và Lọc Luật kết hợp (Association Rules) & Hệ thống gợi ý mua kèm (Recommender System).
"""

# Thư viện hệ thống của Python
import sys
# Thư viện xử lý và định dạng cấu trúc dữ liệu bảng DataFrame
import pandas as pd
# Thư viện hỗ trợ tính toán mảng và số học
import numpy as np
# Hàm sinh luật kết hợp từ bảng tập phổ biến của thư viện mlxtend
from mlxtend.frequent_patterns import association_rules

# Kiểm tra nếu console Windows chưa dùng bảng mã UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        # Cấu hình console xuất chuẩn UTF-8 để hiển thị tiếng Việt có dấu
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Định nghĩa hàm sinh toàn bộ các luật kết hợp từ bảng tập phổ biến
def mine_association_rules(frequent_itemsets_df, min_confidence=0.3, min_lift=1.2):
    # Nếu bảng tập phổ biến truyền vào bị rỗng thì trả về DataFrame rỗng ngay
    if frequent_itemsets_df.empty:
        # Trả về DataFrame rỗng
        return pd.DataFrame()

    # Gọi hàm association_rules của mlxtend để tính toán tất cả các chỉ số của luật
    rules = association_rules(
        # Bảng chứa các tập phổ biến
        frequent_itemsets_df,
        # Chỉ số dùng để lọc ban đầu là độ tin cậy Confidence
        metric='confidence',
        # Ngưỡng độ tin cậy tối thiểu P(Y|X)
        min_threshold=min_confidence,
        # Cung cấp tổng số tập phổ biến để tính toán chính xác chỉ số
        num_itemsets=len(frequent_itemsets_df)
    )
    
    # Nếu không có luật nào đạt ngưỡng độ tin cậy thì trả về DataFrame rỗng
    if rules.empty:
        # Trả về DataFrame rỗng
        return pd.DataFrame()

    # Tạo thêm cột đếm số lượng món hàng ở vế trái (Antecedent)
    rules['antecedent_len'] = rules['antecedents'].apply(len)
    
    # Tạo thêm cột đếm số lượng món hàng ở vế phải (Consequent)
    rules['consequent_len'] = rules['consequents'].apply(len)
    
    # Lọc sơ bộ chỉ giữ lại các luật có độ kích cầu Lift >= min_lift
    rules = rules[rules['lift'] >= min_lift]
    
    # Sắp xếp các luật theo thứ tự ưu tiên: Lift giảm dần -> Confidence giảm dần -> Support giảm dần
    rules = rules.sort_values(by=['lift', 'confidence', 'support'], ascending=[False, False, False]).reset_index(drop=True)
    
    # Trả về bảng chứa toàn bộ các luật kết hợp đã sinh
    return rules

# Định nghĩa bộ lọc luật thông minh 3 tầng loại bỏ luật ngẫu nhiên và luật dư thừa
def filter_valuable_rules(rules_df, min_lift=1.5, max_redundant=True, drop_near_independent=True):
    # Nếu bảng luật đầu vào rỗng thì trả về DataFrame rỗng
    if rules_df.empty:
        # Trả về DataFrame rỗng
        return pd.DataFrame()

    # Tạo bản sao của bảng luật để không làm thay đổi dữ liệu gốc
    filtered = rules_df.copy()

    # TẦNG 1: Kiểm tra xem có bật tùy chọn loại bỏ luật độc lập ngẫu nhiên hay không
    if drop_near_independent:
        # Loại bỏ các luật có Lift nằm trong khoảng [0.85, 1.15] vì hai vế mua cùng nhau do ngẫu nhiên
        filtered = filtered[~((filtered['lift'] >= 0.85) & (filtered['lift'] <= 1.15))]

    # TẦNG 2: Lọc các luật có độ kích cầu thực sự vượt trội với Lift >= min_lift
    filtered = filtered[filtered['lift'] >= min_lift]

    # TẦNG 3: Kiểm tra điều kiện khử luật dư thừa khi bảng còn từ 2 luật trở lên
    if max_redundant and len(filtered) > 1:
        # Khởi tạo danh sách lưu trữ chỉ số các dòng luật giá trị không bị dư thừa
        non_redundant_indices = []
        
        # Chuyển đổi DataFrame sang danh sách từ điển các bản ghi để duyệt với tốc độ cao
        rules_list = filtered.to_dict('records')
        
        # Duyệt qua từng luật r1 trong danh sách để kiểm tra tính dư thừa
        for i, r1 in enumerate(rules_list):
            # Khởi tạo cờ đánh dấu luật r1 ban đầu là không dư thừa
            is_redundant = False
            
            # Lấy tập hợp vế trái a1 và vế phải c1 của luật r1
            a1, c1 = set(r1['antecedents']), set(r1['consequents'])
            
            # Lấy giá trị độ tin cậy của luật r1
            conf1 = r1['confidence']
            
            # So sánh luật r1 với tất cả các luật r2 khác trong cơ sở tri thức
            for j, r2 in enumerate(rules_list):
                # Không so sánh luật với chính nó
                if i != j:
                    # Lấy tập hợp vế trái a2 và vế phải c2 của luật r2
                    a2, c2 = set(r2['antecedents']), set(r2['consequents'])
                    
                    # Lấy giá trị độ tin cậy của luật r2
                    conf2 = r2['confidence']
                    
                    # Kiểm tra điều kiện dư thừa: Cùng vế phải, vế trái a2 là tập con thực sự của a1, và độ tin cậy tương đương
                    if c1 == c2 and a2.issubset(a1) and a2 != a1 and conf2 >= (conf1 - 0.05):
                        # Đánh dấu luật r1 bị dư thừa bởi luật ngắn gọn hơn r2
                        is_redundant = True
                        # Dừng vòng lặp so sánh và bỏ qua luật r1
                        break
            
            # Nếu sau khi kiểm tra mà luật r1 không bị luật nào làm dư thừa
            if not is_redundant:
                # Thêm chỉ số của luật r1 vào danh sách các luật giữ lại
                non_redundant_indices.append(i)
        
        # Trích xuất lại các dòng luật không dư thừa dựa trên danh sách chỉ số
        filtered = filtered.iloc[non_redundant_indices].copy()

    # Sắp xếp lại bảng luật đã lọc sạch theo Lift và Confidence giảm dần
    return filtered.sort_values(by=['lift', 'confidence'], ascending=[False, False]).reset_index(drop=True)

# Định nghĩa hàm trích xuất Top N luật vàng tiêu biểu phục vụ hiển thị báo cáo
def extract_gold_rules(rules_df, top_n=20):
    # Nếu bảng luật rỗng thì trả về DataFrame rỗng
    if rules_df.empty:
        # Trả về DataFrame rỗng
        return pd.DataFrame()

    # Lấy ra top N luật đứng đầu bảng có Lift cao nhất
    top_rules = rules_df.head(top_n).copy()
    
    # Nối các phần tử trong tập vế trái bằng dấu cộng thân thiện: ví dụ "Sản phẩm A + Sản phẩm B"
    top_rules['Antecedent_Text'] = top_rules['antecedents'].apply(lambda s: " + ".join(sorted(list(s))))
    
    # Nối các phần tử trong tập vế phải bằng dấu cộng
    top_rules['Consequent_Text'] = top_rules['consequents'].apply(lambda s: " + ".join(sorted(list(s))))
    
    # Tạo chuỗi hiển thị biểu diễn quy tắc: [Vế trái] ===> [Vế phải]
    top_rules['Rule_Display'] = top_rules['Antecedent_Text'] + "  ===>  " + top_rules['Consequent_Text']
    
    # Chuyển đổi độ hỗ trợ sang dạng phần trăm và làm tròn 2 chữ số thập phân
    top_rules['Support_%'] = (top_rules['support'] * 100).round(2)
    
    # Chuyển đổi độ tin cậy sang dạng phần trăm và làm tròn 2 chữ số thập phân
    top_rules['Confidence_%'] = (top_rules['confidence'] * 100).round(2)
    
    # Làm tròn giá trị độ kích cầu Lift tới 2 chữ số thập phân
    top_rules['Lift'] = top_rules['lift'].round(2)
    
    # Danh mục các cột cần giữ lại để hiển thị trên bảng
    cols_to_keep = ['Rule_Display', 'Support_%', 'Confidence_%', 'Lift', 'Antecedent_Text', 'Consequent_Text']
    
    # Trả về bảng kết quả chỉ gồm các cột cần thiết đã reset chỉ số dòng
    return top_rules[cols_to_keep].reset_index(drop=True)

# Định nghĩa thuật toán gợi ý sản phẩm mua kèm dựa trên giỏ hàng hiện tại của khách
def recommend_products(cart_items, rules_df, top_k=5):
    # Nếu giỏ hàng rỗng hoặc cơ sở luật rỗng thì không có gì để gợi ý
    if not cart_items or rules_df.empty:
        # Trả về danh sách rỗng
        return []

    # Chuyển giỏ hàng hiện tại thành set để kiểm tra quan hệ tập con với tốc độ O(1)
    cart_set = set(cart_items)
    
    # Khởi tạo từ điển lưu trữ các sản phẩm được đề xuất
    recommendations = {}

    # Duyệt qua từng luật kết hợp có trong cơ sở dữ liệu luật
    for _, rule in rules_df.iterrows():
        # Lấy tập hợp các mặt hàng ở vế trái của luật
        ant = set(rule['antecedents'])
        
        # Lấy tập hợp các mặt hàng ở vế phải của luật
        con = set(rule['consequents'])
        
        # Kiểm tra xem giỏ hàng của khách có chứa toàn bộ các món ở vế trái hay không
        if ant.issubset(cart_set):
            # Lấy các món ở vế phải mà khách hàng CHƯA có trong giỏ hàng hiện tại
            suggest_items = con - cart_set
            
            # Duyệt qua từng mặt hàng được đề xuất
            for item in suggest_items:
                # Tính toán điểm số đề xuất: Tích số giữa độ kích cầu Lift và độ tin cậy Confidence
                score = rule['lift'] * rule['confidence']
                
                # Nếu mặt hàng này chưa có trong danh sách hoặc luật này cho điểm số cao hơn
                if item not in recommendations or score > recommendations[item]['score']:
                    # Cập nhật thông tin chi tiết cho món hàng đề xuất
                    recommendations[item] = {
                        # Tên mặt hàng được gợi ý
                        'item': item,
                        # Độ tin cậy của luật
                        'confidence': rule['confidence'],
                        # Độ kích cầu của luật
                        'lift': rule['lift'],
                        # Độ hỗ trợ của luật
                        'support': rule['support'],
                        # Danh sách các mặt hàng trong giỏ là nguyên nhân kích hoạt gợi ý
                        'based_on': list(ant),
                        # Điểm số xếp hạng đề xuất
                        'score': score
                    }

    # Sắp xếp danh sách các sản phẩm gợi ý theo điểm số score giảm dần
    sorted_recs = sorted(recommendations.values(), key=lambda x: x['score'], reverse=True)
    
    # Trả về tối đa top_k sản phẩm gợi ý tốt nhất
    return sorted_recs[:top_k]
