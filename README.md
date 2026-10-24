# App khai thác giỏ hàng bán lẻ — Nhóm 4

Thư mục này chứa mã nguồn và dữ liệu cần thiết để triển khai app Streamlit,
bao gồm tab **Top 20 luật vàng**, gợi ý mua kèm và so sánh thuật toán.

## Đưa lên Streamlit Community Cloud

1. Tạo một repository GitHub và tải **toàn bộ nội dung bên trong thư mục `up`**
   vào thư mục gốc của repository. Giữ nguyên các thư mục con; không chỉ tải riêng file Python.
2. Vào https://share.streamlit.io, chọn **Create app** và chọn repository, branch đã tải lên.
3. Điền **Main file path:** `app/streamlit_app.py`.
4. Trong **Advanced settings**, chọn **Python 3.12**, là phiên bản đã dùng để kiểm tra trên máy.
5. Chọn **Deploy**. Các thư viện được khai báo trong `requirements.txt`.

Nếu bạn đưa nguyên thư mục `up` vào một repository có sẵn thay vì đưa nội dung
của nó vào gốc, Main file path sẽ là `up/app/streamlit_app.py`. Bản sao
`app/requirements.txt` đã được đặt cạnh entrypoint để hỗ trợ cách bố trí này.

Community Cloud triển khai từ repository GitHub; không tải thư mục trực tiếp lên trang Streamlit.

## Cấu trúc cần giữ nguyên

```text
app/
  streamlit_app.py
  requirements.txt
src/
  __init__.py
  algorithms.py
  data_preprocessing.py
  rule_mining.py
data/processed/
  transactions.pkl
outputs/tables/
  benchmark_minsup.csv
  benchmark_datasize.csv
  data_summary.json
  filtered_rules.csv
  gold_rules.csv
requirements.txt
README.md
```

Không cần tải môi trường `.venv`, notebook, báo cáo Word, slide hoặc dữ liệu CSV gốc.
File `transactions.pkl` đã chứa các giao dịch để chạy thử và tạo danh sách sản phẩm;
các bảng CSV/JSON chứa kết quả đã lưu. Không cần chạy lại pipeline trước khi mở app.

## Chạy trên máy khác

Tại thư mục chứa `requirements.txt`:

```shell
python -m pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py
```

Benchmark trực tiếp chạy trên máy chủ khi triển khai, vì vậy thời gian có thể khác
kết quả đo trên máy cá nhân. Các biểu đồ kết quả đã lưu vẫn dùng số liệu đi kèm.

Tài liệu chính thức:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
