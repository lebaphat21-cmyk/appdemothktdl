"""Demo đồ án: python -m streamlit run app/streamlit_app.py."""

import ast
import json
import sys
from html import escape
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms import run_apriori, run_fpgrowth, run_ittree
from src.data_preprocessing import get_onehot_sparse_df, load_transactions
from src.rule_mining import recommend_products

ALGORITHMS = {
    "FP-Growth": ("fpgrowth_time", "#2563EB"),
    "IT-Tree": ("ittree_time", "#059669"),
    "Apriori": ("apriori_time", "#DC2626"),
}
TABLES = PROJECT_ROOT / "outputs" / "tables"

st.set_page_config(page_title="Giỏ hàng bán lẻ | Nhóm 4", page_icon="🛒", layout="wide")


def parse_itemset(value):
    """Đọc tập mục trong CSV mà không thực thi biểu thức Python."""
    if isinstance(value, (set, frozenset)):
        return frozenset(value)
    text = str(value).strip()
    if text.startswith("frozenset(") and text.endswith(")"):
        text = text[10:-1]
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, (set, list, tuple)) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("Tập mục trong bảng luật không hợp lệ.")
    return frozenset(parsed)


def data_signature():
    """Tự làm mới cache khi các tệp đầu vào được cập nhật."""
    paths = list(TABLES.glob("*.csv")) + [TABLES / "data_summary.json"]
    paths += [PROJECT_ROOT / folder / "processed" / "transactions.pkl" for folder in ("data", "DT")]
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths if p.exists())


@st.cache_data(show_spinner="Đang tải dữ liệu đồ án…")
def load_cached_data(signature):
    summary_path = TABLES / "data_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}

    def read_table(name):
        path = TABLES / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    minsup = read_table("benchmark_minsup.csv")
    datasize = read_table("benchmark_datasize.csv")
    gold = read_table("gold_rules.csv")
    rules = read_table("filtered_rules.csv")
    if not rules.empty:
        for column in ("antecedents", "consequents"):
            rules[column] = rules[column].apply(parse_itemset)
    tx_path = next((PROJECT_ROOT / folder / "processed" / "transactions.pkl"
                    for folder in ("data", "DT")
                    if (PROJECT_ROOT / folder / "processed" / "transactions.pkl").exists()), None)
    transactions = load_transactions(tx_path) if tx_path else []
    items = sorted({item for tx in transactions for item in tx})
    if not rules.empty:
        items = sorted(set(items).union(*(rules["antecedents"].tolist() + rules["consequents"].tolist())))
    return summary, minsup, datasize, gold, rules, transactions, items


def runtime_chart(frame, x, title, x_title, log=False):
    fig = go.Figure()
    for name, (column, color) in ALGORITHMS.items():
        fig.add_trace(go.Scatter(
            x=frame[x], y=frame[column], name=name, mode="lines+markers",
            connectgaps=False,
            line=dict(color=color, width=3, dash="dash" if name == "Apriori" else "solid"),
            hovertemplate="%{y:.3f} giây<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title="Thời gian (giây)",
        yaxis_type="log" if log else "linear", hovermode="x unified",
        height=390, margin=dict(l=10, r=10, t=60, b=25),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def association_graph(rules):
    """Mỗi nút là một tập mục; mũi tên giữ nguyên chiều của luật."""
    graph = nx.DiGraph()
    for _, row in rules.iterrows():
        graph.add_edge(row["Antecedent_Text"], row["Consequent_Text"], lift=float(row["Lift"]))
    # Các luật đầu bảng thường thành nhiều cặp rời nhau; bố trí tròn tránh
    # các cụm và nhãn bị dồn vào nhau như khi dùng spring_layout.
    positions = nx.circular_layout(graph)
    fig = go.Figure()
    for source, target, attrs in graph.edges(data=True):
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        # Lệch hai chiều sang hai phía để các luật đối ứng không đè lên nhau.
        dx, dy = x1 - x0, y1 - y0
        offset = 0.035 if graph.has_edge(target, source) else 0
        sx, sy = x0 + .13 * dx - offset * dy, y0 + .13 * dy + offset * dx
        ex, ey = x0 + .85 * dx - offset * dy, y0 + .85 * dy + offset * dx
        fig.add_trace(go.Scatter(
            x=[sx, ex], y=[sy, ey], mode="lines", line=dict(color="#94A3B8", width=1.5),
            text=[f"{escape(source)} → {escape(target)}<br>Lift: {attrs['lift']:.2f}"] * 2,
            hovertemplate="%{text}<extra></extra>", showlegend=False,
        ))
        fig.add_annotation(x=ex, y=ey, ax=sx, ay=sy, xref="x", yref="y", axref="x", ayref="y",
                           text="", showarrow=True, arrowhead=3, arrowsize=1.3,
                           arrowwidth=1.5, arrowcolor="#64748B")
    nodes = list(graph.nodes())
    labels = [f"{i + 1}. {node[:18]}{'…' if len(node) > 18 else ''}" for i, node in enumerate(nodes)]
    fig.add_trace(go.Scatter(
        x=[positions[n][0] for n in nodes], y=[positions[n][1] for n in nodes],
        text=labels, customdata=[escape(n) for n in nodes], mode="markers+text", textposition="top center",
        textfont=dict(size=10), hovertemplate="%{customdata}<extra></extra>",
        marker=dict(color="#2563EB", size=15, line=dict(width=2, color="white")), showlegend=False,
    ))
    fig.update_layout(height=530, margin=dict(l=55, r=55, t=35, b=35),
                      xaxis=dict(visible=False, range=[-1.4, 1.4]),
                      yaxis=dict(visible=False, range=[-1.3, 1.3]), hovermode="closest")
    return fig


try:
    signature = data_signature()
    summary, df_minsup, df_datasize, df_gold, df_rules, transactions, all_items = load_cached_data(signature)
except Exception as exc:
    st.error("Không đọc được dữ liệu đồ án. Hãy kiểm tra các tệp dữ liệu và bảng kết quả.")
    with st.expander("Chi tiết lỗi"):
        st.code(str(exc))
    st.stop()

# Không giữ kết quả của bộ dữ liệu cũ khi tệp đầu vào thay đổi.
if st.session_state.get("data_signature") != signature:
    for key in ("live_result", "recommendation_result"):
        st.session_state.pop(key, None)
    st.session_state["data_signature"] = signature

with st.sidebar:
    st.title("🛒 Data Mining Lab")
    st.caption("ĐỀ TÀI 4 · NHÓM 4")
    st.write("So sánh Apriori, FP-Growth và IT-Tree trên dữ liệu bán lẻ.")
    st.divider()
    st.subheader("Dữ liệu sau làm sạch")
    for label, key in [("Dòng bán lẻ", "cleaned_rows"), ("Hóa đơn hợp lệ", "cleaned_bills"),
                       ("Mặt hàng", "cleaned_items")]:
        value = summary.get(key)
        st.write(f"**{label}:** {value:,}" if value is not None else f"**{label}:** Chưa có dữ liệu")
    if "avg_basket_size" in summary:
        st.write(f"**Giỏ hàng trung bình:** {summary['avg_basket_size']:.1f} món")
    st.divider()
    st.caption("Dùng các tab để xem thực nghiệm, thử giỏ hàng và khám phá luật mua kèm.")

st.title("Phân tích giỏ hàng bán lẻ")
st.write("Từ so sánh thuật toán đến khám phá các sản phẩm thường được mua cùng nhau.")
metrics = st.columns(3)
metrics[0].metric("Dòng dữ liệu gốc", f"{summary['initial_rows']:,}" if "initial_rows" in summary else "—")
metrics[1].metric("Giao dịch dùng khai thác", f"{len(transactions):,}" if transactions else "—")
metrics[2].metric("Luật sau bộ lọc", f"{len(df_rules):,}")
st.caption("Một dòng bán lẻ là một mục hàng; một giao dịch là một giỏ có ít nhất 2 mặt hàng.")
if not transactions:
    st.warning("Chưa có dữ liệu giao dịch. Vẫn có thể xem các bảng kết quả đã lưu; chạy thử đang tạm khóa.")
    with st.expander("Vị trí tệp giao dịch"):
        st.code("data/processed/transactions.pkl\nhoặc DT/processed/transactions.pkl")

tab1, tab2, tab3 = st.tabs(["📊 So sánh thuật toán", "🛒 Gợi ý mua kèm", "🏆 Top 20 luật vàng"])

with tab1:
    st.subheader("Kết quả thực nghiệm đã lưu")
    if not df_minsup.empty:
        options = sorted(df_minsup["minsup"].unique(), reverse=True)
        selected_support = st.selectbox("Ngưỡng để đọc kết quả", options,
                                        index=options.index(.02) if .02 in options else 0,
                                        format_func=lambda v: f"Minsup {v * 100:g}%", key="saved_support")
        row = df_minsup.loc[df_minsup["minsup"] == selected_support].iloc[0]
        times = {name: float(row[column]) for name, (column, _) in ALGORITHMS.items() if pd.notna(row[column])}
        if times:
            winner = min(times, key=times.get)
            st.success(f"Ở minsup {selected_support * 100:g}%, {winner} nhanh nhất trong các phép đo đã lưu: {times[winner]:.3f} giây.")
        time_cols = st.columns(3)
        for col, (name, (column, _)) in zip(time_cols, ALGORITHMS.items()):
            col.metric(name, f"{row[column]:.3f} s" if pd.notna(row[column]) else "Không chạy")
        st.caption("Kết luận chỉ áp dụng cho bộ dữ liệu, cách cài đặt và điều kiện đo của thí nghiệm này.")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(runtime_chart(df_minsup, "minsup_pct", "Thời gian theo minsup", "Minsup", log=True), width="stretch")
            st.caption("Trục thời gian dùng thang logarit. Đường thấp hơn tương ứng thời gian ngắn hơn.")
        with right:
            fig = px.bar(df_minsup, x="minsup_pct", y="itemsets_count", text_auto=",",
                         title="Số tập phổ biến theo minsup", color_discrete_sequence=["#2563EB"],
                         labels={"minsup_pct": "Minsup", "itemsets_count": "Số tập phổ biến"})
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=60, b=25))
            st.plotly_chart(fig, width="stretch")
        missing = df_minsup.loc[df_minsup["apriori_time"].isna(), "minsup_pct"].tolist()
        if missing:
            st.info(f"Apriori không được chạy tại {', '.join(missing)} theo điều kiện của pipeline. Điểm thiếu không phải kết quả đo tràn RAM hoặc hết thời gian.")
    else:
        st.info("Chưa có bảng thực nghiệm theo minsup.")
    if not df_datasize.empty:
        st.plotly_chart(runtime_chart(df_datasize, "fraction_pct", "Thời gian theo quy mô dữ liệu", "Tỷ lệ giao dịch"), width="stretch")
    else:
        st.info("Chưa có bảng thực nghiệm theo quy mô dữ liệu.")
    with st.expander("Cách đọc và điều kiện đo"):
        st.markdown("""
        - **Minsup:** tỷ lệ giỏ tối thiểu chứa một tập mặt hàng để tập đó được xem là phổ biến.
        - **Apriori:** mở rộng ứng viên theo từng độ dài. **FP-Growth:** khai thác cây tiền tố. **IT-Tree:** giao các tập mã giao dịch.
        - Biểu đồ quy mô dùng các phần đầu của danh sách giao dịch, với minsup cố định 2% theo pipeline hiện tại.
        - Thí nghiệm minsup có bật `tracemalloc`; thí nghiệm quy mô và chạy thử không bật. Không đối chiếu trực tiếp thời gian giữa hai chế độ đo.
        - Các số liệu là lần đo đã lưu, chưa có khoảng biến thiên từ nhiều lần lặp. Thời gian thuật toán không bao gồm bước tạo ma trận one-hot bên ngoài.
        """)
        if not df_minsup.empty:
            st.dataframe(df_minsup, hide_index=True, width="stretch")

    st.divider()
    st.subheader("Chạy thử trên máy hiện tại")
    st.caption("Dùng phần đầu danh sách giao dịch để thống nhất với pipeline thực nghiệm đã lưu.")
    cols = st.columns(3)
    sel_algos = cols[0].multiselect("Thuật toán", list(ALGORITHMS), default=["FP-Growth", "IT-Tree"], key="live_algos")
    test_minsup = cols[1].slider("Minsup (%)", 1.0, 5.0, 2.0, .5, key="live_support") / 100
    test_pct = cols[2].slider("Tỷ lệ giao dịch (%)", 10, 50, 20, 10, key="live_pct") / 100
    if st.button("Chạy thực nghiệm", type="primary", disabled=not transactions, key="run_benchmark"):
        if not sel_algos:
            st.warning("Chọn ít nhất một thuật toán trước khi chạy.")
        else:
            n = max(1, int(len(transactions) * test_pct))
            sample = transactions[:n]
            rows = []
            with st.spinner(f"Đang đo trên {n:,} giao dịch…"):
                matrix = None
                for name in sel_algos:
                    try:
                        if name == "IT-Tree":
                            result, elapsed, _ = run_ittree(sample, min_support=test_minsup)
                        else:
                            if matrix is None:
                                matrix, _ = get_onehot_sparse_df(sample)
                            runner = run_fpgrowth if name == "FP-Growth" else run_apriori
                            result, elapsed = runner(matrix, min_support=test_minsup)
                        rows.append({"Thuật toán": name, "Thời gian (giây)": round(elapsed, 4),
                                     "Số tập phổ biến": len(result), "Trạng thái": "Hoàn tất"})
                    except Exception as exc:
                        rows.append({"Thuật toán": name, "Thời gian (giây)": None,
                                     "Số tập phổ biến": None, "Trạng thái": f"Lỗi: {exc}"})
            st.session_state.live_result = dict(rows=rows, n=n, support=test_minsup, pct=test_pct, algorithms=list(sel_algos))
    if "live_result" in st.session_state:
        saved = st.session_state.live_result
        st.markdown("**Kết quả lần chạy gần nhất**")
        st.caption(f"{saved['n']:,} giao dịch · Minsup {saved['support'] * 100:g}% · {', '.join(saved['algorithms'])}")
        if (saved["support"], saved["pct"], saved["algorithms"]) != (test_minsup, test_pct, sel_algos):
            st.info("Tham số đã thay đổi. Bấm Chạy thực nghiệm để cập nhật kết quả.")
        st.dataframe(pd.DataFrame(saved["rows"]), hide_index=True, width="stretch")

with tab2:
    st.subheader("Thử một giỏ hàng")
    st.write("Chọn sản phẩm để tìm các món mua kèm. Gợi ý được xếp theo điểm **Lift × Confidence** của luật phù hợp.")
    defaults = [item for item in ["GREEN REGENCY TEACUP AND SAUCER", "ROSES REGENCY TEACUP AND SAUCER"] if item in all_items]
    selected_items = st.multiselect("Sản phẩm trong giỏ", all_items, default=defaults, key="cart")
    top_k = st.slider("Số gợi ý tối đa", 1, 8, 4, key="top_k")
    if df_rules.empty:
        st.info("Chưa có bảng luật đã lọc để tạo gợi ý.")
    if st.button("Tìm sản phẩm mua kèm", type="primary", disabled=df_rules.empty, key="recommend"):
        if not selected_items:
            st.warning("Hãy chọn ít nhất một sản phẩm trong giỏ.")
        else:
            st.session_state.recommendation_result = dict(
                cart=list(selected_items), top_k=top_k,
                recs=recommend_products(selected_items, df_rules, top_k=top_k),
            )
    if "recommendation_result" in st.session_state:
        saved = st.session_state.recommendation_result
        st.markdown("**Gợi ý lần gần nhất**")
        st.caption("Giỏ đã dùng: " + " + ".join(saved["cart"]) + f" · Tối đa {saved['top_k']} gợi ý")
        if set(saved["cart"]) != set(selected_items) or saved["top_k"] != top_k:
            st.info("Giỏ hàng hoặc số gợi ý đã thay đổi. Bấm Tìm sản phẩm mua kèm để cập nhật.")
        if not saved["recs"]:
            st.info("Chưa có luật phù hợp với giỏ này trong bảng đã lọc. Hãy thử giỏ mẫu gồm các tách trà Regency.")
        for index, rec in enumerate(saved["recs"], 1):
            with st.container(border=True):
                st.markdown(f"**{index}. {rec['item']}**")
                st.caption("Luật kích hoạt: " + " + ".join(sorted(rec["based_on"])))
                st.write(f"**Lift:** {rec['lift']:.2f}× · **Confidence:** {rec['confidence']:.1%} · **Support:** {rec['support']:.2%}")
    with st.expander("Các chỉ số có ý nghĩa gì?"):
        st.write("Support là tỷ lệ giao dịch chứa toàn bộ tập mục của luật. Confidence là tỷ lệ giao dịch có vế phải trong các giao dịch đã có vế trái. Lift so sánh tỷ lệ này với tỷ lệ xuất hiện chung của vế phải.")
        st.caption("Các chỉ số mô tả liên kết trong dữ liệu lịch sử, không chứng minh tác động kích cầu hoặc mức tăng doanh thu. Với luật có nhiều món ở vế phải, chỉ số thuộc về cả luật.")

with tab3:
    st.subheader("🏆 Top 20 luật vàng & Đồ thị mạng lưới")
    st.caption("Các luật đã lọc trong bộ kết quả, xếp theo Lift giảm dần.")
    if df_gold.empty:
        st.info("Chưa có bảng luật nổi bật để hiển thị.")
    else:
        search_kw = st.text_input("Tìm sản phẩm trong luật", placeholder="Ví dụ: REGENCY hoặc TEACUP", key="rule_search")
        shown = df_gold.sort_values("Lift", ascending=False)
        if search_kw.strip():
            shown = shown[shown["Rule_Display"].str.contains(search_kw.strip(), case=False, regex=False, na=False)]
        st.caption(f"{len(shown)} / {len(df_gold)} luật")
        if shown.empty:
            st.info("Không có luật khớp từ khóa. Hãy thử tên ngắn hơn hoặc xóa từ khóa.")
        else:
            display = shown[["Rule_Display", "Support_%", "Confidence_%", "Lift"]].rename(columns={
                "Rule_Display": "Luật kết hợp", "Support_%": "Support (%)", "Confidence_%": "Confidence (%)"})
            st.dataframe(display, hide_index=True, width="stretch")
            st.subheader("Mạng lưới liên kết")
            st.caption("Mũi tên A → B biểu diễn chiều của luật. Mỗi nút là một tập mặt hàng; di chuột để xem tên đầy đủ. Đồ thị dùng tối đa 12 luật khớp bộ lọc.")
            st.plotly_chart(association_graph(shown.head(12)), width="stretch")

