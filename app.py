import streamlit as st
import requests
import os
from PIL import Image

# Cấu hình trang
st.set_page_config(page_title="CUB-200 Bird Search Engine", layout="wide")

st.title("🦅 Hệ thống Truy xuất Ảnh CUB-200")
st.markdown("---")

# --- PHẦN 1: CẤU HÌNH SERVER ---
with st.sidebar:
    st.header("Cài đặt kết nối")
    api_base_url = st.text_input("Dán link Ngrok vào đây:", placeholder="https://xxx-xxx.ngrok-free.app")
    st.info("Lưu ý: Đảm bảo server Colab đang chạy.")

    st.header("Tham số tìm kiếm")
    model_type = st.selectbox("Chọn mô hình:", ["ConvNeXt", "DINOv2", "Combined", "SigLIP"])
    top_k = st.slider("Số lượng kết quả (Top K):", 1, 20, 5)

# --- PHẦN 2: CHỌN PHƯƠNG THỨC TÌM KIẾM ---
query_type = "image"
if model_type == "SigLIP":
    query_type = st.radio("Phương thức tìm kiếm:", ["Tìm bằng ảnh (Image-to-Image)", "Tìm bằng chữ (Text-to-Image)"], horizontal=True)
    query_type = "image" if "ảnh" in query_type else "text"
else:
    st.info(f"Mô hình {model_type} hiện chỉ hỗ trợ tìm kiếm bằng ảnh.")

# Giao diện nhập Query
col1, col2 = st.columns([1, 2])

query_file = None
text_input = ""

with col1:
    if query_type == "image":
        query_file = st.file_uploader("Tải ảnh chim cần tìm:", type=["jpg", "jpeg", "png"])
        if query_file:
            st.image(query_file, caption="Ảnh bạn đã tải lên", width='stretch')
    else:
        text_input = st.text_input("Nhập mô tả con chim:", placeholder="Ví dụ: A yellow bird with black wings")

# --- PHẦN 3: XỬ LÝ TRUY XUẤT ---
if st.button("🔍 Bắt đầu tìm kiếm"):
    if not api_base_url:
        st.error("Vui lòng nhập link Server Ngrok ở thanh bên trái!")
    else:
        api_url = f"{api_base_url}/search"
        
        # Chuẩn bị dữ liệu gửi đi (Form-data)
        data = {
            "model_type": model_type.lower(),
            "query_type": query_type,
            "top_k": top_k,
            "text_query": text_input
        }
        
        files = None
        if query_type == "image" and query_file:
            files = {"file": (query_file.name, query_file.getvalue(), "image/jpeg")}

        try:
            with st.spinner("Đang kết nối tới AI Server..."):
                response = requests.post(api_url, data=data, files=files)
            
            if response.status_code == 200:
                res = response.json()
                if "error" in res:
                    st.error(f"Lỗi từ AI: {res['error']}")
                else:
                    results = res.get("top_k_results", [])
                    st.success(f"Đã tìm thấy {len(results)} ảnh tương đồng!")
                    
                    # Hiển thị kết quả dạng lưới (Grid)
                    cols = st.columns(3)
                    for idx, item in enumerate(results):
                        with cols[idx % 3]:
                            # Giả sử bạn để dataset ở cùng thư mục app.py trên máy local
                            # Đường dẫn trả về từ server là: 'images/001.Black_footed_Albatross/...'
                            # Bạn cần trỏ đúng đường dẫn tuyệt đối trên máy bạn
                            local_img_path = os.path.join("dataset/CUB/CUB_200_2011", "images", item['image_path'])
                            
                            try:
                                img = Image.open(local_img_path)
                                st.image(img, caption=f"Top {idx+1} - Độ giống: {item['similarity']}", width='stretch')
                                st.caption(f"Path: {item['image_path']}")
                            except:
                                st.warning(f"Không tìm thấy file ảnh local: {item['image_path']}")
            else:
                st.error(f"Lỗi kết nối Server (HTTP {response.status_code})")
        except Exception as e:
            st.error(f"Không thể kết nối tới Server. Hãy kiểm tra lại link Ngrok! \nChi tiết: {e}")