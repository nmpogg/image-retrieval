import requests

API_URL = "https://LINK_NGROK_CUA_BAN/search" # Nhớ thêm /search

def search_by_image(image_path, model_name="resnet", top_k=5):
    print(f"\n🔍 [IMAGE SEARCH] Đang tìm ảnh giống '{image_path}' bằng {model_name.upper()}...")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            data = {
                "model_type": model_name,
                "query_type": "image",
                "top_k": top_k
            }
            response = requests.post(API_URL, files=files, data=data)
            _print_results(response)
    except Exception as e:
        print("Lỗi:", e)

def search_by_text(text_query, top_k=5):
    # Tìm bằng chữ thì BẮT BUỘC mô hình phải là CLIP
    print(f"\n📝 [TEXT SEARCH] Đang tìm ảnh với mô tả: '{text_query}'...")
    try:
        data = {
            "model_type": "clip",
            "query_type": "text",
            "text_query": text_query,
            "top_k": top_k
        }
        # Lưu ý: Tìm bằng text không cần gửi tham số 'files'
        response = requests.post(API_URL, data=data)
        _print_results(response)
    except Exception as e:
        print("Lỗi:", e)

def _print_results(response):
    if response.status_code == 200:
        res = response.json()
        if "error" in res:
            print("❌ Lỗi từ Server:", res["error"])
            return
            
        print(f"✅ Hoàn thành! (Dùng {res['model_used'].upper()}) - Top {len(res['top_k_results'])} kết quả:")
        for i, item in enumerate(res["top_k_results"]):
            print(f"   {i+1}. Độ giống: {item['similarity']} | File: {item['image_path']}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")
        print(response.text)


# 1. Thử tìm bằng ảnh với ResNet
# search_by_image("test_bird.jpg", model_name="resnet", top_k=3)

# 2. Thử tìm bằng ảnh với ViT
# search_by_image("test_bird.jpg", model_name="vit", top_k=3)

# 3. Thử tìm bằng câu chữ (Sẽ tự động gọi CLIP)
search_by_text("A yellow bird with black wings", top_k=5)