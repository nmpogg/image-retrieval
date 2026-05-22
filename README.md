# 🦅 CUB-200 Fine Grained Image Retrieval

A comprehensive image retrieval system built on the **CUB-200-2011** fine-grained bird dataset, supporting both image-to-image and text-to-image retrieval across multiple deep learning architectures.

![Image-to-Image Retrieval](image2image.png)
*Image-to-image retrieval — query image (left) with top-k results (right).*

![Text-to-Image Retrieval](text2image.png)
*Text-to-image retrieval — natural language query with top-k results (right).*

---

## Features

- **Image-to-Image Retrieval** — find visually similar birds given a query image
- **Text-to-Image Retrieval** — retrieve images from natural language descriptions via CLIP-based models
- **Multi-architecture Benchmark** — evaluate ResNet-50, EfficientNet, ConvNeXt, ViT, Swin-V2, DINOv2, CLIP, SigLIP, SigLIP2, and more
- **Auto-generated Captions** — VLM-generated descriptions covering appearance, behavior, and background
- **Streamlit Demo** — lightweight web UI with REST API backend

---

## Repository Structure

```
image-retrieval/
├── app.py                  # Streamlit demo client (UI)
├── test.py                 # Script to test the /search endpoint
├── cubdataset.py           # CUB-200-2011 data loader → (image, label, caption)
├── model/
│   ├── clip.py             # CLIP-based image/text encoder + contrastive loss
│   ├── resnet.py           # ResNet backbone
│   └── vit.py              # ViT backbone
├── notebooks/
│   ├── clip-base/          # Fine-tuning & evaluation notebooks for CLIP variants
│   ├── cnn-base/           # Fine-tuning & evaluation notebooks for CNN backbones
│   └── vit-base/           # Fine-tuning & evaluation notebooks for ViT backbones
└── weights/                # Pretrained checkpoints & adapter weights
    ├── best_resnet50_cub.pth
    ├── clip_adapter/
    ├── vit_adapter/
    └── siglip_adapter/
```

---

## 🚀 Quick Start

### 1. Create Environment and Install Dependencies

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
```

Install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org), then install the remaining dependencies:

```bash
pip install pandas pillow scikit-learn scikit-image tqdm timm \
            transformers ftfy tokenizers safetensors streamlit \
            requests matplotlib
```

### 2. Prepare the Dataset

Download **CUB-200-2011** from the [official source](http://www.vision.caltech.edu/datasets/cub_200_2011/) and place it so the directory structure matches:

```
dataset/CUB/CUB_200_2011/
├── images/
├── images.txt
├── image_class_labels.txt
└── train_test_split.txt
```

For captions, place per-image `.txt` files at:

```
dataset/CUB/captions/<class_folder>/<image_name>.txt
```

> **Alternatively**, use the pre-generated captions produced with **Qwen2.5-2B-VL**, available on Kaggle:
> 📦 [nmpogg/cub-caption](https://www.kaggle.com/datasets/nmpogg/cub-caption)

### 3. Load Pretrained Weights

Place `.pth` or `.safetensors` checkpoint files inside `weights/`. Adapter subfolders (`clip_adapter/`, `vit_adapter/`, `siglip_adapter/`, etc.) are already structured for direct loading.

### 4. Run the Streamlit Demo

```bash
streamlit run app.py
```

`app.py` communicates with a backend server exposing a `POST /search` endpoint. Paste your backend URL (e.g., an ngrok tunnel) into the sidebar input in the browser.

### 5. Test the API Endpoint

Edit `API_URL` in `test.py` to point to your backend (`<ngrok_url>/search`), then run:

```bash
python test.py
```

---

## API Reference

### `POST /search`

| Field | Type | Description |
|---|---|---|
| `model_type` | `string` | Backbone name: `resnet`, `vit`, `clip`, `convnext`, `siglip`, ... |
| `query_type` | `string` | `image` or `text` |
| `top_k` | `int` | Number of results to return |
| `file` | `file` | Query image *(required when `query_type=image`)* |
| `text_query` | `string` | Description string *(required when `query_type=text`)* |

**Example response:**

```json
{
  "model_used": "convnext",
  "top_k_results": [
    { "image_path": "001.Black_footed_Albatross/001.jpg", "similarity": 0.9123 },
    { "image_path": "001.Black_footed_Albatross/004.jpg", "similarity": 0.8971 }
  ]
}
```

> `app.py` resolves returned paths relative to `dataset/CUB/CUB_200_2011/images/`. Adjust this prefix if your images are stored elsewhere.

---

## Benchmark Results

### Image-to-Image Retrieval — Precision@k

| Model | P@1 | P@3 | P@5 | P@10 | P@20 |
|---|---|---|---|---|---|
| ResNet-50 | 62.62 | 58.21 | 54.84 | 48.90 | 40.20 |
| EfficientNet-B4 | 70.81 | 67.20 | 64.43 | 59.23 | 50.55 |
| ConvNeXt-Base | 82.08 | 80.33 | 78.75 | 75.35 | 68.83 |
| ViT-B/16 | 81.83 | 80.71 | 79.71 | 77.47 | 73.21 |
| Swin-V2-B | 76.23 | 72.71 | 69.81 | 64.17 | 55.03 |
| DINOv2-B | 87.00 | 85.09 | 83.67 | 80.64 | 74.23 |
| **ConvNeXt + DINOv2** | **87.73** | **85.23** | **83.68** | 80.47 | 74.11 |

### Text-to-Image Retrieval — CLIP-based (zero-shot)

| Model | R@1 | R@5 | R@10 | CRS@1 | MR |
|---|---|---|---|---|---|
| CLIP ViT-B/16 | **1.44** | **5.85** | **10.35** | **94.62** | **4.86** |
| SigLIP-B/16 | 1.36 | 5.59 | 9.97 | 92.21 | 10.49 |
| SigLIP2-B/16 | 1.36 | 5.42 | 9.52 | 92.73 | 7.74 |

> CRS@k measures semantic similarity between the text query and the captions of returned images — a model-based proxy for retrieval quality when no hard ground-truth labels exist.

---

## Tips

- **GPU recommended** for fine-tuning and fast evaluation.
- **Feature caching** — precompute and save embeddings for the full gallery to accelerate nearest-neighbor search at query time.
- **FAISS indexing** — for large-scale or production deployments, build a FAISS/Annoy index over cached embeddings for sub-second retrieval.
- Select the correct PyTorch + CUDA version at [pytorch.org](https://pytorch.org) *before* installing other packages to avoid conflicts.

---

## Roadmap

- [ ] Fine-tune CLIP-based models on CUB-200 captions (currently zero-shot only)
- [ ] FAISS / Annoy index builder script for fast large-scale retrieval
- [ ] Composed retrieval — accept both an image and a text modifier as a joint query
- [ ] Absolute URL responses from the backend to remove client-side path mapping
- [ ] SAM-based foreground cropping to reduce background noise in embeddings

---

## License

This project is for research and educational purposes. CUB-200-2011 dataset usage is subject to its [original license](http://www.vision.caltech.edu/datasets/cub_200_2011/).

---

## Contact

For questions about the dataset, training setup, or deployment, please open an issue or reach out directly.