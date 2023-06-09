# BN-HTR_YOLO: Line/Word Segmentation for Bangla Handwritting Recognition using YOLOv5.
### This is a Demo of our Custom YOLOv5 Line/Word Segmentation Models for Bangla Handwritings using Streamlit

## [Try Out Live Demo](https://bn-htr-yolo.streamlit.app/):
- By **Default** the selected model will be *Downloaded Automatically* from our [HuggingFace Model Hub](https://huggingface.co/crusnic/BN-DRISHTI/tree/main/models).
- You can also use your own model (.pt) file through **File Upload** option while running the demo.
 <br>**OR**</br>
- You can paste the models **Link** directly to the **URL** option, for example:
  - Line Model: `https://huggingface.co/crusnic/BN-DRISHTI/resolve/main/models/line_model_best.pt`
  - Word Model: `https://huggingface.co/crusnic/BN-DRISHTI/resolve/main/models/word_model_best.pt`
  

## Features
- **Caches** the model for faster inference on both CPU and GPU.
- Supports uploading model files (<200MB).
- Supports both CPU and GPU inference.
- Supports:
  - Custom Classes
  - Changing Confidence


## How to run it on your local machine
1. Clone the Repo
```bash
git clone https://github.com/shaoncsecu/BN-HTR_YOLO.git
cd BN-HTR_YOLO
```
2. Install requirements
   - `pip install -r requirements.txt`
3. Add sample images to `data/sample_documents` or `data/sample_lines'
4. Add the model file to `models/` and change `cfg_model_path` to its path.
5. Run the application
   - `streamlit run app.py`
