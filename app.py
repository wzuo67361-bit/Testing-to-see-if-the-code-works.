import io
import gc
import logging
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageEnhance
from rapidocr_onnxruntime import RapidOCR

st.set_page_config(page_title="二手报价单极速工作台", page_icon="📱", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; color: #212529; }
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 100%; }
    .stButton>button { background-color: #0d6efd !important; color: white !important; font-weight: bold !important; }
</style>
""", unsafe_allow_html=True)

# 加载 ONNX 轻量化引擎，内存占用仅 ~150MB
@st.cache_resource(show_spinner="正在加载 ONNX 超轻量高精度引擎...")
def load_ocr_engine():
    return RapidOCR()

def enhance_image(img):
    img_gray = img.convert('L')
    return ImageEnhance.Sharpness(ImageEnhance.Contrast(img_gray).enhance(2.0)).enhance(2.0).convert('RGB')

def process_dynamic_columns(engine, img, y_tolerance, x_tolerance):
    w, h = img.size
    slice_height = 1800   
    overlap = 100          
    
    y_starts = list(range(0, h, slice_height - overlap))
    total_slices = len(y_starts)
    
    boxes_texts = []
    progress_bar = st.progress(0, text="准备开始切割长图...")
    
    for i, y_start in enumerate(y_starts):
        y_end = min(y_start + slice_height, h)
        if y_start >= h: break
        
        progress_bar.progress((i) / total_slices, text=f"🔍 正在切片扫描 ({i+1}/{total_slices})...")
        
        slice_img = img.crop((0, y_start, w, y_end))
        img_array = np.array(enhance_image(slice_img))
        
        # ONNX 引擎推理
        result, _ = engine(img_array)
        
        if result:
            for line in result:
                box, text = line[0], line[1]
                cx = sum([p[0] for p in box]) / 4
                cy = sum([p[1] for p in box]) / 4 + y_start 
                boxes_texts.append({'text': str(text).strip(), 'x': cx, 'y': cy})
        
        del slice_img, img_array, result
        gc.collect() 

    progress_bar.progress(0.9, text="✨ 扫描完毕，正在重建动态表格...")

    if not boxes_texts: return pd.DataFrame()

    unique_boxes = []
    for item in boxes_texts:
        if not any(u['text'] == item['text'] and abs(item['x']-u['x'])<20 and abs(item['y']-u['y'])<20 for u in unique_boxes):
            unique_boxes.append(item)

    unique_boxes.sort(key=lambda x: x['y'])
    rows_list, current_row = [], []
    for item in unique_boxes:
        if not current_row or abs(item['y'] - sum(b['y'] for b in current_row)/len(current_row)) < y_tolerance:
            current_row.append(item)
        else:
            rows_list.append(current_row)
            current_row = [item]
    if current_row: rows_list.append(current_row)
        
    all_x = sorted([item['x'] for item in unique_boxes])
    col_centers, curr_col = [], []
    for x in all_x:
        if not curr_col or abs(x - sum(curr_col)/len(curr_col)) < x_tolerance: 
            curr_col.append(x)
        else:
            col_centers.append(sum(curr_col)/len(curr_col))
            curr_col = [x]
    if curr_col: col_centers.append(sum(curr_col)/len(curr_col))
        
    table_data = []
    for row in rows_list:
        row_data = [""] * len(col_centers)
        for item in row:
            closest_idx = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - item['x']))
            row_data[closest_idx] = (row_data[closest_idx] + " " + item['text']).strip()
        table_data.append(row_data)

    if table_data:
        raw_cols = table_data[0]
        unique_cols, seen = [], {}
        for i, col in enumerate(raw_cols):
            col_str = str(col).strip() or f"未命名列_{i+1}"
            if col_str in seen:
                seen[col_str] += 1
                unique_cols.append(f"{col_str}_{seen[col_str]}")
            else:
                seen[col_str] = 0
                unique_cols.append(col_str)
        df = pd.DataFrame(table_data[1:], columns=unique_cols) if len(table_data)>1 else pd.DataFrame(columns=unique_cols)
    else:
        df = pd.DataFrame()
        
    progress_bar.progress(1.0, text="✅ 动态表格重建完成！")
    return df

st.markdown("### 📱 报价单精准提取工作台 `ONNX 极致防爆内存版`")

uploaded_file = st.file_uploader("📂 上传报价单图片 (支持高度几千像素的极限长图)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    st.markdown("#### ⚙️ 动态列匹配设置")
    col1, col2, col3 = st.columns([3, 3, 4])
    with col1:
        y_tol = st.slider("↕️ 行距容差 (防止断行)", 5, 50, 15, 1)
    with col2:
        x_tol = st.slider("↔️ 列距容差 (列数越多调越小)", 5, 100, 20, 5) 
    with col3:
        st.write("") 
        start_button = st.button("🚀 启动云端极速解析", use_container_width=True)

    if start_button:
        try:
            engine = load_ocr_engine()
            raw_df = process_dynamic_columns(engine, original_image, y_tol, x_tol)
            
            raw_df = raw_df.fillna("").astype(str)
            raw_df.columns = [str(c) for c in raw_df.columns]
            
            st.session_state['ocr_df'] = raw_df
            st.success(f"✅ 成功！自动识别出 {len(raw_df.columns)} 列，共 {raw_df.shape[0]} 行数据。")
        except Exception as e:
            st.error(f"❌ 识别过程中出现错误：{str(e)}")

    if 'ocr_df' in st.session_state and not st.session_state['ocr_df'].empty:
        df = st.session_state['ocr_df']
        
        st.markdown("#### 📝 数据导出与预览")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='报价明细')
        
        st.download_button(
            label="📥 立即导出为 Excel 文件 (.xlsx)",
            data=output.getvalue(),
            file_name="精准动态排版报价单.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("##### 🔍 网页预览（支持左右横向滚动）")
        
        st.dataframe(df, use_container_width=True, height=500)
