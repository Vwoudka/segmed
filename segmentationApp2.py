import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import os
import tempfile
from skimage.transform import resize
import matplotlib.pyplot as plt
import io
import zipfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import zipfile

# --- Configuration from Training Script (Defaults) ---
DEFAULT_IN_CHANNELS = 4
DEFAULT_OUT_CLASSES = 4
DEFAULT_BASE_FEATURES = 32
TARGET_HW_SHAPE = (128, 128)
START_SLICE = 25
END_SLICE = 155
TARGET_DEPTH = END_SLICE - START_SLICE

# --- Label to RGBA Color Mapping ---
LABEL_TO_RGBA = {
    0: (0, 0, 0, 0),
    1: (255, 0, 0, 255),
    2: (0, 255, 0, 255),
    3: (255, 255, 0, 255),
}

SEGMENTATION_LABELS_DICT = {
    1: "Necrotic",
    2: "Edema",
    3: "Enhancing",
}

# --- Translation Dictionary ---
TRANSLATIONS = {
    "English": {
        "title": "🧠 SegMed, a 3D Brain tumour segmentation website",
        "description": """
        Upload a pre-trained 3D U-Net model and NIfTI modalities.
        The app performs segmentation and offers two download options:
        1. A standard **NIfTI label map** (`.nii.gz`).
        2. A **ZIP archive of PNG images** for each of the 130 processed slices.
        **Ensure your uploaded model weights match the U-Net architecture defined here.**
        """,
        "sidebar_header": "⚙️ Configuration",
        "patient_id": "Patient Name/ID",
        "unet_config": "U-Net Architecture",
        "input_channels": "Input Channels (C)",
        "output_classes": "Output Classes (Total, incl. background)",
        "base_features": "Base Features",
        "upload_model": "Upload 3D U-Net Model (.pth)",
        "pretrained_model": "Pretrained Model",
        "load_pretrained": "Load Pretrained Model from GitHub",
        "running_on": "Running on",
        "input_files": "📁 Input NIfTI Files",
        "modality_names": ["T1-native (t1n)", "T1-contrast (t1c)", "T2-FLAIR (t2f)", "T2-weighted (t2w)"],
        "run_button": "🚀 Run 3D Segmentation",
        "results_header": "📊 Segmentation Results",
        "multi_view": "Multi-View Segmentation Overlay",
        "legend_header": "Segmentation Legend",
        "download_header": "💾 Download Options",
        "nifti_option": "1. NIfTI Label Map",
        "download_nifti": "Download Label Segmentation (.nii.gz)",
        "png_option": "2. PNG Slices (Overlay)",
        "prepare_png": "Prepare PNG Slices for Download ",
        "download_png": "Download PNG Slices for {} (.zip)",
        "labels": {
            "Background": "Background (Normal Tissue)",
            "Necrotic": "Necrotic/Non-Enhancing Tumor",
            "Edema": "Edema",
            "Enhancing": "Enhancing Tumor"
        }
    },
    "Español": {
        "title": "🧠 SegMed, un sitio web de segmentación 3D de tumores cerebrales",
        "description": """
        Suba un modelo U-Net 3D preentrenado y modalidades NIfTI.
        La aplicación realiza la segmentación y ofrece dos opciones de descarga:
        1. Un **mapa de etiquetas NIfTI** estándar (`.nii.gz`).
        2. Un **archivo ZIP de imágenes PNG** para cada una de las 130 rebanadas procesadas.
        **Asegúrese de que los pesos del modelo cargado coincidan con la arquitectura U-Net definida aquí.**
        """,
        "sidebar_header": "⚙️ Configuración",
        "patient_id": "Nombre/ID del Paciente",
        "unet_config": "Arquitectura U-Net",
        "input_channels": "Canales de Entrada (C)",
        "output_classes": "Clases de Salida (Total, incl. fondo)",
        "base_features": "Características Base",
        "upload_model": "Subir Modelo U-Net 3D (.pth)",
        "pretrained_model": "Modelo Preentrenado",
        "load_pretrained": "Cargar Modelo Preentrenado desde GitHub",
        "running_on": "Ejecutando en",
        "input_files": "📁 Archivos NIfTI de Entrada",
        "modality_names": ["T1-nativo (t1n)", "T1-contraste (t1c)", "T2-FLAIR (t2f)", "T2-ponderado (t2w)"],
        "run_button": "🚀 Ejecutar Segmentación 3D",
        "results_header": "📊 Resultados de Segmentación",
        "multi_view": "Superposición de Segmentación Multivista",
        "legend_header": "Leyenda de Segmentación",
        "download_header": "💾 Opciones de Descarga",
        "nifti_option": "1. Mapa de Etiquetas NIfTI",
        "download_nifti": "Descargar Segmentación de Etiquetas (.nii.gz)",
        "png_option": "2. Rebanadas PNG (Superposición)",
        "prepare_png": "Preparar Rebanadas PNG para Descargar ",
        "download_png": "Descargar Rebanadas PNG para {} (.zip)",
        "labels": {
            "Background": "Fondo (Tejido Normal)",
            "Necrotic": "Tumor Necrótico/No Reforzado",
            "Edema": "Edema",
            "Enhancing": "Tumor Reforzado"
        }
    },
    "Français": {
        "title": "🧠 SegMed, un site web de segmentation 3D de tumeurs cérébrales",
        "description": """
        Téléchargez un modèle U-Net 3D pré-entraîné et des modalités NIfTI.
        L'application effectue la segmentation et propose deux options de téléchargement :
        1. Une **carte d'étiquettes NIfTI** standard (`.nii.gz`).
        2. Une **archive ZIP d'images PNG** pour chacune des 130 tranches traitées.
        **Assurez-vous que les poids de votre modèle correspondent à l'architecture U-Net définie ici.**
        """,
        "sidebar_header": "⚙️ Configuration",
        "patient_id": "Nom/ID du Patient",
        "unet_config": "Architecture U-Net",
        "input_channels": "Canaux d'Entrée (C)",
        "output_classes": "Classes de Sortie (Total, incl. fond)",
        "base_features": "Fonctions de Base",
        "upload_model": "Télécharger Modèle U-Net 3D (.pth)",
        "pretrained_model": "Modèle Pré-entraîné",
        "load_pretrained": "Charger Modèle Pré-entraîné depuis GitHub",
        "running_on": "Exécution sur",
        "input_files": "📁 Fichiers NIfTI d'Entrée",
        "modality_names": ["T1-natif (t1n)", "T1-contraste (t1c)", "T2-FLAIR (t2f)", "T2-pondéré (t2w)"],
        "run_button": "🚀 Exécuter Segmentation 3D",
        "results_header": "📊 Résultats de Segmentation",
        "multi_view": "Superposition de Segmentation Multi-vues",
        "legend_header": "Légende de Segmentation",
        "download_header": "💾 Options de Téléchargement",
        "nifti_option": "1. Carte d'Étiquettes NIfTI",
        "download_nifti": "Télécharger Segmentation d'Étiquettes (.nii.gz)",
        "png_option": "2. Tranches PNG (Superposition)",
        "prepare_png": "Préparer Tranches PNG pour Téléchargement ",
        "download_png": "Télécharger Tranches PNG pour {} (.zip)",
        "labels": {
            "Background": "Arrière-plan (Tissu Normal)",
            "Necrotic": "Tumeur Nécrotique/Non Rehaussée",
            "Edema": "Œdème",
            "Enhancing": "Tumeur Rehaussée"
        }
    }
}

# --- 3D U-Net Model Definition ---
class UNet3D(nn.Module):
    def __init__(self, in_channels=DEFAULT_IN_CHANNELS, out_channels=DEFAULT_OUT_CLASSES, base_features=DEFAULT_BASE_FEATURES):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_features = base_features
        # Initial convolution block
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(base_features), nn.ReLU(inplace=True)
        )
        # Encoder path
        self.encoder1 = self._make_block(base_features, base_features, blocks=1)
        self.encoder2 = self._make_block(base_features, base_features * 2, blocks=1, stride=2)
        self.encoder3 = self._make_block(base_features * 2, base_features * 4, blocks=1, stride=2)
        self.encoder4 = self._make_block(base_features * 4, base_features * 8, blocks=1, stride=2)
        # Decoder path
        self.upconv3 = nn.ConvTranspose3d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._make_block(base_features * 4 * 2, base_features * 4, blocks=1) 
        self.upconv2 = nn.ConvTranspose3d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._make_block(base_features * 2 * 2, base_features * 2, blocks=1)
        self.upconv1 = nn.ConvTranspose3d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.decoder1 = self._make_block(base_features * 2, base_features, blocks=1) 
        # Final convolution
        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels), nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels), nn.ReLU(inplace=True)
        )

    def _make_block(self, in_channels, out_channels, blocks, stride=1):
        layers = [self._conv_block(in_channels, out_channels, stride)]
        for _ in range(1, blocks): 
            layers.append(self._conv_block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.initial_conv(x); e1 = self.encoder1(x1)
        e2 = self.encoder2(e1); e3 = self.encoder3(e2); e4 = self.encoder4(e3)
        d3 = self.upconv3(e4)
        if d3.shape[2:] != e3.shape[2:]: d3 = F.interpolate(d3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1); d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        if d2.shape[2:] != e2.shape[2:]: d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1); d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        if d1.shape[2:] != e1.shape[2:]: d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1); d1 = self.decoder1(d1)
        logits = self.final_conv(d1)
        return F.interpolate(logits, size=x.shape[2:], mode='trilinear', align_corners=False)


# --- Data Preprocessing Function ---
def load_nii_and_preprocess(file_path, is_label=False, target_hw_shape=TARGET_HW_SHAPE, slice_range=(START_SLICE, END_SLICE)):
    try:
        img = nib.load(file_path)
        data = np.array(img.dataobj).astype(np.float32 if not is_label else np.int64)
        s_start, s_end = slice_range; target_d = s_end - s_start 

        if data.ndim == 4 and data.shape[3] == 1: data = np.squeeze(data, axis=3)
        elif data.ndim != 3: 
            st.warning(f"File {os.path.basename(file_path)} has {data.shape} dims. Expected 3D."); return None, None, None
        
        current_depth = data.shape[2]
        if current_depth > s_start:
            data_cropped_depth = data[:, :, s_start:min(s_end, current_depth)]
        else: 
            data_cropped_depth = np.zeros((data.shape[0], data.shape[1], 0), dtype=data.dtype)
        
        if data_cropped_depth.shape[2] == 0 and current_depth <= s_start and target_d > 0 :
            st.info(f"File {os.path.basename(file_path)} depth ({current_depth}) too shallow for start_slice ({s_start}). Initializing with zeros before padding.")

        processed_depth_val = data_cropped_depth.shape[2]
        if processed_depth_val < target_d:
            padding_needed = target_d - processed_depth_val
            pad_width = ((0,0), (0,0), (0, padding_needed))
            data_cropped_depth = np.pad(data_cropped_depth, pad_width, mode='constant', constant_values=0)
        elif processed_depth_val > target_d: 
            data_cropped_depth = data_cropped_depth[:, :, :target_d]
        
        resized_slices = []
        for i in range(data_cropped_depth.shape[2]): 
            slice_data = data_cropped_depth[:, :, i]
            order = 0 if is_label else 1; anti_aliasing = not is_label; preserve_range = True
            rs = resize(slice_data, target_hw_shape, order=order, mode='reflect' if not is_label else 'edge', 
                        anti_aliasing=anti_aliasing, preserve_range=preserve_range)
            resized_slices.append(rs.astype(np.float32 if not is_label else np.int64))
        
        if not resized_slices: st.error(f"No slices processed for {os.path.basename(file_path)}."); return None, None, None
        
        resized_volume = np.stack(resized_slices, axis=-1) 
        
        if not is_label:
            min_v, max_v = np.min(resized_volume), np.max(resized_volume)
            resized_volume = (resized_volume - min_v) / (max_v - min_v) if max_v - min_v > 1e-5 else np.zeros_like(resized_volume)
        
        expected_shape = (target_hw_shape[0], target_hw_shape[1], target_d)
        if resized_volume.shape != expected_shape:
            st.error(f"Preprocessing failed for {os.path.basename(file_path)}. Shape: {resized_volume.shape}. Expected: {expected_shape}"); return None,None,None
        return resized_volume, img.affine, img.header
    except Exception as e: 
        st.error(f"Error processing NIfTI file {os.path.basename(file_path)}: {e}"); st.exception(e); return None,None,None

# --- Function to convert label volume to RGBA volume ---
def labels_to_rgba(label_volume_dhw, num_total_classes, color_map_dict):
    rgba_volume = np.zeros((*label_volume_dhw.shape, 4), dtype=np.uint8) 
    for class_idx in range(num_total_classes): 
        color = color_map_dict.get(class_idx, (0,0,0,0)) 
        mask = (label_volume_dhw == class_idx)
        rgba_volume[mask] = color
    return rgba_volume
def create_slice_grid(input_volume, rgba_volume, patient_name):
    """
    Creates a high-resolution grid image of all slices in a 13x10 layout
    Args:
        input_volume: (H,W,D) numpy array of input slices
        rgba_volume: (D,H,W,4) numpy array of RGBA segmentations
        patient_name: Patient name for title
    Returns:
        PIL Image object of the grid
    """
    # Grid configuration
    SLICES_PER_ROW = 13
    ROWS = 10
    MARGIN = 5  # White space between slices
    TITLE_HEIGHT = 60  # Space for title
    
    # Get dimensions
    h, w = input_volume.shape[0], input_volume.shape[1]
    total_slices = input_volume.shape[2]
    
    # Calculate grid dimensions
    grid_width = (w * SLICES_PER_ROW) + (MARGIN * (SLICES_PER_ROW - 1))
    grid_height = (h * ROWS) + (MARGIN * (ROWS - 1)) + TITLE_HEIGHT
    
    # Create blank white image
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
    draw = ImageDraw.Draw(grid_img)
    
    # Add title
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    title = f"Patient: {patient_name} - All Slices (Total: {total_slices})"
    draw.text((10, 10), title, font=font, fill='black')
    
    # Composite each slice into the grid
    for i in range(min(total_slices, SLICES_PER_ROW * ROWS)):
        row = i // SLICES_PER_ROW
        col = i % SLICES_PER_ROW
        
        # Calculate position
        x = col * (w + MARGIN)
        y = TITLE_HEIGHT + row * (h + MARGIN)
        
        # Get slices
        input_slice = (input_volume[:, :, i] * 255).astype(np.uint8)
        seg_slice = rgba_volume[i, :, :, :]
        
        # Create composite image
        bg = Image.fromarray(input_slice).convert('RGB')
        overlay = Image.fromarray(seg_slice).convert('RGBA')
        composite = Image.alpha_composite(bg.convert('RGBA'), overlay).convert('RGB')
        
        # Paste into grid
        grid_img.paste(composite, (x, y))
    
    return grid_img

# --- Initialize Session State ---
if 'model_loaded' not in st.session_state: st.session_state.model_loaded = None
if 'device' not in st.session_state: st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'patient_name' not in st.session_state: st.session_state.patient_name = "UnknownPatient"
if 'current_date' not in st.session_state:
    st.session_state.current_date = datetime.now().strftime("%d %B %Y, %H:%M:%S") + " (Local Time)"
if 'language' not in st.session_state: st.session_state.language = "English"


import streamlit as st

# ===== BACKGROUND IMAGE SETUP =====
# Add this at the beginning of your main() function (before any other content)
def main():
    # Set background image
    def set_background_image():
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("https://raw.githubusercontent.com/Vwoudka/segmed/main/.devcontainer/iStock-1452990966-modified-26cda7e8-4ee1-4a98-b681-f8a249f82c52-768x432.jpg");
                    background-size: contain;
                    background-position: center center;
                    background-repeat: no-repeat;
                    background-attachment: scroll;
                }}
                /* Add semi-transparent overlay to make text more readable */
                .main .block-container {{
                    background-color: rgba(255, 255, 255, 0.9);
                    border-radius: 10px;
                    padding: 2rem;
                    margin-top: 2rem;
                    margin-bottom: 2rem;
                }}
                /* Adjust sidebar transparency */
                .sidebar .sidebar-content {{
                    background-color: rgba(255, 255, 255, 0.95);
                }}
            </style>
            """,
            unsafe_allow_html=True
        )

    set_background_image()
    
    # Rest of your existing code...
    
    # Your existing app code continues here...
    st.title("Your App Title")
    # ... rest of your app ...

if __name__ == "__main__":
    main()
    # Create a sidebar container for the language selector
    with st.sidebar:
        # Language selector at the top of the sidebar
        st.session_state.language = st.selectbox(
            "Language", 
            list(TRANSLATIONS.keys()), 
            index=list(TRANSLATIONS.keys()).index(st.session_state.language),
            key="language_selector"
        )
    
    # Get current translations
    t = TRANSLATIONS[st.session_state.language]
    
    # Main title and GitHub link
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title(t["title"])
    with col2:
        st.markdown("[<img src='https://github.githubassets.com/favicons/favicon.png' width='30'>](https://github.com/Vwoudka/segmed)", unsafe_allow_html=True)
    
    st.markdown(t["description"])
    
    # --- Rest of your existing sidebar content ---
    st.sidebar.header(t["sidebar_header"])
    st.sidebar.text_input(t["patient_id"], value=st.session_state.patient_name, key="patient_name_input_key", 
                          on_change=lambda: setattr(st.session_state, 'patient_name', st.session_state.patient_name_input_key))

    st.sidebar.subheader(t["unet_config"])
    # ... rest of your existing code ...

    param_in_channels = st.sidebar.number_input(t["input_channels"], min_value=1, value=DEFAULT_IN_CHANNELS, step=1, key="param_in_c")
    param_out_classes = st.sidebar.number_input(t["output_classes"], min_value=1, value=DEFAULT_OUT_CLASSES, step=1, key="param_out_cls")
    param_base_features = st.sidebar.number_input(t["base_features"], min_value=8, value=DEFAULT_BASE_FEATURES, step=16, key="param_base_f")

    uploaded_model_file = st.sidebar.file_uploader(t["upload_model"], type=["pth"])

    # Pretrained model button
    st.sidebar.header(t["pretrained_model"])
    if st.sidebar.button(t["load_pretrained"]):
        PRETRAINED_MODEL_URL = "https://github.com/Vwoudka/segmed/raw/main/.devcontainer/Use%20This%20One_UNet3D_patients125_epochs20_batch1_depth130.pth"
        
        with st.spinner("Downloading pretrained model..."):
            try:
                import requests
                response = requests.get(PRETRAINED_MODEL_URL)
                response.raise_for_status()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tf_model:
                    tf_model.write(response.content)
                    model_path_temp = tf_model.name
                
                st.info("Loading U-Net with pretrained weights...")
                model = UNet3D(in_channels=param_in_channels, out_channels=param_out_classes, 
                              base_features=param_base_features)
                model.load_state_dict(torch.load(model_path_temp, 
                                              map_location=st.session_state.device,
                                              weights_only=False))
                os.remove(model_path_temp)
                model.to(st.session_state.device).eval()
                st.session_state.model_loaded = model
                st.success("Pretrained model loaded successfully!")
                
            except Exception as e:
                st.error(f"Failed to load pretrained model: {e}")
                st.exception(e)
                

    # --- NIfTI File Uploaders ---
    st.header(t["input_files"])
    num_modality_uploaders = param_in_channels if param_in_channels > 0 else 1
    modality_display_names = t["modality_names"] if param_in_channels == 4 else [f"Modality {j+1}" for j in range(num_modality_uploaders)]

    cols_nifti = st.columns(num_modality_uploaders)
    uploaded_nifti_files = [None] * num_modality_uploaders
    for i in range(num_modality_uploaders):
        with cols_nifti[i]:
            uploaded_nifti_files[i] = st.file_uploader(f"Upload {modality_display_names[i]} (.nii.gz)", type=["nii.gz", "nii"], key=f"nifti_{i}")

    # --- Model Loading ---
    if uploaded_model_file is not None and st.session_state.model_loaded is None:
        try:
            st.info(f"Loading U-Net: In-Ch:{param_in_channels}, Out-Cls:{param_out_classes}, Base-Feat:{param_base_features}...")
            model = UNet3D(in_channels=param_in_channels, out_channels=param_out_classes, base_features=param_base_features)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tf_model:
                tf_model.write(uploaded_model_file.getvalue())
                model_path_temp = tf_model.name
            model.load_state_dict(torch.load(model_path_temp, map_location=st.session_state.device))
            os.remove(model_path_temp) 
            model.to(st.session_state.device).eval() 
            st.session_state.model_loaded = model
            st.success(f"Model loaded successfully and moved to {st.session_state.device}!")
            for key_to_clear in ['prediction_rgba_dhw4', 'prediction_label_dhw', 'input_for_vis_np', 'output_affine', 'output_header', 'zip_buffer_pngs']:
                if key_to_clear in st.session_state: del st.session_state[key_to_clear]
        except Exception as e: 
            st.error(f"Error loading model: {e}. Please ensure U-Net parameters match the model architecture."); st.exception(e)
            st.session_state.model_loaded = None

    # --- Segmentation Execution ---
    if st.button(t["run_button"], disabled=(st.session_state.model_loaded is None or not all(uploaded_nifti_files))):
        with st.spinner("Processing NIfTI files and running segmentation... This may take a moment."):
            try:
                processed_modalities, original_affines, original_headers = [], [], []
                base_hw_shape = None 
                
                for i, uploaded_file_obj in enumerate(uploaded_nifti_files):
                    st.info(f"Processing {modality_display_names[i]}...")
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tf_nifti:
                        tf_nifti.write(uploaded_file_obj.getvalue())
                        nifti_path_temp = tf_nifti.name
                    
                    volume, affine, header = load_nii_and_preprocess(nifti_path_temp, is_label=False)
                    os.remove(nifti_path_temp) 
                    
                    if volume is None: st.error(f"Failed to process {modality_display_names[i]}. Segmentation aborted."); st.stop()
                    
                    if base_hw_shape is None: base_hw_shape = volume.shape[:2] 
                    elif volume.shape[:2] != base_hw_shape: 
                        st.error(f"{modality_display_names[i]} HxW shape {volume.shape[:2]} differs from first modality {base_hw_shape}. All must match."); st.stop()
                    if volume.shape[2] != TARGET_DEPTH: 
                        st.error(f"{modality_display_names[i]} processed depth {volume.shape[2]} != target depth {TARGET_DEPTH}. Aborting."); st.stop()
                    
                    processed_modalities.append(volume) 
                    if i == 0: original_affines.append(affine); original_headers.append(header)
                
                stacked_modalities_np = np.stack(processed_modalities, axis=0)
                input_volume_np_cdhw = stacked_modalities_np.transpose(0,3,1,2) 
                
                input_tensor = torch.from_numpy(np.expand_dims(input_volume_np_cdhw, axis=0)).float().to(st.session_state.device)
                st.info(f"Input tensor shape for model: {input_tensor.shape} (N, C, D, H, W)")

                st.info("Running model inference...")
                with torch.no_grad(): output_logits = st.session_state.model_loaded(input_tensor) 
                
                pred_labels_tensor = torch.argmax(output_logits.squeeze(0), dim=0) 
                pred_labels_np_dhw = pred_labels_tensor.cpu().numpy().astype(np.uint8) 
                st.session_state.prediction_label_dhw = pred_labels_np_dhw 
                st.success("Segmentation labels generated!")

                st.info("Converting labels to RGBA for visualization...")
                pred_rgba_dhw4 = labels_to_rgba(pred_labels_np_dhw, param_out_classes, LABEL_TO_RGBA)
                st.session_state.prediction_rgba_dhw4 = pred_rgba_dhw4 
                
                st.session_state.input_for_vis_np = processed_modalities[0] 
                st.session_state.output_affine = original_affines[0]
                st.session_state.output_header = original_headers[0]
                st.success("Processing complete!")

            except Exception as e:
                st.error(f"An error occurred during segmentation: {e}"); st.exception(e)
                for key_to_clear in ['prediction_rgba_dhw4', 'prediction_label_dhw']: 
                    if key_to_clear in st.session_state: del st.session_state[key_to_clear]

    # --- Display Results and Download Options ---
    if 'prediction_label_dhw' in st.session_state and st.session_state.prediction_label_dhw is not None:
        st.header(t["results_header"])
        
        input_vis_dhw = np.transpose(st.session_state.input_for_vis_np, (2,0,1))
        labels_dhw = st.session_state.prediction_label_dhw
        rgba_dhw4 = st.session_state.prediction_rgba_dhw4

        mid_d = labels_dhw.shape[0] // 2
        mid_h = labels_dhw.shape[1] // 2
        mid_w = labels_dhw.shape[2] // 2

        plot_views_corrected = {
            "Axial": {
                "input": input_vis_dhw[mid_d, :, :],
                "rgba": rgba_dhw4[mid_d, :, :, :],
                "title": f"Axial Slice: {mid_d + START_SLICE} (orig) / {mid_d} (proc)",
                "aspect": "equal"
            },
            "Sagittal": { 
                "input": input_vis_dhw[:, mid_h, :],
                "rgba": rgba_dhw4[:, mid_h, :, :],
                "title": f"Sagittal View (H-slice: {mid_h})",
                "aspect": input_vis_dhw.shape[2] / input_vis_dhw.shape[0]
            },
            "Coronal": { 
                "input": input_vis_dhw[:, :, mid_w],
                "rgba": rgba_dhw4[:, :, mid_w, :],
                "title": f"Coronal View (W-slice: {mid_w})",
                "aspect": input_vis_dhw.shape[1] / input_vis_dhw.shape[0]
            }
        }

        st.subheader(t["multi_view"])
        cols_vis = st.columns(len(plot_views_corrected))

        for i, (view_name, data) in enumerate(plot_views_corrected.items()):
            with cols_vis[i]:
                st.markdown(f"**{data['title']}**")
                fig, ax = plt.subplots(figsize=(6,6)) 
                ax.imshow(data["input"], cmap='gray', aspect=data["aspect"])
                ax.imshow(data["rgba"], aspect=data["aspect"]) 
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig) 

        # Replace the legend section in your code with this:

    st.subheader(t["legend_header"])


    legend_html = """
    <div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom:20px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>
        <div style='display:flex;align-items:center;'>
            <div style='width:20px;height:20px;background-color:rgba(128,128,128,0.1);margin-right:8px;border: 1px dashed #aaa;'></div>
            <span style='font-size:0.9em;color: black;'>{background_label}</span>
        </div>
        <div style='display:flex;align-items:center;'>
            <div style='width:20px;height:20px;background-color:rgba(255,0,0,1.00);margin-right:8px;border: 1px solid #555;'></div>
            <span style='font-size:0.9em;color: black;'>{necrotic_label}</span>
        </div>
        <div style='display:flex;align-items:center;'>
            <div style='width:20px;height:20px;background-color:rgba(0,255,0,1.00);margin-right:8px;border: 1px solid #555;'></div>
            <span style='font-size:0.9em;color: black;'>{edema_label}</span>
        </div>
        <div style='display:flex;align-items:center;'>
            <div style='width:20px;height:20px;background-color:rgba(255,255,0,1.00);margin-right:8px;border: 1px solid #555;'></div>
            <span style='font-size:0.9em;color: black;'>{enhancing_label}</span>
        </div>
    </div>
    """.format(
        background_label=t["labels"]["Background"],
        necrotic_label=t["labels"]["Necrotic"],
        edema_label=t["labels"]["Edema"],
        enhancing_label=t["labels"]["Enhancing"]
    )
    
    st.markdown(legend_html, unsafe_allow_html=True)
        st.header(t["download_header"])
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.subheader(t["nifti_option"])
            data_to_save_nifti = np.transpose(st.session_state.prediction_label_dhw, (1,2,0)).astype(np.uint8)
            nifti_img_out = nib.Nifti1Image(data_to_save_nifti, st.session_state.output_affine, st.session_state.output_header)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii.gz") as tmp_nifti_download_file:
                nib.save(nifti_img_out, tmp_nifti_download_file.name)
                tmp_nifti_download_path = tmp_nifti_download_file.name

            with open(tmp_nifti_download_path, "rb") as fp:
                st.download_button(
                    label=t["download_nifti"],
                    data=fp.read(), 
                    file_name=f"{st.session_state.patient_name}_segmentation_labels.nii.gz",
                    mime="application/gzip"
                )
            if os.path.exists(tmp_nifti_download_path): 
                os.remove(tmp_nifti_download_path)
        
        # In your download options section where you have the PNG download button
        # In your download options section (replace the existing code)
with col_dl2:
    st.subheader(t["png_option"])
    
    # Add format selection
    png_format = st.radio(
        "Select output format:",
        ["Individual Slices (ZIP)", "13×10 Grid Image"],
        key="png_format_selector"
    )
    
    if st.button(t["prepare_png"]):
        if png_format == "Individual Slices (ZIP)":
            with st.spinner(f"Generating {TARGET_DEPTH} PNG slices for {st.session_state.patient_name}..."):
                try:
                    zip_buffer = io.BytesIO()
                    input_slices_hwd = st.session_state.input_for_vis_np 
                    rgba_slices_dhw4 = st.session_state.prediction_rgba_dhw4
                    label_slices_dhw = st.session_state.prediction_label_dhw

                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for slice_idx_png in range(TARGET_DEPTH):
                            current_input_slice_hw = input_slices_hwd[:, :, slice_idx_png]
                            current_rgba_slice_hw4 = rgba_slices_dhw4[slice_idx_png, :, :, :]
                            current_label_slice_hw = label_slices_dhw[slice_idx_png, :, :]

                            present_label_names = []
                            for label_val, label_name_str in SEGMENTATION_LABELS_DICT.items():
                                if np.any(current_label_slice_hw == label_val):
                                    present_label_names.append(label_name_str)
                            
                            labels_present_str = ", ".join(present_label_names) if present_label_names else "None"
                            
                            fig_png, ax_png = plt.subplots(figsize=(6,6))
                            ax_png.imshow(current_input_slice_hw, cmap='gray', aspect='equal')
                            ax_png.imshow(current_rgba_slice_hw4, aspect='equal')
                            ax_png.axis('off')
                            
                            title_str = (f"Patient: {st.session_state.patient_name}\n"
                                        f"Slice: {slice_idx_png + START_SLICE} (Original Index)\n"
                                        f"Labels Present: {labels_present_str}")
                            ax_png.set_title(title_str, fontsize=10)
                            
                            png_buf = io.BytesIO()
                            fig_png.savefig(png_buf, format='png', dpi=100, bbox_inches='tight')
                            plt.close(fig_png)
                            png_buf.seek(0)
                            
                            safe_labels_str = "_".join(labels_present_str.split(", "))
                            png_filename = f"{st.session_state.patient_name}_slice_{slice_idx_png + START_SLICE:03d}_{safe_labels_str or 'no_labels'}.png"
                            zf.writestr(png_filename, png_buf.getvalue())
                    
                    zip_buffer.seek(0)
                    st.session_state.zip_buffer_pngs = zip_buffer
                    st.session_state.png_format = "zip"
                    st.success(f"PNG ZIP archive ready for patient {st.session_state.patient_name}!")
                    
                except Exception as e:
                    st.error(f"Error generating PNG slices: {str(e)}")
                    st.exception(e)

        elif png_format == "13×10 Grid Image":
            with st.spinner("Creating 13×10 grid image..."):
                try:
                    # Grid configuration
                    SLICES_PER_ROW = 13
                    ROWS = 10
                    MARGIN = 5
                    TITLE_HEIGHT = 60
                    
                    # Get data
                    input_volume = st.session_state.input_for_vis_np  # (H,W,D)
                    rgba_volume = st.session_state.prediction_rgba_dhw4  # (D,H,W,4)
                    h, w = input_volume.shape[0], input_volume.shape[1]
                    
                    # Calculate grid dimensions
                    grid_width = (w * SLICES_PER_ROW) + (MARGIN * (SLICES_PER_ROW - 1))
                    grid_height = (h * ROWS) + (MARGIN * (ROWS - 1)) + TITLE_HEIGHT
                    
                    # Create blank image
                    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
                    draw = ImageDraw.Draw(grid_img)
                    
                    # Add title
                    try:
                        font = ImageFont.truetype("arial.ttf", 24)
                    except:
                        font = ImageFont.load_default()
                    
                    title = f"Patient: {st.session_state.patient_name} - Slice Grid (Total: {TARGET_DEPTH} slices)"
                    draw.text((10, 10), title, font=font, fill='black')
                    
                    # Add slices to grid
                    for i in range(min(TARGET_DEPTH, SLICES_PER_ROW * ROWS)):
                        row = i // SLICES_PER_ROW
                        col = i % SLICES_PER_ROW
                        
                        x = col * (w + MARGIN)
                        y = TITLE_HEIGHT + row * (h + MARGIN)
                        
                        # Get slices
                        input_slice = (input_volume[:, :, i] * 255).astype(np.uint8)
                        seg_slice = rgba_volume[i, :, :, :]
                        
                        # Create composite
                        bg = Image.fromarray(input_slice).convert('RGB')
                        overlay = Image.fromarray(seg_slice).convert('RGBA')
                        composite = Image.alpha_composite(bg.convert('RGBA'), overlay).convert('RGB')
                        
                        # Paste into grid
                        grid_img.paste(composite, (x, y))
                    
                    # Save to buffer
                    img_buffer = io.BytesIO()
                    grid_img.save(img_buffer, format='PNG', quality=100)
                    img_buffer.seek(0)
                    
                    st.session_state.grid_image_buffer = img_buffer
                    st.session_state.png_format = "grid"
                    st.success("13×10 grid image created successfully!")
                    
                    # Show preview
                    st.image(grid_img, caption="Preview of 13×10 Grid", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating grid image: {str(e)}")
                    st.exception(e)

    # Download buttons
    if 'png_format' in st.session_state:
        if st.session_state.png_format == "zip" and 'zip_buffer_pngs' in st.session_state:
            st.download_button(
                label=t["download_png"].format(st.session_state.patient_name),
                data=st.session_state.zip_buffer_pngs,
                file_name=f"{st.session_state.patient_name}_segmentation_slices.zip",
                mime="application/zip"
            )
        elif st.session_state.png_format == "grid" and 'grid_image_buffer' in st.session_state:
            st.download_button(
                label="Download 13×10 Grid Image",
                data=st.session_state.grid_image_buffer,
                file_name=f"{st.session_state.patient_name}_slice_grid.png",
                mime="image/png"
            )
          
                

    st.markdown("---")
    st.markdown(f"Timestamp: {st.session_state.current_date}")

if __name__ == "__main__":
    main()
