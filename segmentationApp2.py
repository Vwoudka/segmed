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
from PIL import Image, ImageDraw, ImageFont # <-- ADDED IMPORT FOR PILLOW
import requests # For downloading pretrained model

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

# --- Translation Dictionary (MODIFIED "prepare_png") ---
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
        "prepare_png": "Prepare PNG Slices for Download", # <-- MODIFIED
        "download_png": "Download PNG Slices for {} (.zip)",
        "select_png_format": "Select PNG download format:",
        "individual_slices_zip": "Individual Slices (ZIP)",
        "high_res_grid_png": "High-Res Grid (13x10 PNG)",
        "download_grid_button": "Download High-Res Grid (PNG)",
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
        "prepare_png": "Preparar Rebanadas PNG para Descargar", # <-- MODIFIED
        "download_png": "Descargar Rebanadas PNG para {} (.zip)",
        "select_png_format": "Seleccione el formato de descarga:",
        "individual_slices_zip": "Rebanadas Individuales (ZIP)",
        "high_res_grid_png": "Cuadrícula Alta Resolución (13x10 PNG)",
        "download_grid_button": "Descargar Cuadrícula Alta Resolución (PNG)",
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
        "prepare_png": "Préparer Tranches PNG pour Téléchargement", # <-- MODIFIED
        "download_png": "Télécharger Tranches PNG pour {} (.zip)",
        "select_png_format": "Sélectionnez le format de téléchargement :",
        "individual_slices_zip": "Tranches Individuelles (ZIP)",
        "high_res_grid_png": "Grille Haute Résolution (13x10 PNG)",
        "download_grid_button": "Télécharger Grille Haute Résolution (PNG)",
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

# --- Function to create slice grid ---
def create_slice_grid(input_volume_hwd, rgba_volume_dhw4, patient_name):
    """
    Creates a high-resolution grid image of all slices in a 13x10 layout
    Args:
        input_volume_hwd: (H,W,D) numpy array of input slices (e.g., first modality)
        rgba_volume_dhw4: (D,H,W,4) numpy array of RGBA segmentations
        patient_name: Patient name for title
    Returns:
        PIL Image object of the grid, or None on error
    """
    try:
        # Grid configuration
        SLICES_PER_ROW = 13
        ROWS = 10
        MARGIN = 10  # White space between slices
        TITLE_HEIGHT = 60  # Space for title
        
        # Get dimensions
        h, w, total_slices = input_volume_hwd.shape
        
        # Ensure rgba_volume_dhw4 is in D,H,W,4
        if rgba_volume_dhw4.shape[0] != total_slices or \
           rgba_volume_dhw4.shape[1] != h or \
           rgba_volume_dhw4.shape[2] != w:
            st.error(f"Dimension mismatch for grid: input_hwd {input_volume_hwd.shape}, rgba_dhw4 {rgba_volume_dhw4.shape}")
            return None

        # Calculate grid dimensions
        grid_width = (w * SLICES_PER_ROW) + (MARGIN * (SLICES_PER_ROW + 1)) # Added margin for sides
        grid_height = TITLE_HEIGHT + (h * ROWS) + (MARGIN * (ROWS + 1)) # Added margin for top/bottom
        
        # Create blank white image
        grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
        draw = ImageDraw.Draw(grid_img)
        
        # Add title
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError: # More specific exception for font loading
            font = ImageFont.load_default()
        
        title_text = f"Patient: {patient_name} - Segmentation Slices (Grid View)"
        # Corrected way to get text size with Pillow >= 9.2.0
        try:
            text_bbox = draw.textbbox((0,0), title_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            # text_height = text_bbox[3] - text_bbox[1] # Not strictly needed for centering width
        except AttributeError: # Fallback for older Pillow versions
            text_width, _ = draw.textsize(title_text, font=font)

        draw.text(((grid_width - text_width) / 2, MARGIN // 2 + 5), title_text, font=font, fill='black') # Centered title
        
        # Composite each slice into the grid
        for i in range(min(total_slices, SLICES_PER_ROW * ROWS)):
            row_idx = i // SLICES_PER_ROW
            col_idx = i % SLICES_PER_ROW
            
            # Calculate position with margin
            x = MARGIN + col_idx * (w + MARGIN)
            y = TITLE_HEIGHT + MARGIN + row_idx * (h + MARGIN)
            
            # Get slices
            # Input slice is from (H,W,D)
            input_slice_hw = (input_volume_hwd[:, :, i] * 255).astype(np.uint8)
            # Seg slice is from (D,H,W,4)
            seg_slice_hw4 = rgba_volume_dhw4[i, :, :, :]
            
            # Create composite image
            bg = Image.fromarray(input_slice_hw).convert('RGB')
            overlay = Image.fromarray(seg_slice_hw4).convert('RGBA')
            
            # Ensure bg is RGBA for alpha_composite
            composite = Image.alpha_composite(bg.convert('RGBA'), overlay).convert('RGB')
            
            # Paste into grid
            grid_img.paste(composite, (int(x), int(y)))
        
        return grid_img
    except Exception as e:
        st.error(f"Error creating slice grid: {e}")
        st.exception(e)
        return None

# --- Main Application Logic ---
def main():
    # Set page config (optional, but good for title and layout)
    st.set_page_config(page_title="SegMed 3D", layout="wide")

    # Initialize Session State (if not already done by Streamlit's execution)
    if 'model_loaded' not in st.session_state: st.session_state.model_loaded = None
    if 'device' not in st.session_state: st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if 'patient_name' not in st.session_state: st.session_state.patient_name = "UnknownPatient"
    if 'current_date' not in st.session_state:
        st.session_state.current_date = datetime.now().strftime("%d %B %Y, %H:%M:%S") + " (Local Time)"
    if 'language' not in st.session_state: st.session_state.language = "English"
    # For PNG download format choice
    if 'png_download_format' not in st.session_state: st.session_state.png_download_format = "Individual Slices (ZIP)"


    # ===== BACKGROUND IMAGE SETUP =====
    def set_background_image():
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("https://raw.githubusercontent.com/Vwoudka/segmed/main/.devcontainer/iStock-1452990966-modified-26cda7e8-4ee1-4a98-b681-f8a249f82c52-768x432.jpg");
                    background-size: contain; /* Or cover, depending on desired effect */
                    background-position: center center;
                    background-repeat: no-repeat;
                    background-attachment: scroll; /* Or fixed */
                }}
                /* Add semi-transparent overlay to make text more readable */
                .main .block-container {{
                    background-color: rgba(255, 255, 255, 0.9); /* Slightly more opaque */
                    border-radius: 10px;
                    padding: 2rem;
                    margin-top: 2rem;
                    margin-bottom: 2rem;
                }}
                /* Adjust sidebar transparency */
                .css-1lcbmhc {{ /* This is a common class for sidebar, might need adjustment if Streamlit updates */
                     background-color: rgba(255, 255, 255, 0.95) !important;
                }}
                 /* Targeting sidebar content more specifically */
                section[data-testid="stSidebar"] > div:first-child {{
                    background-color: rgba(255, 255, 255, 0.95);
                }}
            </style>
            """,
            unsafe_allow_html=True
        )
    set_background_image()

    # --- Sidebar ---
    with st.sidebar:
        # Language selector at the top of the sidebar
        selected_language = st.selectbox(
            "Language / Idioma / Langue", 
            list(TRANSLATIONS.keys()), 
            index=list(TRANSLATIONS.keys()).index(st.session_state.language),
            key="language_selector_key" # Use a unique key
        )
        # Update session state language if it changed
        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.rerun() # Rerun to apply language change immediately

    # Get current translations AFTER language selection
    t = TRANSLATIONS[st.session_state.language]

    # Main title and GitHub link
    col1_title, col2_gh = st.columns([6, 1])
    with col1_title:
        st.title(t["title"])
    with col2_gh:
        st.markdown("[<img src='https://github.githubassets.com/favicons/favicon.png' width='30'>](https://github.com/Vwoudka/segmed)", unsafe_allow_html=True)
    
    st.markdown(t["description"])
    
    # --- Rest of Sidebar Content ---
    with st.sidebar:
        st.header(t["sidebar_header"])
        
        # Patient Name/ID
        patient_name_input = st.text_input(
            t["patient_id"], 
            value=st.session_state.patient_name, 
            key="patient_name_input_key_main" # Unique key
        )
        if patient_name_input != st.session_state.patient_name:
            st.session_state.patient_name = patient_name_input

        st.subheader(t["unet_config"])
        param_in_channels = st.number_input(t["input_channels"], min_value=1, value=DEFAULT_IN_CHANNELS, step=1, key="param_in_c")
        param_out_classes = st.number_input(t["output_classes"], min_value=1, value=DEFAULT_OUT_CLASSES, step=1, key="param_out_cls")
        param_base_features = st.number_input(t["base_features"], min_value=8, value=DEFAULT_BASE_FEATURES, step=16, key="param_base_f")

        uploaded_model_file = st.file_uploader(t["upload_model"], type=["pth"])

        st.header(t["pretrained_model"])
        if st.button(t["load_pretrained"]):
            PRETRAINED_MODEL_URL = "https://github.com/Vwoudka/segmed/raw/main/.devcontainer/Use%20This%20One_UNet3D_patients125_epochs20_batch1_depth130.pth"
            
            with st.spinner("Downloading pretrained model..."):
                try:
                    response = requests.get(PRETRAINED_MODEL_URL, timeout=30) # Added timeout
                    response.raise_for_status()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tf_model:
                        tf_model.write(response.content)
                        model_path_temp = tf_model.name
                    
                    st.info("Loading U-Net with pretrained weights...")
                    model = UNet3D(in_channels=param_in_channels, out_channels=param_out_classes, 
                                   base_features=param_base_features)
                    # Forcing weights_only=True if a .pth file is just a state_dict
                    # However, if the .pth file is a pickled model, this would be False.
                    # The error "AttributeError: 'collections.OrderedDict' object has no attribute 'eval'"
                    # suggests it's already a state_dict.
                    model.load_state_dict(torch.load(model_path_temp, 
                                                     map_location=st.session_state.device)) # weights_only=False removed for now based on typical .pth usage
                    
                    os.remove(model_path_temp)
                    model.to(st.session_state.device).eval()
                    st.session_state.model_loaded = model
                    st.success("Pretrained model loaded successfully!")
                    # Clear previous results
                    for key_to_clear in ['prediction_rgba_dhw4', 'prediction_label_dhw', 'input_for_vis_np', 'output_affine', 'output_header', 'zip_buffer_pngs', 'grid_image_buffer']:
                        if key_to_clear in st.session_state: del st.session_state[key_to_clear]

                except requests.exceptions.RequestException as e_req:
                    st.error(f"Network error downloading pretrained model: {e_req}")
                except Exception as e:
                    st.error(f"Failed to load pretrained model: {e}")
                    st.exception(e)
        
        st.info(f"{t['running_on']}: {st.session_state.device}")


    # --- NIfTI File Uploaders ---
    st.header(t["input_files"])
    num_modality_uploaders = param_in_channels if param_in_channels > 0 else 1
    modality_display_names_list = t["modality_names"] # Ensure this is a list
    # Ensure modality_display_names matches num_modality_uploaders
    if len(modality_display_names_list) < num_modality_uploaders:
        modality_display_names = [f"Modality {j+1}" for j in range(num_modality_uploaders)]
    else:
        modality_display_names = modality_display_names_list[:num_modality_uploaders]


    cols_nifti = st.columns(num_modality_uploaders)
    uploaded_nifti_files = [None] * num_modality_uploaders
    for i in range(num_modality_uploaders):
        with cols_nifti[i]:
            uploaded_nifti_files[i] = st.file_uploader(f"Upload {modality_display_names[i]} (.nii.gz)", type=["nii.gz", "nii"], key=f"nifti_{i}")

    # --- Model Loading (from user upload) ---
    if uploaded_model_file is not None and st.session_state.model_loaded is None: # Only load if no model is loaded yet
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
            for key_to_clear in ['prediction_rgba_dhw4', 'prediction_label_dhw', 'input_for_vis_np', 'output_affine', 'output_header', 'zip_buffer_pngs', 'grid_image_buffer']:
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
                
                # Transpose from (num_modalities, H, W, D) to (num_modalities, D, H, W) for model
                stacked_modalities_np = np.stack(processed_modalities, axis=0) # C, H, W, D
                input_volume_np_cdhw = stacked_modalities_np.transpose(0,3,1,2) # C, D, H, W
                
                input_tensor = torch.from_numpy(np.expand_dims(input_volume_np_cdhw, axis=0)).float().to(st.session_state.device) # N, C, D, H, W
                st.info(f"Input tensor shape for model: {input_tensor.shape} (N, C, D, H, W)")

                st.info("Running model inference...")
                with torch.no_grad(): output_logits = st.session_state.model_loaded(input_tensor) 
                
                pred_labels_tensor = torch.argmax(output_logits.squeeze(0), dim=0) # D, H, W
                pred_labels_np_dhw = pred_labels_tensor.cpu().numpy().astype(np.uint8) 
                st.session_state.prediction_label_dhw = pred_labels_np_dhw 
                st.success("Segmentation labels generated!")

                st.info("Converting labels to RGBA for visualization...")
                pred_rgba_dhw4 = labels_to_rgba(pred_labels_np_dhw, param_out_classes, LABEL_TO_RGBA)
                st.session_state.prediction_rgba_dhw4 = pred_rgba_dhw4 
                
                # Use the first modality (H,W,D) for visualization base
                st.session_state.input_for_vis_np_hwd = processed_modalities[0] 
                st.session_state.output_affine = original_affines[0]
                st.session_state.output_header = original_headers[0]
                st.success("Processing complete!")
                # Clear any old download buffers
                if 'zip_buffer_pngs' in st.session_state: del st.session_state['zip_buffer_pngs']
                if 'grid_image_buffer' in st.session_state: del st.session_state['grid_image_buffer']


            except Exception as e:
                st.error(f"An error occurred during segmentation: {e}"); st.exception(e)
                for key_to_clear in ['prediction_rgba_dhw4', 'prediction_label_dhw', 'input_for_vis_np_hwd']: 
                    if key_to_clear in st.session_state: del st.session_state[key_to_clear]

    # --- Display Results and Download Options ---
    if 'prediction_label_dhw' in st.session_state and st.session_state.prediction_label_dhw is not None:
        st.header(t["results_header"])
        
        # input_for_vis_np_hwd is (H,W,D)
        # prediction_label_dhw is (D,H,W)
        # prediction_rgba_dhw4 is (D,H,W,4)
        
        input_vis_hwd = st.session_state.input_for_vis_np_hwd # H, W, D
        labels_dhw = st.session_state.prediction_label_dhw     # D, H, W
        rgba_dhw4 = st.session_state.prediction_rgba_dhw4     # D, H, W, 4

        # Determine mid slices for visualization
        # input_vis_hwd: H, W, D
        # labels_dhw: D, H, W
        # rgba_dhw4:  D, H, W, 4 (already in D,H,W for first dim)

        num_slices_d = labels_dhw.shape[0]
        height_h = labels_dhw.shape[1]
        width_w = labels_dhw.shape[2]

        mid_d_processed = num_slices_d // 2 # Index for processed depth
        mid_h_processed = height_h // 2
        mid_w_processed = width_w // 2

        # Axial View: Input (H,W,D) -> pick D slice. Output (D,H,W) -> pick D slice
        axial_input_slice = input_vis_hwd[:, :, mid_d_processed] # H, W
        axial_rgba_slice = rgba_dhw4[mid_d_processed, :, :, :] # H, W, 4

        # Sagittal View: Input (H,W,D) -> pick W slice, then transpose. Output (D,H,W) -> pick W slice
        sagittal_input_slice = np.transpose(input_vis_hwd[:, mid_w_processed, :], (1,0)) # D, H (after picking W, then transpose H,D -> D,H)
        sagittal_rgba_slice = rgba_dhw4[:, :, mid_w_processed, :] # D, H, 4

        # Coronal View: Input (H,W,D) -> pick H slice, then transpose. Output (D,H,W) -> pick H slice
        coronal_input_slice = np.transpose(input_vis_hwd[mid_h_processed, :, :], (1,0)) # D, W (after picking H, then transpose W,D -> D,W)
        coronal_rgba_slice = rgba_dhw4[:, mid_h_processed, :, :] # D, W, 4

        plot_views_corrected = {
            "Axial": {
                "input": axial_input_slice, # H,W
                "rgba": axial_rgba_slice,   # H,W,4
                "title": f"Axial Slice: {mid_d_processed + START_SLICE} (orig) / {mid_d_processed} (proc)",
                "aspect": "equal" # H vs W aspect
            },
            "Sagittal": { 
                "input": sagittal_input_slice, # D,H
                "rgba": sagittal_rgba_slice,   # D,H,4
                "title": f"Sagittal View (W-slice: {mid_w_processed})",
                "aspect": height_h / num_slices_d # Aspect D vs H
            },
            "Coronal": { 
                "input": coronal_input_slice, # D,W
                "rgba": coronal_rgba_slice,   # D,W,4
                "title": f"Coronal View (H-slice: {mid_h_processed})",
                "aspect": width_w / num_slices_d # Aspect D vs W
            }
        }

        st.subheader(t["multi_view"])
        cols_vis = st.columns(len(plot_views_corrected))

        for i, (view_name, data) in enumerate(plot_views_corrected.items()):
            with cols_vis[i]:
                st.markdown(f"**{data['title']}**")
                fig, ax = plt.subplots(figsize=(6,6)) 
                ax.imshow(np.rot90(data["input"]), cmap='gray', aspect=data["aspect"]) # Rotated for standard medical view
                ax.imshow(np.rot90(data["rgba"], k=1, axes=(0,1)), aspect=data["aspect"]) # Rotated for standard medical view
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig) 

        st.subheader(t["legend_header"])
        legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom:20px; padding: 10px; background-color: #f0f2f6; border-radius: 5px;'>"
        # Make sure to iterate through defined labels in LABEL_TO_RGBA to match colors correctly
        display_labels_ordered = {0: "Background", 1: "Necrotic", 2: "Edema", 3: "Enhancing"}

        for label_idx, name_key  in display_labels_ordered.items():
            if name_key not in t["labels"]: continue # Skip if label not in translation (e.g. for fewer classes)
            
            label_name_display = t["labels"][name_key]
            r,g,b,a = LABEL_TO_RGBA.get(label_idx, (0,0,0,0)) # Get color by index
            
            c_disp = f"rgba({r},{g},{b},{a/255:.2f})"
            b_style = "border: 1px solid #555;"
            text_color = "color: black;"
            if a==0 and name_key=="Background":
                c_disp="rgba(128,128,128,0.1)" # So it's visible on white
                b_style="border: 1px dashed #aaa;"
            legend_html += f"""
            <div style='display:flex;align-items:center;'>
                <div style='width:20px;height:20px;background-color:{c_disp};margin-right:8px;{b_style}'></div>
                <span style='font-size:0.9em;{text_color}'>{label_name_display}</span>
            </div>
            """
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

        st.header(t["download_header"])
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            st.subheader(t["nifti_option"])
            # prediction_label_dhw is (D,H,W), nibabel expects (X,Y,Z) so (W,H,D) or (H,W,D)
            # Standard NIfTI is often RAS: (Right-Left, Anterior-Posterior, Superior-Inferior)
            # Our processing is (D, H, W) -> (Depth, Height, Width)
            # We need to transpose it to (H, W, D) or (W, H, D) to match original orientation as much as possible
            # Let's assume original was (H,W,D)-like for visual consistency in viewers.
            data_to_save_nifti = np.transpose(st.session_state.prediction_label_dhw, (1,2,0)).astype(np.uint8) # H, W, D
            nifti_img_out = nib.Nifti1Image(data_to_save_nifti, st.session_state.output_affine, st.session_state.output_header)
            
            nifti_filename = f"{st.session_state.patient_name}_segmentation_labels.nii.gz"
            with io.BytesIO() as nifti_buffer:
                nib.save(nifti_img_out, filename=nifti_buffer) # Nibabel can write to a BytesIO if filename is passed
                nifti_buffer.seek(0)
                st.download_button(
                    label=t["download_nifti"],
                    data=nifti_buffer.getvalue(), 
                    file_name=nifti_filename,
                    mime="application/gzip"
                )
        
        with col_dl2:
            st.subheader(t["png_option"])
            
            download_format_choice = st.radio(
                t["select_png_format"],
                [t["individual_slices_zip"], t["high_res_grid_png"]],
                key="png_format_radio",
                on_change=lambda: setattr(st.session_state, 'png_download_format', st.session_state.png_format_radio)
            )

            if st.button(t["prepare_png"], key="prepare_png_slices_button"):
                if st.session_state.png_download_format == t["individual_slices_zip"]:
                    with st.spinner(f"Generating {TARGET_DEPTH} PNG slices for {st.session_state.patient_name}..."):
                        zip_buffer_individual = io.BytesIO()
                        # input_for_vis_np_hwd is (H,W,D)
                        # prediction_rgba_dhw4 is (D,H,W,4)
                        # prediction_label_dhw is (D,H,W)
                        input_slices_hwd = st.session_state.input_for_vis_np_hwd 
                        rgba_slices_dhw4 = st.session_state.prediction_rgba_dhw4
                        label_slices_dhw = st.session_state.prediction_label_dhw

                        with zipfile.ZipFile(zip_buffer_individual, "w", zipfile.ZIP_DEFLATED, False) as zf:
                            for slice_idx_png in range(TARGET_DEPTH): # Iterate through depth
                                current_input_slice_hw = input_slices_hwd[:, :, slice_idx_png] # H,W
                                current_rgba_slice_hw4 = rgba_slices_dhw4[slice_idx_png, :, :, :] # H,W,4
                                current_label_slice_hw = label_slices_dhw[slice_idx_png, :, :] # H,W

                                present_label_names = []
                                for label_val, label_name_str in SEGMENTATION_LABELS_DICT.items():
                                    if np.any(current_label_slice_hw == label_val):
                                        present_label_names.append(label_name_str)
                                
                                labels_present_str = ", ".join(present_label_names) if present_label_names else "None"
                                
                                fig_png, ax_png = plt.subplots(figsize=(TARGET_HW_SHAPE[1]/100, TARGET_HW_SHAPE[0]/100), dpi=100) # Match aspect
                                ax_png.imshow(current_input_slice_hw, cmap='gray', aspect='equal')
                                ax_png.imshow(current_rgba_slice_hw4, aspect='equal')
                                ax_png.axis('off')
                                
                                title_str = (f"Patient: {st.session_state.patient_name}\n"
                                             f"Slice: {slice_idx_png + START_SLICE} (Original Index)\n"
                                             f"Labels Present: {labels_present_str}")
                                ax_png.set_title(title_str, fontsize=8) # Reduced font size for individual slices
                                
                                png_buf_individual = io.BytesIO()
                                fig_png.savefig(png_buf_individual, format='png', dpi=150, bbox_inches='tight', pad_inches=0.05)
                                plt.close(fig_png)
                                png_buf_individual.seek(0)
                                
                                safe_labels_str = labels_present_str.replace(', ', '_').replace(',', '_').replace('/', '-')
                                png_filename = f"{st.session_state.patient_name}_slice_{slice_idx_png + START_SLICE:03d}_labels_{safe_labels_str}.png"
                                zf.writestr(png_filename, png_buf_individual.getvalue())
                        zip_buffer_individual.seek(0)
                        st.session_state.zip_buffer_pngs = zip_buffer_individual # Store in session state
                        st.success(f"PNG ZIP archive ready for patient {st.session_state.patient_name}!")
                        if 'grid_image_buffer' in st.session_state: del st.session_state['grid_image_buffer']


                elif st.session_state.png_download_format == t["high_res_grid_png"]:
                    with st.spinner("Creating high-resolution grid image..."):
                        try:
                            grid_img = create_slice_grid(
                                st.session_state.input_for_vis_np_hwd, # (H,W,D)
                                st.session_state.prediction_rgba_dhw4, # (D,H,W,4)
                                st.session_state.patient_name
                            )
                            
                            if grid_img is not None:
                                img_buffer_grid = io.BytesIO()
                                grid_img.save(img_buffer_grid, format='PNG', quality=95, dpi=(300, 300)) # High quality and DPI
                                img_buffer_grid.seek(0)
                                st.session_state.grid_image_buffer = img_buffer_grid # Store in session state
                                st.success("High-resolution grid PNG ready!")
                                if 'zip_buffer_pngs' in st.session_state: del st.session_state['zip_buffer_pngs']
                            else:
                                st.error("Failed to generate slice grid. See logs if any.")
                        
                        except Exception as e_grid:
                            st.error(f"Failed to generate slice grid: {str(e_grid)}")
                            st.exception(e_grid)
            
            # Download button for Individual Slices ZIP
            if st.session_state.png_download_format == t["individual_slices_zip"] and 'zip_buffer_pngs' in st.session_state and st.session_state.zip_buffer_pngs:
                st.download_button(
                    label=t["download_png"].format(st.session_state.patient_name),
                    data=st.session_state.zip_buffer_pngs,
                    file_name=f"{st.session_state.patient_name}_segmentation_slices.zip",
                    mime="application/zip",
                    key="download_zip_button",
                    on_click=lambda: st.session_state.pop('zip_buffer_pngs', None) # Clear after download
                )

            # Download button for High-Res Grid PNG
            if st.session_state.png_download_format == t["high_res_grid_png"] and 'grid_image_buffer' in st.session_state and st.session_state.grid_image_buffer:
                st.image(st.session_state.grid_image_buffer, caption="Preview of Slice Grid (PNG)", use_column_width=True)
                st.download_button(
                    label=t["download_grid_button"],
                    data=st.session_state.grid_image_buffer,
                    file_name=f"{st.session_state.patient_name}_slice_grid_13x10.png",
                    mime="image/png",
                    key="download_grid_png_button",
                    on_click=lambda: st.session_state.pop('grid_image_buffer', None) # Clear after download
                )

    st.markdown("---")
    st.markdown(f"Timestamp: {st.session_state.current_date} | SegMed v0.2") # Added version

if __name__ == "__main__":
    main()
