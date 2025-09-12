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
import matplotlib.patches as mpatches
import io
import zipfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import math
import gdown
import gc

# --- Configuration ---
DEFAULT_IN_CHANNELS = 4
DEFAULT_OUT_CLASSES = 4 # Incl. background
DEFAULT_BASE_FEATURES = 32
TARGET_HW_SHAPE = (128, 128)
START_SLICE = 0
END_SLICE = 182
TARGET_DEPTH = END_SLICE - START_SLICE # This is 182

# --- Label and Color Definitions ---
LABEL_TO_RGBA = {
    0: (0, 0, 0, 0),       # Background
    1: (255, 0, 0, 255),   # Necrotic (Red)
    2: (0, 255, 0, 255),   # Edema (Green)
    3: (255, 255, 0, 255), # Enhancing (Yellow)
}
SEGMENTATION_LABELS_DICT = {
    1: "Necrotic",
    2: "Edema",
    3: "Enhancing",
}

# --- Translations ---
TRANSLATIONS = {
    "English": {
        "title": "3D Brain Tumour Segmentation",
        "description": """
        SegMed is a Computer-Aided Diagnosis (CAD) system for brain tumor segmentation.
        Created by Marouane Rhazzafe (undergraduate biomedical engineering student, ISSS Settat)
        as a Final Year Project during an internship at Military Hospital MOULAY ISMAIL, Meknès, Morocco.

        Load an example pre-trained model or upload your own 3D U-Net model.
        The app offers two download options for the segmentation:
        1. A standard **NIfTI label map** (`.nii.gz`).
        2. A **ZIP archive of PNG images** for each of the {TARGET_DEPTH} processed slices, with legends.
        **Ensure your uploaded model weights match the U-Net architecture defined here and the configured target depth ({TARGET_DEPTH} slices). The model architecture assumes InstanceNorm3D layers have affine=False and Conv3D layers in blocks have bias=False, while initial/upsampling convs may have bias=True.**
        """,
        "sidebar_header": "⚙️ Configuration",
        "patient_id": "Patient Name/ID",
        "unet_config": "U-Net Architecture",
        "voxel_dims_header": "Voxel Dimensions for Volume Calc (mm)",
        "use_header_dims_label": "Use dimensions from NIfTI header",
        "vox_x_label": "Vox X (mm)",
        "vox_y_label": "Vox Y (mm)",
        "vox_z_label": "Vox Z (mm)",
        "input_channels": "Input Channels (C)",
        "output_classes": "Output Classes (Total, incl. background)",
        "base_features": "Base Features",
        "upload_model": "Upload 3D U-Net Model (.pth)",
        "pretrained_model": "Example Pretrained Model",
        "load_pretrained": "Load Example Pretrained Model",
        "running_on": "Running on",
        "input_files": "📁 Input NIfTI Files",
        "modality_names": ["T1-native (t1n)", "T1-contrast (t1c)", "T2-FLAIR (t2f)", "T2-weighted (t2w)"],
        "run_button": "🚀 Run 3D Segmentation",
        "results_header": "📊 Segmentation Results",
        "volumetric_analysis_header": "🔬 Volumetric Analysis",
        "multi_view": "Multi-View Segmentation Overlay",
        "legend_header": "Segmentation Legend",
        "download_header": "💾 Download Options",
        "nifti_option": "1. NIfTI Label Map",
        "download_nifti": "Download Label Segmentation (.nii.gz)",
        "png_option": "2. PNG Slices (Overlay)",
        "prepare_png": "Prepare PNG Slices for Download",
        "download_png": "Download PNG Slices for {} (.zip)",
        "grid_image_option_label": "Slice Grid Image with Legend",
        "download_grid_image_label": "Download Slice Grid Image (.png)",
        "png_individual_option_label": "Individual Slices with Legend ({TARGET_DEPTH}) (ZIP)",
        "labels": {
            "Background": "Background (Normal Tissue)",
            "Necrotic": "Necrotic/Non-Enhancing",
            "Edema": "Peritumoral Edema",
            "Enhancing": "Enhancing Tumor"
        },
        "volume_label_unit": "cm³"
    },
    "Français": {
        "title": "Segmentation 3D de Tumeurs Cérébrales",
        "description": """
        SegMed est un système de Diagnostic Assisté par Ordinateur (DAO) pour la segmentation des tumeurs cérébrales.
        Réalisé par Marouane Rhazzafe (étudiant en génie biomédical, ISSS Settat)
        dans le cadre de son Projet de Fin d'Études lors d'un stage à l'Hôpital Militaire MOULAY ISMAIL, Meknès, Maroc.

        Chargez un exemple de modèle pré-entraîné ou téléversez votre propre modèle 3D U-Net.
        L'application offre deux options de téléchargement pour la segmentation :
        1. Une **carte d'étiquettes NIfTI** standard (`.nii.gz`).
        2. Une **archive ZIP d'images PNG** pour chacune des {TARGET_DEPTH} tranches traitées, avec légendes.
        **Assurez-vous que les poids de votre modèle téléversé correspondent à l'architecture U-Net définie ici (incluant affine=False pour InstanceNorm, bias=False pour les Conv3D dans les blocs, etc.) et à la profondeur cible configurée ({TARGET_DEPTH} tranches).**
        """,
        "sidebar_header": "⚙️ Configuration",
        "patient_id": "Nom/ID du Patient",
        "unet_config": "Architecture U-Net",
        "voxel_dims_header": "Dimensions des Voxels pour Calcul Vol. (mm)",
        "use_header_dims_label": "Utiliser dimensions de l'en-tête NIfTI",
        "vox_x_label": "Vox X (mm)",
        "vox_y_label": "Vox Y (mm)",
        "vox_z_label": "Vox Z (mm)",
        "input_channels": "Canaux d'Entrée (C)",
        "output_classes": "Classes de Sortie (Total, incl. fond)",
        "base_features": "Fonctions de Base",
        "upload_model": "Télécharger Modèle U-Net 3D (.pth)",
        "pretrained_model": "Exemple de Modèle Pré-entraîné",
        "load_pretrained": "Charger Exemple de Modèle Pré-entraîné",
        "running_on": "Exécution sur",
        "input_files": "📁 Fichiers NIfTI d'Entrée",
        "modality_names": ["T1-natif (t1n)", "T1-contraste (t1c)", "T2-FLAIR (t2f)", "T2-pondéré (t2w)"],
        "run_button": "🚀 Exécuter Segmentation 3D",
        "results_header": "📊 Résultats de Segmentation",
        "volumetric_analysis_header": "🔬 Analyse Volumétrique",
        "multi_view": "Superposition de Segmentation Multi-vues",
        "legend_header": "Légende de Segmentation",
        "download_header": "💾 Options de Téléchargement",
        "nifti_option": "1. Carte d'Étiquettes NIfTI",
        "download_nifti": "Télécharger Segmentation (.nii.gz)",
        "png_option": "2. Tranches PNG (Superposition)",
        "prepare_png": "Préparer Tranches PNG pour Téléchargement",
        "download_png": "Télécharger Tranches PNG pour {} (.zip)",
        "grid_image_option_label": "Image en Grille avec Légende",
        "download_grid_image_label": "Télécharger l'Image en Grille (.png)",
        "png_individual_option_label": "Tranches Individuelles avec Légende ({TARGET_DEPTH}) (ZIP)",
        "labels": {
            "Background": "Arrière-plan (Tissu Normal)",
            "Necrotic": "Nécrotique/Non-Rehaussé",
            "Edema": "Œdème Péritumoral",
            "Enhancing": "Tumeur Rehaussée"
        },
        "volume_label_unit": "cm³"
    }
}


# --- Model Definition ---
class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=False)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int, affine=False)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1, affine=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='trilinear', align_corners=False)
        psi_output = self.relu(g1 + x1)
        psi_output = self.psi(psi_output)
        return x * psi_output

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=DEFAULT_IN_CHANNELS, out_channels=DEFAULT_OUT_CLASSES, base_features=DEFAULT_BASE_FEATURES):
        super(AttentionUNet3D, self).__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=3, padding=1, bias=True),
            nn.InstanceNorm3d(base_features, affine=False),
            nn.ReLU(inplace=True)
        )
        self.encoder1 = self._make_block(base_features, base_features)
        self.encoder2 = self._make_block(base_features, base_features * 2, stride=2)
        self.encoder3 = self._make_block(base_features * 2, base_features * 4, stride=2)
        self.encoder4 = self._make_block(base_features * 4, base_features * 8, stride=2)

        self.up3 = self._make_upsample(base_features * 8, base_features * 4)
        self.attn3 = AttentionGate3D(F_g=base_features * 4, F_l=base_features * 4, F_int=base_features * 2)
        self.decoder3 = self._make_block(base_features * 8, base_features * 4)

        self.up2 = self._make_upsample(base_features * 4, base_features * 2)
        self.attn2 = AttentionGate3D(F_g=base_features * 2, F_l=base_features * 2, F_int=base_features)
        self.decoder2 = self._make_block(base_features * 4, base_features * 2)

        self.up1 = self._make_upsample(base_features * 2, base_features)
        self.attn1 = AttentionGate3D(F_g=base_features, F_l=base_features, F_int=base_features // 2)
        self.decoder1 = self._make_block(base_features * 2, base_features)

        self.final_conv = nn.Conv3d(base_features, out_channels, kernel_size=1, bias=True)

    def _make_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=False),
            nn.ReLU(inplace=True)
        )

    def _make_upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=True),
            nn.InstanceNorm3d(out_channels, affine=False),
            nn.ReLU(inplace=True)
        )

    def _match_spatial_dims(self, tensor_to_pad, target_tensor):
        s_pad = tensor_to_pad.size()
        s_target = target_tensor.size()
        padding = []
        for i in range(3):
            dim_index_in_tensor = 4 - i
            diff = s_target[dim_index_in_tensor] - s_pad[dim_index_in_tensor]
            pad1 = diff // 2
            pad2 = diff - pad1
            padding.extend([pad1, pad2])
        return F.pad(tensor_to_pad, padding)

    def forward(self, x):
        enc_initial = self.initial_conv(x)
        enc1 = self.encoder1(enc_initial)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec3_up = self.up3(enc4)
        if dec3_up.shape[2:] != enc3.shape[2:]: dec3_up = self._match_spatial_dims(dec3_up, enc3)
        att3 = self.attn3(g=dec3_up, x=enc3)
        dec3_cat = torch.cat([dec3_up, att3], dim=1)
        dec3 = self.decoder3(dec3_cat)

        dec2_up = self.up2(dec3)
        if dec2_up.shape[2:] != enc2.shape[2:]: dec2_up = self._match_spatial_dims(dec2_up, enc2)
        att2 = self.attn2(g=dec2_up, x=enc2)
        dec2_cat = torch.cat([dec2_up, att2], dim=1)
        dec2 = self.decoder2(dec2_cat)

        dec1_up = self.up1(dec2)
        if dec1_up.shape[2:] != enc1.shape[2:]: dec1_up = self._match_spatial_dims(dec1_up, enc1)
        att1 = self.attn1(g=dec1_up, x=enc1)
        dec1_cat = torch.cat([dec1_up, att1], dim=1)
        dec1 = self.decoder1(dec1_cat)

        logits = self.final_conv(dec1)
        return logits


# --- Utility Functions ---
def load_nii_and_preprocess(file_path, is_label=False, target_hw_shape=TARGET_HW_SHAPE, slice_range=(START_SLICE, END_SLICE)):
    try:
        img = nib.load(file_path)
        data = np.array(img.dataobj, dtype=(np.int64 if is_label else np.float32))
        s_start, s_end = slice_range
        target_d = s_end - s_start

        if data.ndim == 4 and data.shape[3] == 1: data = np.squeeze(data, axis=3)
        if data.ndim != 3:
            st.warning(f"File {os.path.basename(file_path)}: unexpected dimensions {data.shape}. Expected 3D array."); return None, None, None

        current_h_orig, current_w_orig, current_depth_orig = data.shape

        if current_depth_orig > s_start:
            data_cropped_at_start = data[:, :, s_start:min(s_end, current_depth_orig)]
        else:
            data_cropped_at_start = np.zeros((current_h_orig, current_w_orig, 0), dtype=data.dtype)

        current_depth_after_crop = data_cropped_at_start.shape[2]
        if current_depth_after_crop < target_d:
            padding_needed = target_d - current_depth_after_crop
            pad_config_depth = ((0,0), (0,0), (0, padding_needed))
            volume_adjusted_depth = np.pad(data_cropped_at_start, pad_config_depth, mode='constant', constant_values=0)
        elif current_depth_after_crop > target_d:
            volume_adjusted_depth = data_cropped_at_start[:, :, :target_d]
        else:
            volume_adjusted_depth = data_cropped_at_start

        resized_slices = []
        if volume_adjusted_depth.shape[2] > 0:
            for i in range(volume_adjusted_depth.shape[2]):
                slice_to_resize = volume_adjusted_depth[:, :, i]
                rs_order = 0 if is_label else 1
                rs_mode = 'edge' if is_label else 'reflect'
                rs_aa = not is_label
                resized_slice = resize(slice_to_resize, target_hw_shape, order=rs_order, mode=rs_mode,
                                       anti_aliasing=rs_aa, preserve_range=True)
                resized_slices.append(resized_slice.astype(data.dtype))

        if not resized_slices and target_d > 0:
            st.error(f"File {os.path.basename(file_path)}: No slices generated, target depth {target_d}."); return None,None,None

        final_volume_hwd = np.stack(resized_slices, axis=-1) if resized_slices else \
                           np.zeros((target_hw_shape[0], target_hw_shape[1], 0), dtype=data.dtype)

        if not is_label and final_volume_hwd.size > 0:
            min_v, max_v = np.min(final_volume_hwd), np.max(final_volume_hwd)
            final_volume_hwd = (final_volume_hwd - min_v) / (max_v - min_v) if (max_v - min_v) > 1e-6 else np.zeros_like(final_volume_hwd)

        expected_shape = (target_hw_shape[0], target_hw_shape[1], target_d)
        if final_volume_hwd.shape != expected_shape:
            st.error(f"File {os.path.basename(file_path)}: Final shape {final_volume_hwd.shape} != expected {expected_shape}."); return None,None,None
        return final_volume_hwd, img.affine, img.header
    except Exception as e:
        st.error(f"Error processing NIfTI {os.path.basename(file_path)}: {e}"); st.exception(e); return None,None,None


def labels_to_rgba(label_volume_dhw, num_total_classes, color_map_dict):
    rgba_volume = np.zeros((*label_volume_dhw.shape, 4), dtype=np.uint8)
    for class_value in range(num_total_classes):
        color = color_map_dict.get(class_value, (0,0,0,0))
        mask = (label_volume_dhw == class_value)
        rgba_volume[mask] = color
    return rgba_volume

def draw_horizontal_legend_pil(draw, start_y, image_width, legend_elements, font,
                                 box_size=12, text_offset=4, item_spacing=10, text_fill=(0,0,0,255)):
    total_legend_width = 0
    element_widths = []

    for item in legend_elements:
        try:
            text_bbox = draw.textbbox((0,0), item['label'], font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except AttributeError:
            text_width = font.getsize(item['label'])[0]

        item_width = box_size + text_offset + text_width
        element_widths.append(item_width)
        total_legend_width += item_width

    if legend_elements:
        total_legend_width += item_spacing * (len(legend_elements) - 1)

    current_x = (image_width - total_legend_width) / 2
    if current_x < 5: current_x = 5

    for i, item in enumerate(legend_elements):
        try:
            ascent, descent = font.getmetrics()
            text_height_approx = ascent + descent
        except AttributeError:
            text_height_approx = font.getsize("A")[1]

        box_y_offset = (text_height_approx - box_size) / 2
        box_y = start_y + box_y_offset

        draw.rectangle([current_x, box_y, current_x + box_size, box_y + box_size], fill=item['color'])
        draw.text((current_x + box_size + text_offset, start_y), item['label'], font=font, fill=text_fill)
        current_x += element_widths[i] + item_spacing

def create_slice_grid(input_volume_hwd, rgba_volume_dhw4, patient_name, t, legend_font_pil):
    SLICES_PER_ROW = 13
    MARGIN = 5
    TITLE_HEIGHT = 60
    LEGEND_AREA_HEIGHT = 40
    IMG_H, IMG_W, TOTAL_SLICES = input_volume_hwd.shape

    if TOTAL_SLICES == 0:
        placeholder = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), "No slices for grid.", fill="black")
        return placeholder

    num_rows = math.ceil(TOTAL_SLICES / SLICES_PER_ROW)
    grid_content_w = (IMG_W * SLICES_PER_ROW) + (MARGIN * (SLICES_PER_ROW - 1))
    grid_content_h = (IMG_H * num_rows) + (MARGIN * (num_rows - 1))

    grid_w = max(grid_content_w, 300)
    grid_h = TITLE_HEIGHT + grid_content_h + LEGEND_AREA_HEIGHT

    grid_img = Image.new('RGB', (int(grid_w), int(grid_h)), color='white')
    draw = ImageDraw.Draw(grid_img)

    try:
        title_font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        title_font = ImageFont.load_default()

    draw.text((10, 10), f"Pt: {patient_name} - {TOTAL_SLICES} slices ({SLICES_PER_ROW}x{num_rows})", font=title_font, fill='black')

    for i in range(TOTAL_SLICES):
        r, c = i // SLICES_PER_ROW, i % SLICES_PER_ROW
        px, py = c * (IMG_W + MARGIN), TITLE_HEIGHT + r * (IMG_H + MARGIN)

        slice_data_float = input_volume_hwd[:, :, i]
        base_img_pil = Image.fromarray((slice_data_float * 255).astype(np.uint8)).convert('RGBA')

        segslice_np = rgba_volume_dhw4[i, :, :, :]
        overlay_pil = Image.fromarray(segslice_np).convert('RGBA')

        composite_pil = Image.alpha_composite(base_img_pil, overlay_pil)
        grid_img.paste(composite_pil.convert('RGB'), (int(px), int(py)))

    LEGEND_BG_SWATCH_COLOR = (220, 220, 220, 255)
    legend_elements_for_grid = [{"label": t['labels']['Background'], "color": LEGEND_BG_SWATCH_COLOR}]
    for val, name_key in SEGMENTATION_LABELS_DICT.items():
        legend_elements_for_grid.append({
            "label": t['labels'].get(name_key, name_key),
            "color": LABEL_TO_RGBA.get(val, (0,0,0,255))
        })

    legend_start_y = TITLE_HEIGHT + grid_content_h + (MARGIN if num_rows > 0 else 0) + 5
    draw_horizontal_legend_pil(draw, legend_start_y, grid_w, legend_elements_for_grid, legend_font_pil)

    return grid_img

default_session_state = {
    'model_loaded':None,'device':torch.device('cuda'if torch.cuda.is_available()else'cpu'),
    'patient_name':"UnknownPatient",'current_date':datetime.now().strftime("%d %B %Y, %H:%M:%S")+" (Local)",
    'language':"English",'use_header_dimensions_for_volume':True,'voxel_dim_x':1.0,'voxel_dim_y':1.0,'voxel_dim_z':1.0,
    'prediction_label_dhw':None,'prediction_rgba_dhw4':None,'input_for_vis_np_hwd':None,
    'output_affine':None,'output_header':None,'zip_buffer_pngs':None,'grid_image_buffer':None,'png_download_type':None,
}
for k,v in default_session_state.items():
    if k not in st.session_state: st.session_state[k]=v

def clear_segmentation_results():
    keys=['prediction_label_dhw','prediction_rgba_dhw4','input_for_vis_np_hwd','output_affine','output_header',
          'zip_buffer_pngs','grid_image_buffer','png_download_type']
    for key in keys: st.session_state[key] = None

def apply_page_styling():
    st.markdown(f"""<style>
    .stApp {{
        background-image: url("https://raw.githubusercontent.com/Vwoudka/segmed/main/.devcontainer/iStock-1452990966-modified-26cda7e8-4ee1-4a98-b681-f8a249f82c52-768x432.jpg");
        background-size: contain; background-position: center center;
        background-repeat: no-repeat; background-attachment: scroll;
    }}
    .main .block-container {{
        background-color: #000000;
        color: white !important;
        border-radius:10px; padding:2rem; margin-top:2rem; margin-bottom:2rem;
        box-shadow:0 4px 12px rgba(0,0,0,0.0); border:1px solid rgba(255,255,255,0.15);
    }}
    .main .block-container label,
    .main .block-container .stMarkdown p,
    .main .block-container .stMetricLabel,
    .main .block-container .stMetricValue,
    .main .block-container .stCaption,
    .main .block-container .streamlit-expanderHeader p
    {{
        color: black !important;
    }}
    .main .block-container .stAlert *,
    .main .block-container .stAlert a
    {{
        color: inherit !important;
    }}
    .main .block-container .stButton>button {{
        color: #0E1117;
    }}
    .css-1d391kg, [data-testid="stSidebar"] {{
        background-color:rgba(0,0,0,1)!important;
    }}
    </style>""", unsafe_allow_html=True)

def display_volumetric_analysis(label_vol_dhw, vox_vol_mm3, seg_map_dict, trans, dim_src_info):
    st.subheader(trans["volumetric_analysis_header"])
    st.caption(f"Voxel Dimensions Used: {dim_src_info}")
    if not seg_map_dict: st.info("No labels for volume calculation."); return

    num_metrics = len(seg_map_dict)
    cols = st.columns(num_metrics if num_metrics > 0 else 1)
    for i, (val, name_key) in enumerate(seg_map_dict.items()):
        with cols[i % num_metrics]:
            disp_name = trans["labels"].get(name_key, name_key)
            vox_count = np.sum(label_vol_dhw == val)
            vol_cm3 = (vox_count * vox_vol_mm3) / 1000.0
            st.metric(label=disp_name, value=f"{vol_cm3:.2f} {trans.get('volume_label_unit','cm³')}")

# Cached function for model loading
@st.cache_resource
def load_model_cached(model_source, arch_params, device):
    in_c, out_c, base_f = arch_params
    model = AttentionUNet3D(in_c, out_c, base_f)
    pth_to_remove = None
    try:
        with st.spinner("Loading model into memory... (This happens only once per session)"):
            if isinstance(model_source, str) and model_source.startswith("GDRIVE_ID:"):
                gdrive_id = model_source.split(":", 1)[1]
                st.info(f"Downloading pretrained model from Google Drive...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_f:
                    pth_to_remove = tmp_f.name
                gdown.download(id=gdrive_id, output=pth_to_remove, quiet=False)
                loaded_data = torch.load(pth_to_remove, map_location=device)
            else:
                loaded_data = torch.load(io.BytesIO(model_source.getvalue()), map_location=device)
            if isinstance(loaded_data, dict) and 'model_state_dict' in loaded_data:
                model.load_state_dict(loaded_data['model_state_dict'])
            else:
                model.load_state_dict(loaded_data)
            model.to(device).eval()
            st.success("Model loaded and cached successfully!")
            return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.exception(e)
        return None
    finally:
        if pth_to_remove and os.path.exists(pth_to_remove):
            os.remove(pth_to_remove)

# --- Main App Execution ---
if __name__ == "__main__":
    st.set_page_config(page_title="SegMed",layout="wide",initial_sidebar_state="expanded")
    apply_page_styling()

    sel_lang = st.sidebar.selectbox("Language/Langue",list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.language),key="main_language_selector")
    if sel_lang != st.session_state.language:
        st.session_state.language=sel_lang; st.experimental_rerun()
    t = TRANSLATIONS[st.session_state.language]

    try:
        FONT_FOR_LEGEND_PIL = ImageFont.truetype("arial.ttf", 10)
    except IOError:
        FONT_FOR_LEGEND_PIL = ImageFont.load_default()

    title_col, gh_col = st.columns([0.9,0.1])
    with title_col: st.title(t["title"])
    with gh_col: st.markdown("<div style='text-align:right;margin-top:20px;'><a href='https://github.com/Vwoudka/segmed' target='_blank'><img src='https://github.githubassets.com/favicons/favicon.png' width='30' alt='GitHub'></a></div>",unsafe_allow_html=True)
    st.markdown(t["description"].format(TARGET_DEPTH=TARGET_DEPTH))

    st.sidebar.header(t["sidebar_header"])
    st.session_state.patient_name = st.sidebar.text_input(t["patient_id"],value=st.session_state.patient_name,key="patient_name_input")

    st.sidebar.subheader(t["unet_config"])
    p_in_c=st.sidebar.number_input(t["input_channels"],min_value=1,value=DEFAULT_IN_CHANNELS,key="param_in_channels")
    p_out_c=st.sidebar.number_input(t["output_classes"],min_value=1,value=DEFAULT_OUT_CLASSES,key="param_out_classes")
    p_base_f=st.sidebar.number_input(t["base_features"],min_value=8,value=DEFAULT_BASE_FEATURES,step=16,key="param_base_features")

    st.sidebar.subheader(t["voxel_dims_header"])
    use_hdr_dims = st.sidebar.checkbox(t["use_header_dims_label"],value=st.session_state.use_header_dimensions_for_volume,key="use_header_dims_check")
    st.session_state.use_header_dimensions_for_volume = use_hdr_dims

    val_vx, val_vy, val_vz = st.session_state.voxel_dim_x, st.session_state.voxel_dim_y, st.session_state.voxel_dim_z
    if use_hdr_dims and st.session_state.output_header:
        try:
            hz=st.session_state.output_header.get_zooms()
            val_vx,val_vy,val_vz=abs(float(hz[0])),abs(float(hz[1])),abs(float(hz[2]))
            if not all(d > 1e-6 for d in (val_vx,val_vy,val_vz)):
                val_vx,val_vy,val_vz = st.session_state.voxel_dim_x,st.session_state.voxel_dim_y,st.session_state.voxel_dim_z
                st.sidebar.caption("Header dims zero/neg. Using manual.")
        except Exception:
            val_vx,val_vy,val_vz = st.session_state.voxel_dim_x,st.session_state.voxel_dim_y,st.session_state.voxel_dim_z
            st.sidebar.caption("Header read failed. Using manual.")

    vox_cs=st.sidebar.columns(3)
    man_vx_in=vox_cs[0].number_input(t["vox_x_label"],value=val_vx,format="%.4f",step=1e-4,disabled=use_hdr_dims,key="manual_vx_val_in")
    man_vy_in=vox_cs[1].number_input(t["vox_y_label"],value=val_vy,format="%.4f",step=1e-4,disabled=use_hdr_dims,key="manual_vy_val_in")
    man_vz_in=vox_cs[2].number_input(t["vox_z_label"],value=val_vz,format="%.4f",step=1e-4,disabled=use_hdr_dims,key="manual_vz_val_in")
    if not use_hdr_dims:
        st.session_state.voxel_dim_x,st.session_state.voxel_dim_y,st.session_state.voxel_dim_z = man_vx_in,man_vy_in,man_vz_in

    model_source_to_load = None
    up_model_f_obj = st.sidebar.file_uploader(t["upload_model"], type=["pth"], key="model_uploader_main")

    st.sidebar.header(t["pretrained_model"])
    if st.sidebar.button(t["load_pretrained"], key="load_example_model_main_btn"):
        model_source_to_load = "GDRIVE_ID:1nYmcyYQPkfxXFGdh9i2QVeakYNvdS4yH"
        clear_segmentation_results()

    if up_model_f_obj:
        model_source_to_load = up_model_f_obj
        clear_segmentation_results()

    if model_source_to_load:
        arch_params = (p_in_c, p_out_c, p_base_f)
        st.session_state.model_loaded = load_model_cached(model_source_to_load, arch_params, st.session_state.device)

    st.header(t["input_files"])
    num_uploaders=p_in_c if p_in_c>0 else 1; mod_t_names=t.get("modality_names",[])
    uploader_disp_labels=[mod_t_names[j]if j<len(mod_t_names)else f"Modality {j+1}" for j in range(num_uploaders)]
    nii_cols_upload=st.columns(num_uploaders); uploaded_nii_f_objs=[None]*num_uploaders
    for i in range(num_uploaders):
        with nii_cols_upload[i]:uploaded_nii_f_objs[i]=st.file_uploader(f"Upload {uploader_disp_labels[i]}",["nii.gz","nii"],key=f"nii_upload_widget_{i}")

    if st.button(t["run_button"],disabled=(not st.session_state.model_loaded or not all(uploaded_nii_f_objs)),key="run_segmentation_main_btn"):
        with st.spinner(f"Segmenting {TARGET_DEPTH} slices... This may take some time."):
            clear_segmentation_results()
            try:
                proc_vols_list, aff_list, hdr_list = [],[],[]
                for i, file_obj in enumerate(uploaded_nii_f_objs):
                    st.info(f"Processing: {uploader_disp_labels[i]}...")
                    with tempfile.NamedTemporaryFile(delete=True, suffix=".nii.gz") as temp_nii:
                        temp_nii.write(file_obj.getvalue())
                        temp_nii_path = temp_nii.name
                        vol_hwd, aff_m, nii_hdr = load_nii_and_preprocess(temp_nii_path, False, TARGET_HW_SHAPE, (START_SLICE, END_SLICE))
                    if vol_hwd is None:
                        st.error(f"Processing {uploader_disp_labels[i]} failed.");
                        st.stop()
                    proc_vols_list.append(vol_hwd)
                    if i==0:
                        aff_list.append(aff_m)
                        hdr_list.append(nii_hdr)
                
                st.info("Preparing data tensor for the model...")
                stacked_chwd=np.stack(proc_vols_list,axis=0)
                del proc_vols_list
                gc.collect()

                input_cdhw=stacked_chwd.transpose(0,3,1,2)
                del stacked_chwd
                gc.collect()

                input_tensor=torch.from_numpy(np.expand_dims(input_cdhw,0)).float().to(st.session_state.device)
                del input_cdhw
                gc.collect()

                st.info(f"Model input tensor shape: {input_tensor.shape} (N,C,D,H,W)")
                with torch.no_grad():
                    logits_ncdhw=st.session_state.model_loaded(input_tensor)

                if logits_ncdhw.shape[2:]!=input_tensor.shape[2:]:
                    st.warning(f"Model output DHW {logits_ncdhw.shape[2:]} differs from input {input_tensor.shape[2:]}. Interpolating.")
                    logits_ncdhw=F.interpolate(logits_ncdhw,size=input_tensor.shape[2:],mode='trilinear',align_corners=False)

                pred_labels_dhw_arr=torch.argmax(logits_ncdhw.squeeze(0),dim=0).cpu().numpy().astype(np.uint8)
                st.session_state.prediction_label_dhw=pred_labels_dhw_arr
                st.session_state.prediction_rgba_dhw4=labels_to_rgba(pred_labels_dhw_arr,p_out_c,LABEL_TO_RGBA)

                with tempfile.NamedTemporaryFile(delete=True, suffix=".nii.gz") as temp_nii_vis:
                    temp_nii_vis.write(uploaded_nii_f_objs[0].getvalue())
                    st.session_state.input_for_vis_np_hwd, _, _ = load_nii_and_preprocess(temp_nii_vis.name, False,TARGET_HW_SHAPE,(START_SLICE,END_SLICE))

                st.session_state.output_affine=aff_list[0];st.session_state.output_header=hdr_list[0]
                st.success("Segmentation complete!")

            except Exception as e:st.error(f"Segmentation process error: {e}");st.exception(e);clear_segmentation_results()


    if st.session_state.prediction_label_dhw is not None:
        st.header(t["results_header"])
        input_vis_hwd = st.session_state.input_for_vis_np_hwd
        pred_labels_dhw = st.session_state.prediction_label_dhw
        pred_rgba_dhw4 = st.session_state.prediction_rgba_dhw4
        mid_d, mid_h, mid_w = pred_labels_dhw.shape[0]//2, pred_labels_dhw.shape[1]//2, pred_labels_dhw.shape[2]//2

        plot_configs = [
            {"name":"Axial", "data":{"input":input_vis_hwd[:,:,mid_d], "rgba":pred_rgba_dhw4[mid_d,:,:,:], "title":f"Axial Slice: {mid_d+START_SLICE}(orig)/{mid_d}(proc)", "aspect":"equal"}},
            {"name":"Sagittal", "data":{"input":input_vis_hwd[:,mid_w,:], "rgba":np.transpose(pred_rgba_dhw4[:,:,mid_w,:],(1,0,2)), "title":f"Sagittal (W-slice:{mid_w})", "aspect":input_vis_hwd.shape[2]/(input_vis_hwd.shape[0] or 1)}},
            {"name":"Coronal", "data":{"input":input_vis_hwd[mid_h,:,:], "rgba":np.transpose(pred_rgba_dhw4[:,mid_h,:,:],(1,0,2)), "title":f"Coronal (H-slice:{mid_h})", "aspect":input_vis_hwd.shape[2]/(input_vis_hwd.shape[1] or 1)}}
        ]
        st.subheader(t["multi_view"]); vis_cols = st.columns(len(plot_configs))
        for i, cfg in enumerate(plot_configs):
            pd = cfg["data"]
            with vis_cols[i]:
                st.markdown(f"**{pd['title']}**")
                fig, ax = plt.subplots(figsize=(5,5));
                ax.imshow(pd["input"], cmap='gray', aspect=pd["aspect"])
                ax.imshow(pd["rgba"], aspect=pd["aspect"])
                ax.axis('off'); st.pyplot(fig); plt.close(fig)

        st.subheader(t["legend_header"])
        legend_html_items = []
        bg_col_rgba = LABEL_TO_RGBA.get(0, (128,128,128,30))
        legend_html_items.append(f"<div style='display:flex;align-items:center;'><div style='width:20px;height:20px;background-color:rgba({bg_col_rgba[0]},{bg_col_rgba[1]},{bg_col_rgba[2]},{bg_col_rgba[3]/255.0});margin-right:8px;border:1px dashed #aaa;'></div><span style='font-size:0.9em;'>{t['labels']['Background']}</span></div>")
        for val, name_key in SEGMENTATION_LABELS_DICT.items():
            color = LABEL_TO_RGBA.get(val, (0,0,0,255))
            rgba_css_val = f"rgba({color[0]},{color[1]},{color[2]},{color[3]/255.0})"
            disp_name = t['labels'].get(name_key, name_key)
            legend_html_items.append(f"<div style='display:flex;align-items:center;'><div style='width:20px;height:20px;background-color:{rgba_css_val};margin-right:8px;border:1px solid #555;'></div><span style='font-size:0.9em;'>{disp_name}</span></div>")

        legend_html_str = "".join(legend_html_items)
        st.markdown(f"<div style='display:flex;flex-direction:row;flex-wrap:wrap;justify-content:center;align-items:center;gap:20px;padding:10px;background-color:rgba(70,70,70,0.85);border-radius:5px;'>{legend_html_str}</div>", unsafe_allow_html=True)

        voxel_volume_mm3, dim_source = 0.0, "N/A"
        if st.session_state.use_header_dimensions_for_volume and st.session_state.output_header:
            try:
                zooms = st.session_state.output_header.get_zooms(); vx,vy,vz = abs(zooms[0]),abs(zooms[1]),abs(zooms[2])
                if all(d > 1e-9 for d in (vx,vy,vz)): voxel_volume_mm3 = vx*vy*vz; dim_source=f"Header ({vx:.3f}x{vy:.3f}x{vz:.3f} mm)"
                else: st.sidebar.caption("Warn: Header dims zero/neg.")
            except Exception as e: st.sidebar.caption(f"Warn: Header zoom err: {e}")

        if voxel_volume_mm3 <= 1e-9:
            mvx,mvy,mvz = st.session_state.voxel_dim_x,st.session_state.voxel_dim_y,st.session_state.voxel_dim_z
            if all(d > 1e-9 for d in (mvx,mvy,mvz)): voxel_volume_mm3 = mvx*mvy*mvz; dim_source=f"Manual ({mvx:.3f}x{mvy:.3f}x{mvz:.3f} mm)"
            else: dim_source = "Manual input dims invalid."

        if isinstance(voxel_volume_mm3,float) and voxel_volume_mm3 > 1e-9:
            display_volumetric_analysis(pred_labels_dhw,voxel_volume_mm3,SEGMENTATION_LABELS_DICT,t,dim_source)
        else: st.error(f"Voxel volume invalid ({dim_source}). Cannot calculate. Check NIfTI/manual input.")

        st.header(t["download_header"]); dl_c1,dl_c2 = st.columns(2)
        with dl_c1:
            st.subheader(t["nifti_option"])
            if st.session_state.output_affine is not None and st.session_state.output_header:
                try:
                    nii_data_hwd=np.transpose(pred_labels_dhw,(1,2,0)).astype(np.uint8)
                    nii_img_obj=nib.Nifti1Image(nii_data_hwd,st.session_state.output_affine,st.session_state.output_header)
                    nii_img_obj.set_filename("segmentation.nii.gz")
                    bio_nii=io.BytesIO(nii_img_obj.to_bytes())
                    st.download_button(t["download_nifti"],bio_nii,f"{st.session_state.patient_name}_seg.nii.gz","application/gzip",key="dl_nifti_main_btn")
                except Exception as e: st.error(f"NIfTI prep error: {e}")
        with dl_c2:
            st.subheader(t["png_option"])
            png_format_choice=st.radio("Format:",[t["grid_image_option_label"], t["png_individual_option_label"].format(TARGET_DEPTH=TARGET_DEPTH)],key="png_format_choice_radio")
            if st.button(t["prepare_png"],key="prepare_png_main_btn"):
                if png_format_choice==t["grid_image_option_label"]:
                    with st.spinner("Generating grid image..."):
                        try:
                            grid_img_pil=create_slice_grid(input_vis_hwd,pred_rgba_dhw4,st.session_state.patient_name, t, FONT_FOR_LEGEND_PIL)
                            bio_grid=io.BytesIO();grid_img_pil.save(bio_grid,'PNG',quality=95);bio_grid.seek(0)
                            st.session_state.grid_image_buffer=bio_grid;st.session_state.png_download_type="grid"
                            st.image(grid_img_pil,caption=f"Preview ({TARGET_DEPTH} slices with legend)",use_column_width=True);st.success("Grid ready.")
                        except Exception as e:st.error(f"Grid image error: {e}")
                else:
                    with st.spinner(f"Generating {TARGET_DEPTH} PNGs with legends for ZIP..."):
                        try:
                            zip_bio_out=io.BytesIO()
                            with zipfile.ZipFile(zip_bio_out,"w",zipfile.ZIP_DEFLATED) as zf_out:
                                for idx in range(TARGET_DEPTH):
                                    slice_in_hw=input_vis_hwd[:,:,idx]; slice_rgba_hw4=pred_rgba_dhw4[idx,:,:,:]; slice_lbl_hw=pred_labels_dhw[idx,:,:]
                                    unique_lbl_vals=np.unique(slice_lbl_hw)
                                    present_lbl_names=[SEGMENTATION_LABELS_DICT[val] for val in unique_lbl_vals if val in SEGMENTATION_LABELS_DICT]
                                    labels_found_str=", ".join(present_lbl_names) if present_lbl_names else "No Tumor Labels"

                                    fig_png_slice, ax_png_slice = plt.subplots(figsize=(6, 6.2), dpi=150)
                                    fig_png_slice.patch.set_facecolor('white')

                                    ax_png_slice.imshow(slice_in_hw, cmap='gray', aspect='equal')
                                    ax_png_slice.imshow(slice_rgba_hw4, aspect='equal')
                                    ax_png_slice.axis('off')
                                    ax_png_slice.set_title(f"Patient: {st.session_state.patient_name}\nSlice: {idx+START_SLICE}(orig)/{idx}(proc) | Labels: {labels_found_str}", fontsize=7, color='black')

                                    legend_patches = []
                                    bg_label_text = t['labels'].get("Background", "Background")
                                    bg_color_rgba = (0.8, 0.8, 0.8, 1.0)
                                    legend_patches.append(mpatches.Patch(color=bg_color_rgba, label=bg_label_text))
                                    for label_val, label_name_key in SEGMENTATION_LABELS_DICT.items():
                                        text_label = t['labels'].get(label_name_key, label_name_key)
                                        color_rgba = np.array(LABEL_TO_RGBA.get(label_val, (0,0,0,255))) / 255.0
                                        legend_patches.append(mpatches.Patch(color=color_rgba, label=text_label))

                                    fig_png_slice.legend(handles=legend_patches, loc='lower center', ncol=len(legend_patches),
                                                         bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize='x-small')
                                    fig_png_slice.subplots_adjust(bottom=0.12, top=0.9)

                                    png_byte_buffer=io.BytesIO()
                                    fig_png_slice.savefig(png_byte_buffer,format='png',bbox_inches='tight', facecolor=fig_png_slice.get_facecolor()); plt.close(fig_png_slice); png_byte_buffer.seek(0)

                                    fn_safe_labels="_".join(labels_found_str.replace("/","-").split(", ")).replace(" ","_") if present_lbl_names else "NoTumor"
                                    zf_out.writestr(f"{st.session_state.patient_name}_slice_{idx+START_SLICE:03d}_{fn_safe_labels}.png",png_byte_buffer.getvalue())
                            st.session_state.zip_buffer_pngs=zip_bio_out;st.session_state.png_download_type="zip";st.success("ZIP archive ready.")
                        except Exception as e:st.error(f"PNG ZIP creation error: {e}"); st.exception(e)

            if st.session_state.png_download_type=="grid" and st.session_state.grid_image_buffer:
                st.download_button(t["download_grid_image_label"],st.session_state.grid_image_buffer,f"{st.session_state.patient_name}_slice_grid.png","image/png",key="dl_grid_img_main_btn")
            elif st.session_state.png_download_type=="zip" and st.session_state.zip_buffer_pngs:
                st.download_button(t["download_png"].format(st.session_state.patient_name),st.session_state.zip_buffer_pngs,f"{st.session_state.patient_name}_slices_legend.zip","application/zip",key="dl_zip_archive_main_btn")
    else:
        st.info("Segmentation results, volumetric analysis, and download options will appear here after running segmentation.")

    st.markdown("---");st.markdown(f"Timestamp: {st.session_state.current_date}");st.caption(f"{t['running_on']}: {st.session_state.device}")

