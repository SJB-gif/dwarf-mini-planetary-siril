"""
DWARF Mini Planetary / Lunar / Solar Preprocessor for Siril 1.4+
=================================================================

Optimised version
-----------------
This version keeps the successful processing path from the previous script, but
reduces unnecessary work and adds optional CPU/GPU acceleration.

Main changes:
- Siril OSC/CFA debayering is still used first.
- No manual Bayer/demosaic fallback code is kept.
- Reference selection samples the whole sequence, not just the start.
- Registration is parallelised across CPU worker threads.
- Optional GPU FFT acceleration is used when CuPy is already installed.
- Registration can be done on downsampled luminance frames, then applied at full resolution.
- Debug aligned-preview FITS writes are optional and off by default.
- Console output is kept to summary lines; Siril may still print its own link/calibrate output.

Expected folder layout:
    selected/project folder/
        lights/
            *.fit / *.fits / *.fit.fz / *.fits.fz

Outputs:
    result/DWARF_Mini_<target>_stack_raw.fit
    result/DWARF_Mini_<target>_stack_base.fit
    result/DWARF_Mini_<target>_stack_base.fit
"""

import os
import sys
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import sirilpy as s

s.ensure_installed("PyQt6", "numpy", "astropy")

from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMessageBox,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QGroupBox,
    QRadioButton,
    QCheckBox,
    QSpinBox,
    QDialogButtonBox,
    QLineEdit,
    QProgressDialog,
)
from sirilpy import LogColor
from astropy.io import fits
import numpy as np

try:
    import cupy as cp  # type: ignore
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


APP_NAME = "DWARF Mini Planetary Preprocessor Optimised"
MIN_SIRIL_VERSION = "1.4.0"

LIGHTS_DIR = "lights"
PROCESS_DIR = "process"
RESULT_DIR = "result"
ALIGNED_TMP_DIR = "aligned_tmp"

DEFAULT_TARGET_NAME = "Moon"
# Output basenames are built at runtime from the selected target name.

RAW_SEQUENCE_BASENAME = "lights"
RAW_SEQUENCE_NAME = "lights_"

SUPPORTED_FITS_EXTENSIONS = (
    ".fit",
    ".fits",
    ".fit.fz",
    ".fits.fz",
)

PROCESSED_FITS_EXTENSIONS = (
    ".fit",
    ".fits",
)

FILTER_OPTIONS = ["Astro filter (UV/IR)", "Dual-Band"]
OUTPUT_OPTIONS = ["Monochrome luminance (recommended)", "RGB colour"]
STACK_MODE_OPTIONS = ["Weighted all valid frames (recommended)", "Keep best percentage only"]
REFERENCE_MODE_OPTIONS = ["Median-quality sampled frame (recommended)", "Sharpest sampled frame"]

FEATURE_PATCH_SIZE = 96
FEATURE_MAX_COUNT = 18
FEATURE_MIN_DISTANCE = 48
MIN_FEATURES_FOR_MEDIAN = 3


@dataclass
class Settings:
    project_dir: str
    target_name: str
    filter_mode: str
    output_mode: str
    stack_mode: str
    keep_percent: int
    reference_mode: str
    reg_downsample: int
    cpu_workers: int
    use_gpu_fft: bool
    clean_outside_limb: bool
    save_debug_previews: bool


# -------------------------
# Siril / UI helpers
# -------------------------

def qpath(path):
    return '"' + os.path.abspath(path).replace("\\", "/") + '"'


def run_cmd(siril, *args):
    siril.cmd(*args)


def log_stage(siril, message, color=LogColor.GREEN):
    siril.log(message, color)


def sanitize_target_name(name):
    """
    Make a user-supplied target name safe for filenames and FITS header cards.
    Examples:
        "Moon" -> "Moon"
        "Jupiter GRS" -> "Jupiter_GRS"
        "Saturn / Titan" -> "Saturn_Titan"
    """
    cleaned = str(name or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or DEFAULT_TARGET_NAME


def infer_target_name(path):
    """
    Guess a sensible default target name from the project path.
    The user can override this in the dialog.
    """
    known_targets = [
        "Moon",
        "Sun",
        "Mercury",
        "Venus",
        "Mars",
        "Jupiter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Pluto",
    ]

    parts = []
    current = os.path.abspath(path or "")
    while True:
        head, tail = os.path.split(current)
        if tail:
            parts.append(tail)
        if not head or head == current:
            break
        current = head

    lower_parts = [p.lower() for p in parts]
    for target in known_targets:
        if target.lower() in lower_parts:
            return target

    return DEFAULT_TARGET_NAME


def output_basenames_for_target(target_name):
    safe_target = sanitize_target_name(target_name)
    basename = f"DWARF_Mini_{safe_target}_stack"
    return basename + "_raw", basename + "_base"


def list_fits_files(folder):
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if name.startswith("."):
            continue
        if name.lower().endswith(SUPPORTED_FITS_EXTENSIONS):
            files.append(path)
    return sorted(files)


def list_processed_fits_files(folder, prefix):
    files = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        if name.startswith("."):
            continue
        lower_name = name.lower()
        if not lower_name.startswith(prefix.lower()):
            continue
        if lower_name.endswith(PROCESSED_FITS_EXTENSIONS):
            files.append(path)
    return sorted(files)


def expected_keep_count(frame_count, keep_percent):
    if frame_count <= 0:
        return 0
    minimum = min(3, frame_count)
    return min(frame_count, max(minimum, int(round(frame_count * keep_percent / 100.0))))


class ProgressReporter:
    def __init__(self, title, total_steps):
        self.total_steps = max(1, int(total_steps))
        self.current = 0
        self.dialog = QProgressDialog(title, "Cancel", 0, self.total_steps)
        self.dialog.setWindowTitle(APP_NAME)
        self.dialog.setMinimumDuration(0)
        self.dialog.setAutoClose(False)
        self.dialog.setAutoReset(False)
        self.dialog.show()
        QApplication.processEvents()

    def set_label(self, text):
        self.dialog.setLabelText(text)
        QApplication.processEvents()
        self.check_cancelled()

    def step(self, amount=1, text=None):
        if text is not None:
            self.dialog.setLabelText(text)
        self.current = min(self.total_steps, self.current + int(amount))
        self.dialog.setValue(self.current)
        QApplication.processEvents()
        self.check_cancelled()

    def set_value(self, value, text=None):
        if text is not None:
            self.dialog.setLabelText(text)
        self.current = min(self.total_steps, max(0, int(value)))
        self.dialog.setValue(self.current)
        QApplication.processEvents()
        self.check_cancelled()

    def close(self):
        self.dialog.setValue(self.total_steps)
        self.dialog.close()
        QApplication.processEvents()

    def check_cancelled(self):
        if self.dialog.wasCanceled():
            raise RuntimeError("Processing cancelled by user.")


class ProcessingSettingsDialog(QDialog):
    def __init__(self, default_project_dir, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME + " Settings")
        self.resize(760, 520)

        main_layout = QVBoxLayout(self)

        folder_group = QGroupBox("Input folder")
        folder_layout = QGridLayout(folder_group)

        folder_layout.addWidget(QLabel("Parent folder containing lights/:"), 0, 0, 1, 2)

        self.folder_edit = QLineEdit(default_project_dir or os.path.expanduser("~"))
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_folder)

        folder_layout.addWidget(self.folder_edit, 1, 0)
        folder_layout.addWidget(self.browse_button, 1, 1)

        folder_layout.addWidget(QLabel("Target name for output files:"), 2, 0, 1, 2)
        self.target_edit = QLineEdit(infer_target_name(default_project_dir))
        folder_layout.addWidget(self.target_edit, 3, 0, 1, 2)

        self.files_found_label = QLabel("")
        folder_layout.addWidget(self.files_found_label, 4, 0, 1, 2)

        main_layout.addWidget(folder_group)

        options_group = QGroupBox("Processing options")
        options_layout = QGridLayout(options_group)

        filter_group = QGroupBox("Filter used")
        filter_layout = QVBoxLayout(filter_group)
        self.astro_filter_radio = QRadioButton("Astro filter (UV/IR)")
        self.dual_band_radio = QRadioButton("Dual-Band")
        self.astro_filter_radio.setChecked(True)
        filter_layout.addWidget(self.astro_filter_radio)
        filter_layout.addWidget(self.dual_band_radio)

        output_group = QGroupBox("Output mode")
        output_layout = QVBoxLayout(output_group)
        self.mono_radio = QRadioButton("Monochrome luminance (recommended for Moon/Sun)")
        self.rgb_radio = QRadioButton("RGB colour")
        self.mono_radio.setChecked(True)
        output_layout.addWidget(self.mono_radio)
        output_layout.addWidget(self.rgb_radio)

        stack_group = QGroupBox("Stacking")
        stack_layout = QVBoxLayout(stack_group)
        self.weighted_checkbox = QCheckBox("Use weighted all valid frames (recommended)")
        self.weighted_checkbox.setChecked(True)

        percent_row = QHBoxLayout()
        self.keep_percentage_checkbox = QCheckBox("Use best-frame percentage instead")
        self.keep_percentage_checkbox.setChecked(False)
        self.keep_percent_spin = QSpinBox()
        self.keep_percent_spin.setRange(1, 100)
        self.keep_percent_spin.setValue(80)
        self.keep_percent_spin.setSuffix("%")
        self.keep_percent_spin.setEnabled(False)

        percent_row.addWidget(self.keep_percentage_checkbox)
        percent_row.addWidget(self.keep_percent_spin)
        percent_row.addStretch()

        self.weighted_checkbox.toggled.connect(self.update_keep_percent_enabled)
        self.keep_percentage_checkbox.toggled.connect(self.update_keep_percent_enabled)
        self.weighted_checkbox.toggled.connect(self.keep_percentage_checkbox.setDisabled)
        self.keep_percentage_checkbox.toggled.connect(self.weighted_checkbox.setDisabled)

        stack_layout.addWidget(self.weighted_checkbox)
        stack_layout.addLayout(percent_row)

        reference_group = QGroupBox("Reference frame")
        reference_layout = QVBoxLayout(reference_group)
        self.median_reference_radio = QRadioButton("Median-quality sampled frame (recommended)")
        self.sharpest_reference_radio = QRadioButton("Sharpest sampled frame")
        self.median_reference_radio.setChecked(True)
        reference_layout.addWidget(self.median_reference_radio)
        reference_layout.addWidget(self.sharpest_reference_radio)

        speed_group = QGroupBox("Speed / optimisation")
        speed_layout = QGridLayout(speed_group)

        self.downsample_spin = QSpinBox()
        self.downsample_spin.setRange(1, 4)
        self.downsample_spin.setValue(2)
        self.downsample_spin.setToolTip("Registration is done on a smaller luminance image, then applied at full resolution.")

        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(1, max(1, (os.cpu_count() or 4)))
        self.worker_spin.setValue(max(1, min((os.cpu_count() or 4) - 1, 8)))
        self.worker_spin.setToolTip("CPU worker threads for frame registration. Use fewer if your machine becomes unresponsive.")

        self.gpu_checkbox = QCheckBox("Use GPU FFT if CuPy is installed")
        self.gpu_checkbox.setChecked(CUPY_AVAILABLE)
        self.gpu_checkbox.setEnabled(CUPY_AVAILABLE)
        self.gpu_checkbox.setToolTip("Experimental. Uses CuPy only if already installed in Siril's Python environment.")

        self.clean_outside_limb_checkbox = QCheckBox("Clean outside-limb background in display outputs")
        self.clean_outside_limb_checkbox.setChecked(True)
        self.clean_outside_limb_checkbox.setToolTip(
            "Display-only cleanup. It does not affect the raw stack and does not mask frames during stacking."
        )

        self.debug_previews_checkbox = QCheckBox("Save debug aligned preview FITS files")
        self.debug_previews_checkbox.setChecked(False)

        speed_layout.addWidget(QLabel("Registration downsample:"), 0, 0)
        speed_layout.addWidget(self.downsample_spin, 0, 1)
        speed_layout.addWidget(QLabel("CPU worker threads:"), 1, 0)
        speed_layout.addWidget(self.worker_spin, 1, 1)
        speed_layout.addWidget(self.gpu_checkbox, 2, 0, 1, 2)
        speed_layout.addWidget(self.clean_outside_limb_checkbox, 3, 0, 1, 2)
        speed_layout.addWidget(self.debug_previews_checkbox, 4, 0, 1, 2)

        options_layout.addWidget(filter_group, 0, 0)
        options_layout.addWidget(output_group, 0, 1)
        options_layout.addWidget(stack_group, 1, 0, 1, 2)
        options_layout.addWidget(reference_group, 2, 0, 1, 2)
        options_layout.addWidget(speed_group, 3, 0, 1, 2)

        main_layout.addWidget(options_group)

        gpu_note = "CuPy detected." if CUPY_AVAILABLE else "CuPy not detected; GPU FFT will be unavailable."
        note = QLabel(
            "Recommended: Astro filter, Monochrome luminance, Weighted all valid frames, "
            "Median-quality sampled frame, registration downsample 2. " + gpu_note
        )
        note.setWordWrap(True)
        main_layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons)

        self.folder_edit.textChanged.connect(self.update_file_count)
        self.update_file_count()
        self.update_keep_percent_enabled()

    def browse_folder(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select the parent folder that contains the 'lights' folder",
            self.folder_edit.text() or os.path.expanduser("~"),
            QFileDialog.Option.ShowDirsOnly,
        )
        if selected:
            self.folder_edit.setText(selected)
            current_target = sanitize_target_name(self.target_edit.text())
            if current_target in ("", DEFAULT_TARGET_NAME, "planetary"):
                self.target_edit.setText(infer_target_name(selected))

    def update_keep_percent_enabled(self):
        use_percentage = self.keep_percentage_checkbox.isChecked() and not self.weighted_checkbox.isChecked()
        self.keep_percent_spin.setEnabled(use_percentage)

    def update_file_count(self):
        folder = self.folder_edit.text().strip()
        lights_dir = os.path.join(folder, LIGHTS_DIR)
        if os.path.isdir(lights_dir):
            count = len(list_fits_files(lights_dir))
            self.files_found_label.setText(f"Found {count} FITS light frame(s) in lights/.")
            if count > 0:
                default_percent = 80 if count >= 50 else 100
                self.keep_percent_spin.setValue(default_percent)
        else:
            self.files_found_label.setText("No lights/ folder found at this location.")

    def accept(self):
        folder = self.folder_edit.text().strip()
        if not folder:
            QMessageBox.critical(self, "Missing folder", "Please select a parent folder.")
            return

        lights_dir = os.path.join(folder, LIGHTS_DIR)
        if not os.path.isdir(lights_dir):
            QMessageBox.critical(self, "Invalid folder", "The selected folder must contain a lights/ subfolder.")
            return

        if len(list_fits_files(lights_dir)) == 0:
            QMessageBox.critical(self, "No FITS files", "No FITS files were found in the lights/ folder.")
            return

        self.target_edit.setText(sanitize_target_name(self.target_edit.text()))

        super().accept()

    def values(self):
        filter_mode = "Dual-Band" if self.dual_band_radio.isChecked() else "Astro filter (UV/IR)"
        output_mode = "RGB colour" if self.rgb_radio.isChecked() else "Monochrome luminance (recommended)"

        if self.weighted_checkbox.isChecked():
            stack_mode = "Weighted all valid frames (recommended)"
            keep_percent = 100
        else:
            stack_mode = "Keep best percentage only"
            keep_percent = int(self.keep_percent_spin.value())

        reference_mode = (
            "Sharpest sampled frame"
            if self.sharpest_reference_radio.isChecked()
            else "Median-quality sampled frame (recommended)"
        )

        return Settings(
            project_dir=self.folder_edit.text().strip(),
            target_name=sanitize_target_name(self.target_edit.text()),
            filter_mode=filter_mode,
            output_mode=output_mode,
            stack_mode=stack_mode,
            keep_percent=keep_percent,
            reference_mode=reference_mode,
            reg_downsample=int(self.downsample_spin.value()),
            cpu_workers=int(self.worker_spin.value()),
            use_gpu_fft=bool(self.gpu_checkbox.isChecked() and CUPY_AVAILABLE),
            clean_outside_limb=bool(self.clean_outside_limb_checkbox.isChecked()),
            save_debug_previews=bool(self.debug_previews_checkbox.isChecked()),
        )


def guess_default_project_dir(siril):
    current = siril.get_siril_wd()
    if current and os.path.isdir(os.path.join(current, LIGHTS_DIR)):
        return current
    if current and os.path.basename(current).lower() == LIGHTS_DIR:
        parent = os.path.dirname(current)
        if os.path.isdir(os.path.join(parent, LIGHTS_DIR)):
            return parent
    return current or os.path.expanduser("~")


def choose_processing_settings(siril):
    dialog = ProcessingSettingsDialog(guess_default_project_dir(siril))
    if dialog.exec() != QDialog.DialogCode.Accepted:
        raise RuntimeError("Processing settings cancelled.")
    return dialog.values()


# -------------------------
# Siril preprocessing
# -------------------------

def reset_output_folders(project_dir):
    process_dir = os.path.join(project_dir, PROCESS_DIR)
    result_dir = os.path.join(project_dir, RESULT_DIR)

    if os.path.isdir(process_dir):
        shutil.rmtree(process_dir)

    os.makedirs(process_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    return process_dir, result_dir


def debayer_with_siril(siril, project_dir, process_dir):
    run_cmd(siril, "cd", qpath(project_dir))
    os.chdir(project_dir)

    run_cmd(siril, "cd", LIGHTS_DIR)
    run_cmd(siril, "link", RAW_SEQUENCE_BASENAME, "-out=../process")
    run_cmd(siril, "cd", "../process")

    run_cmd(siril, "calibrate", RAW_SEQUENCE_NAME, "-cfa", "-equalize_cfa", "-debayer")

    debayered = list_processed_fits_files(process_dir, "pp_" + RAW_SEQUENCE_NAME)
    if not debayered:
        raise RuntimeError("Siril debayering did not produce pp_lights_ files in process/.")

    return debayered


# -------------------------
# FITS / image helpers
# -------------------------

def first_image_hdu(hdul):
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None:
            data = hdu.data
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                return hdu
    raise ValueError("No image data found in FITS file.")


def read_fits_image(path):
    with fits.open(path, memmap=False) as hdul:
        hdu = first_image_hdu(hdul)
        data = np.asarray(hdu.data)
        header = hdu.header.copy()
    return data, header


def normalize_float(data):
    arr = np.asarray(data, dtype=np.float32)
    arr = np.nan_to_num(arr, copy=False)
    return arr


def channel_first_rgb(data):
    arr = normalize_float(data)

    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        return arr[:3]

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return np.moveaxis(arr[..., :3], -1, 0)

    return None


def make_luminance_from_data(data):
    arr = normalize_float(data)
    rgb = channel_first_rgb(arr)

    if rgb is not None:
        return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)

    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)

    if arr.ndim > 2:
        return arr[0].astype(np.float32, copy=False)

    raise ValueError(f"Unsupported image shape: {arr.shape}")


def make_output_image(data, output_mode):
    arr = normalize_float(data)

    if output_mode.startswith("Monochrome"):
        return make_luminance_from_data(arr)

    rgb = channel_first_rgb(arr)
    if rgb is not None:
        return rgb.astype(np.float32, copy=False)

    return arr.astype(np.float32, copy=False)


def downsample_mean(img, factor):
    if factor <= 1:
        return np.asarray(img, dtype=np.float32)

    arr = np.asarray(img, dtype=np.float32)
    h = (arr.shape[0] // factor) * factor
    w = (arr.shape[1] // factor) * factor

    if h == 0 or w == 0:
        return arr

    cropped = arr[:h, :w]
    return cropped.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3)).astype(np.float32)


def robust_display_scale(img):
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32)

    lo, hi = np.percentile(img[finite], [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)

    return np.clip((img - lo) / (hi - lo), 0, 1).astype(np.float32, copy=False)


def object_mask_and_centroid(lum):
    img = robust_display_scale(lum)

    if img.size == 0:
        return np.zeros_like(img, dtype=bool), np.array([0.0, 0.0], dtype=np.float32)

    threshold = max(float(np.percentile(img, 75)), float(np.median(img) + 0.35 * np.std(img)))
    mask = img > threshold

    if np.count_nonzero(mask) < max(25, img.size * 0.002):
        threshold = float(np.percentile(img, 60))
        mask = img > threshold

    if np.count_nonzero(mask) == 0:
        return mask, np.array([img.shape[0] / 2, img.shape[1] / 2], dtype=np.float32)

    yy, xx = np.nonzero(mask)
    weights = img[yy, xx]
    weight_sum = float(np.sum(weights))

    if weight_sum <= 0:
        centroid = np.array([np.mean(yy), np.mean(xx)], dtype=np.float32)
    else:
        centroid = np.array(
            [np.sum(yy * weights) / weight_sum, np.sum(xx * weights) / weight_sum],
            dtype=np.float32,
        )

    return mask, centroid


def crop_around_centroid(img, centroid, crop_fraction=0.72):
    h, w = img.shape
    crop_h = min(max(64, int(h * crop_fraction)), h)
    crop_w = min(max(64, int(w * crop_fraction)), w)

    cy, cx = centroid
    y0 = int(round(cy - crop_h / 2))
    x0 = int(round(cx - crop_w / 2))
    y0 = max(0, min(h - crop_h, y0))
    x0 = max(0, min(w - crop_w, x0))

    return img[y0:y0 + crop_h, x0:x0 + crop_w]


def laplacian_variance(img):
    img = robust_display_scale(img)
    if img.shape[0] < 3 or img.shape[1] < 3:
        return 0.0

    centre = img[1:-1, 1:-1]
    lap = (
        -4.0 * centre
        + img[:-2, 1:-1]
        + img[2:, 1:-1]
        + img[1:-1, :-2]
        + img[1:-1, 2:]
    )
    return float(np.var(lap))


def gradient_score_map(img):
    img = robust_display_scale(img)
    score = np.zeros_like(img, dtype=np.float32)

    if img.shape[0] < 3 or img.shape[1] < 3:
        return score

    gy = img[2:, 1:-1] - img[:-2, 1:-1]
    gx = img[1:-1, 2:] - img[1:-1, :-2]
    score[1:-1, 1:-1] = np.sqrt(gx * gx + gy * gy)
    return score


def safe_extract_patch(img, center_y, center_x, size):
    half = size // 2
    y0 = int(round(center_y)) - half
    x0 = int(round(center_x)) - half
    y1 = y0 + size
    x1 = x0 + size

    if y0 < 0 or x0 < 0 or y1 > img.shape[0] or x1 > img.shape[1]:
        return None

    return img[y0:y1, x0:x1]


def find_tracking_features(reference_lum):
    img = robust_display_scale(reference_lum)
    h, w = img.shape

    if h < 64 or w < 64:
        return []

    patch_size = min(FEATURE_PATCH_SIZE, max(32, min(h, w) // 4))
    patch_size = int(patch_size // 2 * 2)
    half = patch_size // 2
    margin = half + 4

    mask, centroid = object_mask_and_centroid(reference_lum)
    score = gradient_score_map(reference_lum)

    candidate_mask = mask & (img > 0.05) & (img < 0.98)
    candidate_mask[:margin, :] = False
    candidate_mask[-margin:, :] = False
    candidate_mask[:, :margin] = False
    candidate_mask[:, -margin:] = False

    stride = 3
    ys, xs = np.nonzero(candidate_mask[::stride, ::stride])
    ys = ys * stride
    xs = xs * stride

    if len(ys) == 0:
        return [{"y": float(centroid[0]), "x": float(centroid[1]), "size": patch_size}]

    values = score[ys, xs]
    order = np.argsort(values)[::-1]

    features = []
    min_dist_sq = FEATURE_MIN_DISTANCE * FEATURE_MIN_DISTANCE

    for idx in order[:10000]:
        y = float(ys[idx])
        x = float(xs[idx])

        if safe_extract_patch(reference_lum, y, x, patch_size) is None:
            continue

        too_close = False
        for feature in features:
            dy = y - feature["y"]
            dx = x - feature["x"]
            if dy * dy + dx * dx < min_dist_sq:
                too_close = True
                break

        if too_close:
            continue

        features.append({"y": y, "x": x, "size": patch_size})

        if len(features) >= FEATURE_MAX_COUNT:
            break

    if not features:
        features.append({"y": float(centroid[0]), "x": float(centroid[1]), "size": patch_size})

    return features


# -------------------------
# Registration / FFT helpers
# -------------------------

def prepare_registration_image(lum):
    img = robust_display_scale(lum)
    img -= np.median(img)
    std = np.std(img)
    if std > 0:
        img /= std

    y_window = np.hanning(img.shape[0]).astype(np.float32)
    x_window = np.hanning(img.shape[1]).astype(np.float32)
    img = img * y_window[:, None] * x_window[None, :]

    return img.astype(np.float32, copy=False)


def quadratic_subpixel_offset(left, centre, right):
    denom = left - 2.0 * centre + right
    if abs(float(denom)) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if not np.isfinite(offset):
        return 0.0
    return float(np.clip(offset, -0.5, 0.5))


def phase_shift_cpu(reference, moving):
    ref = prepare_registration_image(reference)
    mov = prepare_registration_image(moving)

    if ref.shape != mov.shape:
        raise ValueError(f"Shape mismatch: reference={ref.shape}, moving={mov.shape}")

    ref_fft = np.fft.fft2(ref)
    mov_fft = np.fft.fft2(mov)
    cross_power = ref_fft * np.conj(mov_fft)
    magnitude = np.abs(cross_power)
    cross_power /= np.maximum(magnitude, 1e-12)

    correlation_abs = np.abs(np.fft.ifft2(cross_power))
    return phase_peak_from_correlation(correlation_abs, reference.shape)


def phase_shift_gpu(reference, moving):
    if not CUPY_AVAILABLE or cp is None:
        return phase_shift_cpu(reference, moving)

    ref = cp.asarray(prepare_registration_image(reference))
    mov = cp.asarray(prepare_registration_image(moving))

    if ref.shape != mov.shape:
        raise ValueError(f"Shape mismatch: reference={ref.shape}, moving={mov.shape}")

    ref_fft = cp.fft.fft2(ref)
    mov_fft = cp.fft.fft2(mov)
    cross_power = ref_fft * cp.conj(mov_fft)
    magnitude = cp.abs(cross_power)
    cross_power = cross_power / cp.maximum(magnitude, 1e-12)

    correlation_abs_gpu = cp.abs(cp.fft.ifft2(cross_power))
    correlation_abs = cp.asnumpy(correlation_abs_gpu)

    return phase_peak_from_correlation(correlation_abs, reference.shape)


def phase_peak_from_correlation(correlation_abs, shape):
    peak = np.unravel_index(np.argmax(correlation_abs), correlation_abs.shape)
    peak_y, peak_x = peak
    peak_value = float(correlation_abs[peak])

    y_minus = correlation_abs[(peak_y - 1) % correlation_abs.shape[0], peak_x]
    y_centre = correlation_abs[peak_y, peak_x]
    y_plus = correlation_abs[(peak_y + 1) % correlation_abs.shape[0], peak_x]

    x_minus = correlation_abs[peak_y, (peak_x - 1) % correlation_abs.shape[1]]
    x_centre = correlation_abs[peak_y, peak_x]
    x_plus = correlation_abs[peak_y, (peak_x + 1) % correlation_abs.shape[1]]

    sub_y = quadratic_subpixel_offset(y_minus, y_centre, y_plus)
    sub_x = quadratic_subpixel_offset(x_minus, x_centre, x_plus)

    shifts = np.array([float(peak_y) + sub_y, float(peak_x) + sub_x], dtype=np.float32)

    half_shape = np.array(shape, dtype=np.float32) / 2.0
    full_shape = np.array(shape, dtype=np.float32)
    shifts[shifts > half_shape] -= full_shape[shifts > half_shape]

    return float(shifts[0]), float(shifts[1]), peak_value


def phase_shift(reference, moving, use_gpu=False):
    if use_gpu:
        return phase_shift_gpu(reference, moving)
    return phase_shift_cpu(reference, moving)


def estimate_shift(reference_lum, reference_centroid, moving_lum, use_gpu=False):
    _, moving_centroid = object_mask_and_centroid(moving_lum)
    centroid_shift = reference_centroid - moving_centroid

    try:
        fft_y, fft_x, peak = phase_shift(reference_lum, moving_lum, use_gpu=use_gpu)
        fft_shift = np.array([fft_y, fft_x], dtype=np.float32)
    except Exception:
        return float(centroid_shift[0]), float(centroid_shift[1]), 0.0, "centroid"

    disagreement = float(np.linalg.norm(fft_shift - centroid_shift))
    max_reasonable_disagreement = max(20.0, 0.08 * max(reference_lum.shape))

    if disagreement > max_reasonable_disagreement or peak < 0.015:
        return float(centroid_shift[0]), float(centroid_shift[1]), peak, "centroid"

    return float(fft_y), float(fft_x), peak, "fft"


def fit_affine_source_transform(features, feature_shifts, good_mask):
    good_features = [f for f, good in zip(features, good_mask) if good]
    good_shifts = np.asarray([s for s, good in zip(feature_shifts, good_mask) if good], dtype=np.float32)

    if len(good_features) < 4:
        return None

    ref_y = np.asarray([f["y"] for f in good_features], dtype=np.float32)
    ref_x = np.asarray([f["x"] for f in good_features], dtype=np.float32)
    src_y = ref_y - good_shifts[:, 0]
    src_x = ref_x - good_shifts[:, 1]

    design = np.column_stack([np.ones_like(ref_y), ref_y, ref_x])

    try:
        coeff_y, _, _, _ = np.linalg.lstsq(design, src_y, rcond=None)
        coeff_x, _, _, _ = np.linalg.lstsq(design, src_x, rcond=None)
    except Exception:
        return None

    matrix = np.vstack([coeff_y, coeff_x]).astype(np.float32)
    linear = matrix[:, 1:3]

    if not np.all(np.isfinite(linear)):
        return None

    det = float(np.linalg.det(linear))
    if det < 0.75 or det > 1.25:
        return None

    return matrix


def scale_affine_matrix_to_full_resolution(matrix, scale):
    if matrix is None:
        return None
    full = np.asarray(matrix, dtype=np.float32).copy()
    full[0, 0] *= scale
    full[1, 0] *= scale
    return full


def estimate_feature_shift(reference_lum, reference_centroid, features, moving_lum, use_gpu=False):
    global_y, global_x, global_peak, global_method = estimate_shift(
        reference_lum,
        reference_centroid,
        moving_lum,
        use_gpu=use_gpu,
    )

    feature_shifts = []
    feature_subset = []
    peaks = []

    for feature in features:
        size = int(feature["size"])
        ref_patch = safe_extract_patch(reference_lum, feature["y"], feature["x"], size)
        if ref_patch is None:
            continue

        expected_y = feature["y"] - global_y
        expected_x = feature["x"] - global_x
        mov_patch = safe_extract_patch(moving_lum, expected_y, expected_x, size)
        if mov_patch is None:
            continue

        try:
            residual_y, residual_x, peak = phase_shift(ref_patch, mov_patch, use_gpu=use_gpu)
        except Exception:
            continue

        if abs(residual_y) > size * 0.35 or abs(residual_x) > size * 0.35:
            continue
        if peak < 0.010:
            continue

        feature_subset.append(feature)
        feature_shifts.append([global_y + residual_y, global_x + residual_x])
        peaks.append(peak)

    if len(feature_shifts) < MIN_FEATURES_FOR_MEDIAN:
        return global_y, global_x, global_peak, global_method, None

    shifts = np.asarray(feature_shifts, dtype=np.float32)
    median_shift = np.median(shifts, axis=0)
    distances = np.sqrt(np.sum((shifts - median_shift[None, :]) ** 2, axis=1))
    mad = float(np.median(np.abs(distances - np.median(distances))))
    threshold = max(2.0, 3.0 * mad)
    good = distances <= threshold

    if np.count_nonzero(good) >= MIN_FEATURES_FOR_MEDIAN:
        final_shift = np.median(shifts[good], axis=0)
        final_peak = float(np.median(np.asarray(peaks)[good]))
        affine_matrix = fit_affine_source_transform(feature_subset, feature_shifts, good)
        method = (
            f"affine:{int(np.count_nonzero(good))}"
            if affine_matrix is not None
            else f"features:{int(np.count_nonzero(good))}"
        )
        return float(final_shift[0]), float(final_shift[1]), final_peak, method, affine_matrix

    return float(median_shift[0]), float(median_shift[1]), float(np.median(peaks)), f"features:{len(feature_shifts)}", None


def load_registration_luminance(path, downsample_factor):
    data, _header = read_fits_image(path)
    lum = make_luminance_from_data(data)
    return downsample_mean(lum, downsample_factor)


def score_luminance(lum):
    _, centroid = object_mask_and_centroid(lum)
    roi = crop_around_centroid(lum, centroid)
    quality = laplacian_variance(roi)
    return quality, centroid


def choose_reference_frame(files, reference_mode, downsample_factor, progress=None):
    sample_count = min(80, len(files))
    if sample_count == 1:
        sample_indices = [0]
    else:
        sample_indices = sorted(set(np.linspace(0, len(files) - 1, sample_count, dtype=int).tolist()))

    sample_metrics = []

    for sample_number, file_index in enumerate(sample_indices, start=1):
        lum = load_registration_luminance(files[file_index], downsample_factor)
        quality, centroid = score_luminance(lum)
        sample_metrics.append((quality, file_index, lum, centroid))
        if progress is not None:
            progress.step(text=f"Sampling reference frame {sample_number}/{len(sample_indices)}")

    sample_metrics.sort(key=lambda item: item[0])

    if reference_mode.startswith("Sharpest"):
        return sample_metrics[-1]

    return sample_metrics[len(sample_metrics) // 2]


def register_one_frame(
    item,
    reference_lum,
    reference_centroid,
    tracking_features,
    downsample_factor,
    use_gpu,
):
    index, path = item
    lum = load_registration_luminance(path, downsample_factor)
    quality, _centroid = score_luminance(lum)

    shift_y_small, shift_x_small, peak, method, affine_matrix_small = estimate_feature_shift(
        reference_lum,
        reference_centroid,
        tracking_features,
        lum,
        use_gpu=use_gpu,
    )

    scale = downsample_factor
    shift_y_full = shift_y_small * scale
    shift_x_full = shift_x_small * scale
    affine_full = scale_affine_matrix_to_full_resolution(affine_matrix_small, scale)

    data, _header = read_fits_image(path)
    full_h = data.shape[-2] if data.ndim == 3 and data.shape[0] in (3, 4) else data.shape[0]
    full_w = data.shape[-1] if data.ndim == 3 and data.shape[0] in (3, 4) else data.shape[1]
    max_shift = 0.35 * max(full_h, full_w)
    valid = abs(shift_y_full) <= max_shift and abs(shift_x_full) <= max_shift

    return {
        "index": index,
        "path": path,
        "quality": float(quality),
        "shift_y": float(shift_y_full),
        "shift_x": float(shift_x_full),
        "peak": float(peak),
        "method": method,
        "affine": affine_full,
        "valid": valid,
    }


def score_and_register_frames(siril, files, settings, progress=None):
    reference_quality, reference_index, reference_lum, reference_centroid = choose_reference_frame(
        files,
        settings.reference_mode,
        settings.reg_downsample,
        progress=progress,
    )

    tracking_features = find_tracking_features(reference_lum)

    log_stage(
        siril,
        f"Reference: {os.path.basename(files[reference_index])}; quality={reference_quality:.6f}; "
        f"features={len(tracking_features)}; downsample={settings.reg_downsample}x",
        LogColor.GREEN,
    )

    # GPU FFT is kept single-worker by default to avoid multiple threads fighting
    # over the same CUDA context. CPU registration can use multiple workers.
    workers = 1 if settings.use_gpu_fft else max(1, settings.cpu_workers)

    metrics = []
    method_counts = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                register_one_frame,
                (i, path),
                reference_lum,
                reference_centroid,
                tracking_features,
                settings.reg_downsample,
                settings.use_gpu_fft,
            ): i
            for i, path in enumerate(files, start=1)
        }

        completed = 0
        for future in as_completed(future_map):
            metric = future.result()
            metrics.append(metric)

            method_key = str(metric["method"]).split(":")[0]
            method_counts[method_key] = method_counts.get(method_key, 0) + 1

            completed += 1
            if progress is not None:
                progress.step(text=f"Registering frame {completed}/{len(files)}")

    metrics.sort(key=lambda m: m["index"])

    valid_count = sum(1 for m in metrics if m["valid"])
    method_summary = ", ".join(f"{key}={value}" for key, value in sorted(method_counts.items()))
    log_stage(siril, f"Registration: {valid_count}/{len(metrics)} valid frames. Methods: {method_summary}", LogColor.GREEN)

    _reference_data, reference_header = read_fits_image(files[reference_index])
    reference_info = {
        "path": files[reference_index],
        "filename": os.path.basename(files[reference_index]),
        "index": int(reference_index + 1),
        "quality": float(reference_quality),
        "header": reference_header.copy(),
        "method_summary": method_summary,
        "valid_count": int(valid_count),
    }

    return metrics, reference_info


# -------------------------
# Warping / stacking
# -------------------------

class GridCache:
    def __init__(self):
        self.cache = {}

    def get(self, shape):
        if shape not in self.cache:
            h, w = shape
            self.cache[shape] = np.indices((h, w), dtype=np.float32)
        return self.cache[shape]


def shift_no_wrap_2d(img, shift_y, shift_x, fill_value=0.0):
    src = np.asarray(img, dtype=np.float32)
    h, w = src.shape
    dst = np.full((h, w), fill_value, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    sy = float(shift_y)
    sx = float(shift_x)

    out_y = np.arange(h, dtype=np.float32)
    out_x = np.arange(w, dtype=np.float32)
    src_y = out_y - sy
    src_x = out_x - sx

    valid_y = (src_y >= 0.0) & (src_y < h - 1)
    valid_x = (src_x >= 0.0) & (src_x < w - 1)

    if not np.any(valid_y) or not np.any(valid_x):
        return dst, mask

    oy = np.where(valid_y)[0]
    ox = np.where(valid_x)[0]

    y0 = np.floor(src_y[valid_y]).astype(np.int32)
    x0 = np.floor(src_x[valid_x]).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = (src_y[valid_y] - y0.astype(np.float32)).astype(np.float32)
    wx = (src_x[valid_x] - x0.astype(np.float32)).astype(np.float32)

    top_left = src[np.ix_(y0, x0)]
    top_right = src[np.ix_(y0, x1)]
    bottom_left = src[np.ix_(y1, x0)]
    bottom_right = src[np.ix_(y1, x1)]

    interp_top = top_left * (1.0 - wx)[None, :] + top_right * wx[None, :]
    interp_bottom = bottom_left * (1.0 - wx)[None, :] + bottom_right * wx[None, :]
    interp = interp_top * (1.0 - wy)[:, None] + interp_bottom * wy[:, None]

    dst[np.ix_(oy, ox)] = interp
    mask[np.ix_(oy, ox)] = 1.0

    return dst, mask


def affine_no_wrap_2d(img, matrix, fill_value=0.0, grid_cache=None):
    src = np.asarray(img, dtype=np.float32)
    h, w = src.shape
    dst = np.full((h, w), fill_value, dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (2, 3):
        return shift_no_wrap_2d(src, 0.0, 0.0, fill_value)

    if grid_cache is None:
        yy, xx = np.indices((h, w), dtype=np.float32)
    else:
        yy, xx = grid_cache.get((h, w))

    src_y = matrix[0, 0] + matrix[0, 1] * yy + matrix[0, 2] * xx
    src_x = matrix[1, 0] + matrix[1, 1] * yy + matrix[1, 2] * xx

    valid = (src_y >= 0.0) & (src_y < h - 1) & (src_x >= 0.0) & (src_x < w - 1)
    if not np.any(valid):
        return dst, mask

    y0 = np.floor(src_y[valid]).astype(np.int32)
    x0 = np.floor(src_x[valid]).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1
    wy = (src_y[valid] - y0.astype(np.float32)).astype(np.float32)
    wx = (src_x[valid] - x0.astype(np.float32)).astype(np.float32)

    sampled = (
        src[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + src[y1, x0] * wy * (1.0 - wx)
        + src[y0, x1] * (1.0 - wy) * wx
        + src[y1, x1] * wy * wx
    )

    dst[valid] = sampled
    mask[valid] = 1.0
    return dst, mask


def transform_no_wrap(data, shift_y, shift_x, fill_value=0.0, affine_matrix=None, grid_cache=None):
    arr = np.asarray(data, dtype=np.float32)

    if arr.ndim == 2:
        if affine_matrix is not None:
            return affine_no_wrap_2d(arr, affine_matrix, fill_value, grid_cache=grid_cache)
        return shift_no_wrap_2d(arr, shift_y, shift_x, fill_value)

    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        shifted_channels = []
        mask = None
        for c in range(arr.shape[0]):
            if affine_matrix is not None:
                shifted, mask = affine_no_wrap_2d(arr[c], affine_matrix, fill_value, grid_cache=grid_cache)
            else:
                shifted, mask = shift_no_wrap_2d(arr[c], shift_y, shift_x, fill_value)
            shifted_channels.append(shifted)
        return np.stack(shifted_channels, axis=0), mask

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        cf = np.moveaxis(arr[..., :3], -1, 0)
        shifted, mask = transform_no_wrap(cf, shift_y, shift_x, fill_value, affine_matrix, grid_cache)
        return shifted, mask

    raise ValueError(f"Unsupported image shape for transforming: {arr.shape}")


def output_luminance(image):
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        return (0.2126 * arr[0] + 0.7152 * arr[1] + 0.0722 * arr[2]).astype(np.float32)
    return arr.astype(np.float32, copy=False)


def blur2d_small(src, iterations=1):
    """Small dependency-free blur for soft masks."""
    out = np.asarray(src, dtype=np.float32)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="edge")
        out = (
            padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:]
            + 2.0 * padded[1:-1, :-2] + 4.0 * padded[1:-1, 1:-1] + 2.0 * padded[1:-1, 2:]
            + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
        ) / 16.0
    return out.astype(np.float32)


def dilate_mask(mask, iterations=1):
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return out


def erode_mask(mask, iterations=1):
    out = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
            & padded[:-2, :-2]
            & padded[:-2, 2:]
            & padded[2:, :-2]
            & padded[2:, 2:]
        )
    return out


def keep_largest_component(mask):
    """
    Keep only the largest 8-connected component in a binary mask.

    This removes isolated sky speckles without scipy/scikit-image.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    best_pixels = None
    best_count = 0

    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue

        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        pixels = []

        while stack:
            y, x = stack.pop()
            pixels.append((y, x))

            for ny in (y - 1, y, y + 1):
                if ny < 0 or ny >= h:
                    continue
                for nx in (x - 1, x, x + 1):
                    if nx < 0 or nx >= w or (ny == y and nx == x):
                        continue
                    if mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))

        if len(pixels) > best_count:
            best_count = len(pixels)
            best_pixels = pixels

    out = np.zeros(mask.shape, dtype=bool)
    if best_pixels:
        yy = [p[0] for p in best_pixels]
        xx = [p[1] for p in best_pixels]
        out[yy, xx] = True

    return out


def make_final_lunar_alpha(display_stack):
    """
    Build a single final display mask from the finished stack.

    This is intentionally NOT used during stacking. It only suppresses outside-sky
    halo in display products, so the raw stack remains untouched and the dark
    terminator is not cut off by per-frame masks.
    """
    lum = output_luminance(display_stack)
    finite = np.isfinite(lum)
    if not np.any(finite):
        return np.ones_like(lum, dtype=np.float32)

    values = lum[finite]
    p5, p35, p995 = np.percentile(values, [5.0, 35.0, 99.5])
    sky = values[values <= p35]
    if sky.size < 50:
        sky = values

    sky_med = float(np.median(sky))
    sky_mad = float(np.median(np.abs(sky - sky_med)))
    sky_sigma = max(1.4826 * sky_mad, 1e-6)
    dynamic = max(float(p995 - p5), 1e-6)

    # Conservative body threshold. It should include the lunar disc but not the
    # faint outside halo. The largest component step removes sky speckles.
    threshold = sky_med + max(5.0 * sky_sigma, 0.030 * dynamic)
    body = (lum > threshold) & finite

    body = erode_mask(body, iterations=1)
    body = dilate_mask(body, iterations=3)
    body = keep_largest_component(body)

    # Expand to ensure the true limb is inside the mask, then feather just the
    # outer transition. This avoids the hard cutoff caused by per-frame masking.
    body = dilate_mask(body, iterations=3)
    alpha = blur2d_small(body.astype(np.float32), iterations=4)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    return alpha


def clean_outside_limb_display(display_stack):
    """
    Display-only outside-limb cleanup. Raw stack is not altered.
    """
    arr = np.asarray(display_stack, dtype=np.float32)
    alpha = make_final_lunar_alpha(arr)

    if arr.ndim == 3:
        return (arr * alpha[None, :, :]).astype(np.float32)

    return (arr * alpha).astype(np.float32)


def photometric_stats(image):
    lum = output_luminance(image)
    finite = np.isfinite(lum)
    if not np.any(finite):
        return 0.0, 1.0

    values = lum[finite]
    p60, p995 = np.percentile(values, [60, 99.5])
    surface_mask = finite & (lum > p60) & (lum < p995)

    if np.count_nonzero(surface_mask) < max(100, lum.size * 0.01):
        p5, p995 = np.percentile(values, [5, 99.5])
        surface_mask = finite & (lum > p5) & (lum < p995)

    surface_values = lum[surface_mask] if np.any(surface_mask) else values
    median = float(np.median(surface_values))
    q10, q90 = np.percentile(surface_values, [10, 90])
    spread = float(q90 - q10)

    if not np.isfinite(spread) or spread <= 1e-6:
        spread = float(np.std(surface_values))

    if not np.isfinite(spread) or spread <= 1e-6:
        spread = 1.0

    return median, spread


def match_image_to_reference(image, reference_median, reference_spread):
    arr = np.asarray(image, dtype=np.float32)
    median, spread = photometric_stats(arr)
    scale = reference_spread / max(spread, 1e-6)
    return ((arr - median) * scale + reference_median).astype(np.float32, copy=False)


def clean_lunar_display_stack(stacked):
    arr = np.asarray(stacked, dtype=np.float32)
    lum = output_luminance(arr)
    finite = np.isfinite(lum)

    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)

    values = lum[finite]
    black = float(np.percentile(values, 0.2))
    white = float(np.percentile(values, 99.85))

    if white <= black:
        black = float(np.min(values))
        white = float(np.max(values))

    if white <= black:
        return arr.astype(np.float32, copy=False)

    display = (arr - black) / (white - black)
    display = np.clip(display, 0.0, 1.0)
    display = np.power(display, 0.90).astype(np.float32)

    return display


def unsharp_mask(image, amount=0.75):
    arr = np.asarray(image, dtype=np.float32)

    def blur2d(src):
        padded = np.pad(src, 1, mode="edge")
        return (
            padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:]
            + 2 * padded[1:-1, :-2] + 4 * padded[1:-1, 1:-1] + 2 * padded[1:-1, 2:]
            + padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:]
        ) / 16.0

    if arr.ndim == 3:
        blurred = np.stack([blur2d(arr[c]) for c in range(arr.shape[0])], axis=0)
    else:
        blurred = blur2d(arr)

    return np.clip(arr + amount * (arr - blurred), 0.0, 1.0).astype(np.float32)


def stack_selected_frames(
    siril,
    metrics,
    settings,
    reference_info,
    raw_result_path,
    base_result_path,
    progress=None,
):
    valid_metrics = [m for m in metrics if m["valid"]]
    if not valid_metrics:
        raise RuntimeError("All frames were rejected by the shift sanity check.")

    valid_metrics.sort(key=lambda m: m["quality"], reverse=True)

    if settings.stack_mode.startswith("Weighted"):
        selected = valid_metrics
        keep_count = len(selected)
    else:
        keep_count = expected_keep_count(len(valid_metrics), settings.keep_percent)
        selected = valid_metrics[:keep_count]

    qualities = np.asarray([m["quality"] for m in selected], dtype=np.float32)
    q_low = float(np.percentile(qualities, 10)) if len(qualities) > 1 else float(qualities[0])
    q_high = float(np.percentile(qualities, 95)) if len(qualities) > 1 else float(qualities[0] + 1.0)
    q_span = max(q_high - q_low, 1e-8)

    for metric in selected:
        q_norm = float(np.clip((metric["quality"] - q_low) / q_span, 0.0, 1.0))
        if settings.stack_mode.startswith("Weighted"):
            metric["stack_weight"] = 0.20 + 1.80 * (q_norm ** 2)
        else:
            metric["stack_weight"] = 1.0

    ref_data, _ref_header = read_fits_image(selected[0]["path"])
    ref_output = make_output_image(ref_data, settings.output_mode)
    reference_median, reference_spread = photometric_stats(ref_output)

    stack_sum = None
    weight_sum = None
    output_header = reference_info["header"].copy()
    grid_cache = GridCache()

    if settings.save_debug_previews:
        aligned_tmp = os.path.join(os.path.dirname(os.path.dirname(raw_result_path)), PROCESS_DIR, ALIGNED_TMP_DIR)
        os.makedirs(aligned_tmp, exist_ok=True)
    else:
        aligned_tmp = None

    for out_index, metric in enumerate(selected, start=1):
        data, header = read_fits_image(metric["path"])
        output_image = make_output_image(data, settings.output_mode)
        output_image = match_image_to_reference(output_image, reference_median, reference_spread)

        finite = np.isfinite(output_image)
        fill_value = float(np.median(output_image[finite])) if np.any(finite) else 0.0

        shifted, mask = transform_no_wrap(
            output_image,
            metric["shift_y"],
            metric["shift_x"],
            fill_value,
            metric.get("affine"),
            grid_cache=grid_cache,
        )

        if stack_sum is None:
            stack_sum = np.zeros_like(shifted, dtype=np.float64)
            if shifted.ndim == 3:
                weight_sum = np.zeros(shifted.shape[1:], dtype=np.float64)
            else:
                weight_sum = np.zeros_like(shifted, dtype=np.float64)
            # Output headers are based on the selected reference frame, not the first stacked frame.

        frame_weight = float(metric.get("stack_weight", 1.0))

        if shifted.ndim == 3:
            stack_sum += shifted.astype(np.float64) * mask[None, :, :] * frame_weight
            weight_sum += mask.astype(np.float64) * frame_weight
        else:
            stack_sum += shifted.astype(np.float64) * mask.astype(np.float64) * frame_weight
            weight_sum += mask.astype(np.float64) * frame_weight

        if settings.save_debug_previews and aligned_tmp and out_index <= 5:
            preview_header = header.copy()
            preview_header["PYALNY"] = (metric["shift_y"], "Applied Python alignment shift Y")
            preview_header["PYALNX"] = (metric["shift_x"], "Applied Python alignment shift X")
            preview_header["PYQUAL"] = (metric["quality"], "Python sharpness quality score")
            fits.writeto(
                os.path.join(aligned_tmp, f"aligned_preview_{out_index:03d}.fit"),
                shifted.astype(np.float32),
                preview_header,
                overwrite=True,
            )

        if progress is not None:
            progress.step(text=f"Stacking frame {out_index}/{keep_count}")

    if stack_sum is None or weight_sum is None:
        raise RuntimeError("No frames were stacked.")

    if stack_sum.ndim == 3:
        safe_weights = np.maximum(weight_sum, 1.0)[None, :, :]
    else:
        safe_weights = np.maximum(weight_sum, 1.0)

    stacked = (stack_sum / safe_weights).astype(np.float32)

    output_header["HISTORY"] = "Stacked by DWARF Mini Planetary Preprocessor Optimised"
    output_header["PYPROC"] = (APP_NAME, "Processing script")
    output_header["PYTARGET"] = (settings.target_name[:68], "Selected target/object name")
    output_header["PYREF"] = (reference_info["filename"][:68], "Reference frame header copied from this frame")
    output_header["PYREFIDX"] = (reference_info["index"], "Reference frame index in debayered sequence")
    output_header["PYREFQ"] = (reference_info["quality"], "Reference frame quality score")
    output_header["PYREG"] = (reference_info["method_summary"][:68], "Registration method summary")
    output_header["PYVALID"] = (reference_info["valid_count"], "Valid registered frames")
    output_header["PYFILTER"] = (settings.filter_mode, "Selected DWARF Mini filter")
    output_header["PYOUT"] = (settings.output_mode, "Selected output mode")
    output_header["PYSTACK"] = (keep_count, "Number of frames stacked")
    output_header["PYTOTAL"] = (len(metrics), "Total input frames")
    output_header["PYKEEP"] = (settings.keep_percent, "Best frame percentage kept")
    output_header["PYREGDS"] = (settings.reg_downsample, "Registration downsample factor")
    output_header["PYGPU"] = (bool(settings.use_gpu_fft), "Used CuPy GPU FFT if true")
    output_header["PYCLEAN"] = (bool(settings.clean_outside_limb), "Outside-limb display cleanup if true")
    output_header["DATE"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    raw_header = output_header.copy()
    raw_header["PYFILE"] = ("RAW", "Raw photometrically matched stack")
    fits.writeto(raw_result_path, stacked, raw_header, overwrite=True)

    base_stack = clean_lunar_display_stack(stacked)
    if settings.clean_outside_limb:
        base_stack = clean_outside_limb_display(base_stack)

    # The base file is the practical post-processing input: conservative display
    # stretch, optional outside-limb cleanup, and very mild sharpening.
    base_stack = unsharp_mask(base_stack, amount=0.35)

    base_header = output_header.copy()
    base_header["PYFILE"] = ("BASE", "Practical processing base stack")
    base_header["HISTORY"] = "Base copy: conservative stretch, optional outside-limb cleanup, mild unsharp mask"
    fits.writeto(base_result_path, base_stack.astype(np.float32), base_header, overwrite=True)

    summary = {
        "keep_count": keep_count,
        "quality_best": float(np.max(qualities)),
        "quality_worst": float(np.min(qualities)),
        "reference_median": reference_median,
        "reference_spread": reference_spread,
    }

    return raw_result_path, base_result_path, keep_count, summary


# -------------------------
# Main
# -------------------------

def main():
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    siril = s.SirilInterface()

    try:
        siril.connect()
        log_stage(siril, f"Connected to Siril for {APP_NAME}", LogColor.GREEN)

        run_cmd(siril, "requires", MIN_SIRIL_VERSION)
        run_cmd(siril, "close")
        run_cmd(siril, "setcompress", "0")
        run_cmd(siril, "set32bits")
        run_cmd(siril, "setext", "fit")

        settings = choose_processing_settings(siril)

        lights_dir = os.path.join(settings.project_dir, LIGHTS_DIR)
        light_files = list_fits_files(lights_dir)

        if not light_files:
            raise RuntimeError("No FITS files were found in the lights folder.")

        process_dir, result_dir = reset_output_folders(settings.project_dir)
        raw_output_basename, base_output_basename = output_basenames_for_target(settings.target_name)
        raw_result_path = os.path.join(result_dir, raw_output_basename + ".fit")
        base_result_path = os.path.join(result_dir, base_output_basename + ".fit")

        log_stage(siril, "DWARF Mini planetary preprocessing started", LogColor.GREEN)
        log_stage(siril, f"Target: {settings.target_name}", LogColor.BLUE)
        log_stage(siril, f"Frames: {len(light_files)} | Output: {settings.output_mode} | Stack: {settings.stack_mode}", LogColor.BLUE)
        log_stage(
            siril,
            f"Reference: {settings.reference_mode} | Downsample: {settings.reg_downsample}x | "
            f"Workers: {settings.cpu_workers} | GPU FFT: {settings.use_gpu_fft} | Outside cleanup: {settings.clean_outside_limb}",
            LogColor.BLUE,
        )

        if settings.filter_mode == "Dual-Band":
            log_stage(siril, "Dual-Band selected. For Moon/Sun, Astro filter is usually preferred.", LogColor.SALMON)

        total_steps = 80 + len(light_files) + len(light_files) + 20
        progress = ProgressReporter("Preparing lunar stack...", total_steps)

        try:
            progress.set_label("Debayering OSC/CFA frames with Siril...")
            debayered_files = debayer_with_siril(siril, settings.project_dir, process_dir)
            progress.set_value(80, f"Siril debayered {len(debayered_files)} frames")

            log_stage(siril, f"Siril OSC debayered {len(debayered_files)} frames", LogColor.GREEN)

            metrics, reference_info = score_and_register_frames(siril, debayered_files, settings, progress=progress)

            raw_result_path, base_result_path, keep_count, stack_summary = stack_selected_frames(
                siril,
                metrics,
                settings,
                reference_info,
                raw_result_path,
                base_result_path,
                progress=progress,
            )

            progress.set_label("Saving and loading result...")
            progress.close()
        except Exception:
            try:
                progress.close()
            except Exception:
                pass
            raise

        run_cmd(siril, "load", qpath(base_result_path))

        log_stage(siril, "DWARF Mini planetary preprocessing complete.", LogColor.GREEN)
        log_stage(siril, f"Frames stacked: {keep_count} of {len(debayered_files)}", LogColor.GREEN)
        log_stage(siril, f"Quality selected: best={stack_summary['quality_best']:.6f}, worst={stack_summary['quality_worst']:.6f}", LogColor.BLUE)
        log_stage(siril, f"Saved raw result: {raw_result_path}", LogColor.GREEN)
        log_stage(siril, f"Saved base result: {base_result_path}", LogColor.GREEN)
        log_stage(siril, f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", LogColor.GREEN)

        newline = chr(10)
        complete_message = newline.join([
            "Preprocessing complete.",
            "",
            f"Target: {settings.target_name}",
            "",
            "Saved raw stack:",
            raw_result_path,
            "",
            "Saved base stack:",
            base_result_path,
            "",
            f"Frames stacked: {keep_count} of {len(debayered_files)}",
            f"Registration downsample: {settings.reg_downsample}x",
            f"GPU FFT used: {settings.use_gpu_fft}",
            f"Outside-limb display cleanup: {settings.clean_outside_limb}",
            "",
            "FITS headers are copied from the selected reference frame, with processing cards added.",
        ])

        QMessageBox.information(None, "Complete", complete_message)

    except Exception as exc:
        try:
            siril.log(f"Processing failed: {exc}", LogColor.RED)
        except Exception:
            pass

        QMessageBox.critical(None, "Processing failed", str(exc))
        raise

    finally:
        try:
            siril.disconnect()
        except Exception:
            pass

        if created_app:
            app.quit()


if __name__ == "__main__":
    main()
