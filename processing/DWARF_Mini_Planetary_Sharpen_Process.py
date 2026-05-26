"""
DWARF Mini Planetary Sharpen / Post-Processing Script
====================================================

Post-processing only.
This script takes an already-created FITS stack and applies
planetary/lunar sharpening enhancement.

It does NOT:
- debayer
- calibrate
- register
- align
- stack
- preprocess sequences

Recommended input:
    result/DWARF_Mini_planetary_stack_base.fit

Output:
    Default naming:
        *_base.fit -> *_final.fit
        *_raw.fit  -> *_final_from_raw.fit
        other FITS -> *_processed.fit

    The output path can also be edited in the dialog.

UI:
    Automatic mode uses the selected preset.
    Turn Automatic off to edit the main tuning controls.
    Advanced is only available when Automatic is off.
"""

import os
import sys
from datetime import datetime, timezone

try:
    import sirilpy as s
    HAVE_SIRIL = True
    s.ensure_installed("PyQt6", "numpy", "astropy")
except Exception:
    HAVE_SIRIL = False
    s = None

import numpy as np
from astropy.io import fits
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QSpinBox,
    QVBoxLayout,
)

APP_NAME = "DWARF Mini Planetary Sharpen Processor"
APP_VERSION = "0.5"


PRESETS = {
    "Soft": {
        "denoise_strength": 0.14,
        "rl_iterations": 10,
        "psf_radius": 1.2,
        "deconv_blend": 0.75,
        "detail_strength": 0.55,
        "limb_protection": 0.92,
        "noise_gate": 0.80,
        "final_contrast": 0.08,
    },
    "Balanced": {
        "denoise_strength": 0.12,
        "rl_iterations": 12,
        "psf_radius": 1.2,
        "deconv_blend": 0.80,
        "detail_strength": 0.65,
        "limb_protection": 0.90,
        "noise_gate": 0.70,
        "final_contrast": 0.10,
    },
    "Strong": {
        "denoise_strength": 0.12,
        "rl_iterations": 12,
        "psf_radius": 1.2,
        "deconv_blend": 0.80,
        "detail_strength": 0.70,
        "limb_protection": 0.90,
        "noise_gate": 0.70,
        "final_contrast": 0.10,
    },
}


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.resize(780, 520)

        self._updating_ui = False

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)

        self.input_edit = QLineEdit()
        self.input_button = QPushButton("Browse...")
        self.input_button.clicked.connect(self.choose_input)

        self.output_edit = QLineEdit()
        self.output_button = QPushButton("Browse...")
        self.output_button.clicked.connect(self.choose_output)

        grid.addWidget(QLabel("Input FITS:"), 0, 0)
        grid.addWidget(self.input_edit, 0, 1)
        grid.addWidget(self.input_button, 0, 2)

        grid.addWidget(QLabel("Output FITS:"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(self.output_button, 1, 2)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()))
        self.preset_combo.setCurrentText("Balanced")
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)

        self.auto_checkbox = QCheckBox("Automatic")
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.toggled.connect(self.update_ui_state)

        self.advanced_checkbox = QCheckBox("Advanced")
        self.advanced_checkbox.setChecked(False)
        self.advanced_checkbox.toggled.connect(self.on_advanced_toggled)

        row = 2
        grid.addWidget(QLabel("Preset:"), row, 0)
        grid.addWidget(self.preset_combo, row, 1)
        row += 1

        grid.addWidget(self.auto_checkbox, row, 0, 1, 2)
        grid.addWidget(self.advanced_checkbox, row, 2)
        row += 1

        self.deconv_blend_spin = self._make_double_spin(0.0, 1.0, 0.05, 0.80, 2)
        self.detail_spin = self._make_double_spin(0.0, 2.0, 0.05, 0.65, 2)
        self.noise_gate_spin = self._make_double_spin(0.0, 5.0, 0.1, 0.70, 1)

        grid.addWidget(QLabel("Deconvolution blend:"), row, 0)
        grid.addWidget(self.deconv_blend_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Detail strength:"), row, 0)
        grid.addWidget(self.detail_spin, row, 1)
        row += 1

        grid.addWidget(QLabel("Noise gate:"), row, 0)
        grid.addWidget(self.noise_gate_spin, row, 1)
        row += 1

        self.advanced_group = QGroupBox("Advanced controls")
        adv_layout = QGridLayout(self.advanced_group)

        self.denoise_spin = self._make_double_spin(0.0, 1.0, 0.02, 0.12, 2)
        self.iter_spin = QSpinBox()
        self.iter_spin.setRange(0, 50)
        self.iter_spin.setValue(12)

        self.psf_spin = self._make_double_spin(0.5, 6.0, 0.1, 1.2, 1)
        self.limb_spin = self._make_double_spin(0.0, 1.0, 0.05, 0.90, 2)
        self.final_contrast_spin = self._make_double_spin(0.0, 1.0, 0.05, 0.10, 2)

        adv_row = 0
        adv_layout.addWidget(QLabel("Denoise strength:"), adv_row, 0)
        adv_layout.addWidget(self.denoise_spin, adv_row, 1)
        adv_row += 1

        adv_layout.addWidget(QLabel("Richardson-Lucy iterations:"), adv_row, 0)
        adv_layout.addWidget(self.iter_spin, adv_row, 1)
        adv_row += 1

        adv_layout.addWidget(QLabel("PSF radius:"), adv_row, 0)
        adv_layout.addWidget(self.psf_spin, adv_row, 1)
        adv_row += 1

        adv_layout.addWidget(QLabel("Limb protection:"), adv_row, 0)
        adv_layout.addWidget(self.limb_spin, adv_row, 1)
        adv_row += 1

        adv_layout.addWidget(QLabel("Final local contrast:"), adv_row, 0)
        adv_layout.addWidget(self.final_contrast_spin, adv_row, 1)

        layout.addWidget(self.advanced_group)

        self.load_result_checkbox = QCheckBox("Load processed FITS in Siril when complete")
        self.load_result_checkbox.setChecked(True)
        self.load_result_checkbox.setEnabled(HAVE_SIRIL)
        layout.addWidget(self.load_result_checkbox)

        note = QLabel(
            "Automatic uses the selected preset and greys out manual controls. "
            "Turn Automatic off to manually tune Deconvolution blend, Detail strength, and Noise gate. "
            "Advanced is only available when Automatic is off."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.apply_preset_to_controls(self.preset_combo.currentText())
        self.update_ui_state()

    def _make_double_spin(self, minimum, maximum, step, value, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setDecimals(decimals)
        return spin

    def choose_input(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select preprocessed FITS stack",
            os.path.expanduser("~"),
            "FITS files (*.fit *.fits);;All files (*.*)",
        )
        if filename:
            self.input_edit.setText(filename)
            self.output_edit.setText(output_path_for(filename))

    def choose_output(self):
        input_path = self.input_edit.text().strip()
        default_path = self.output_edit.text().strip() or output_path_for(input_path) if input_path else os.path.expanduser("~")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save processed FITS as",
            default_path,
            "FITS files (*.fit *.fits);;All files (*.*)",
        )
        if filename:
            if not filename.lower().endswith((".fit", ".fits")):
                filename += ".fit"
            self.output_edit.setText(filename)

    def on_preset_changed(self, preset_name):
        if not preset_name or preset_name not in PRESETS:
            return
        self.apply_preset_to_controls(preset_name)

    def apply_preset_to_controls(self, preset_name):
        values = PRESETS[preset_name]
        self._updating_ui = True
        try:
            self.denoise_spin.setValue(values["denoise_strength"])
            self.iter_spin.setValue(values["rl_iterations"])
            self.psf_spin.setValue(values["psf_radius"])
            self.deconv_blend_spin.setValue(values["deconv_blend"])
            self.detail_spin.setValue(values["detail_strength"])
            self.limb_spin.setValue(values["limb_protection"])
            self.noise_gate_spin.setValue(values["noise_gate"])
            self.final_contrast_spin.setValue(values["final_contrast"])
        finally:
            self._updating_ui = False

    def on_advanced_toggled(self, checked):
        # Advanced is only reachable when Automatic is already off.
        if self.auto_checkbox.isChecked() and checked:
            self.advanced_checkbox.setChecked(False)
            return
        self.update_ui_state()

    def update_ui_state(self):
        automatic = self.auto_checkbox.isChecked()

        if automatic:
            # Auto mode owns all values. Keep the preset selectable, but prevent
            # manual tuning and prevent Advanced from being opened.
            self.advanced_checkbox.setChecked(False)
            self.advanced_checkbox.setEnabled(False)
            self.advanced_group.setVisible(False)
            self.deconv_blend_spin.setEnabled(False)
            self.detail_spin.setEnabled(False)
            self.noise_gate_spin.setEnabled(False)
            return

        # Manual mode: the three main controls are editable. Advanced becomes
        # available but remains hidden until the user ticks it.
        self.advanced_checkbox.setEnabled(True)
        advanced = self.advanced_checkbox.isChecked()
        self.advanced_group.setVisible(advanced)

        self.deconv_blend_spin.setEnabled(True)
        self.detail_spin.setEnabled(True)
        self.noise_gate_spin.setEnabled(True)

    def accept(self):
        input_path = self.input_edit.text().strip()
        if not input_path or not os.path.isfile(input_path):
            QMessageBox.critical(self, "Missing FITS", "Please select a valid FITS file.")
            return

        output_path = self.output_edit.text().strip()
        if not output_path:
            output_path = output_path_for(input_path)
            self.output_edit.setText(output_path)

        if not output_path.lower().endswith((".fit", ".fits")):
            output_path += ".fit"
            self.output_edit.setText(output_path)

        output_dir = os.path.dirname(os.path.abspath(output_path))
        if not os.path.isdir(output_dir):
            QMessageBox.critical(self, "Invalid output folder", "Please choose an existing output folder.")
            return

        super().accept()

    def values(self):
        preset_name = self.preset_combo.currentText()
        base = PRESETS[preset_name].copy()

        advanced = self.advanced_checkbox.isChecked()
        automatic = self.auto_checkbox.isChecked()

        if automatic and not advanced:
            final_settings = base
        elif advanced:
            final_settings = {
                "denoise_strength": float(self.denoise_spin.value()),
                "rl_iterations": int(self.iter_spin.value()),
                "psf_radius": float(self.psf_spin.value()),
                "deconv_blend": float(self.deconv_blend_spin.value()),
                "detail_strength": float(self.detail_spin.value()),
                "limb_protection": float(self.limb_spin.value()),
                "noise_gate": float(self.noise_gate_spin.value()),
                "final_contrast": float(self.final_contrast_spin.value()),
            }
        else:
            final_settings = base
            final_settings["deconv_blend"] = float(self.deconv_blend_spin.value())
            final_settings["detail_strength"] = float(self.detail_spin.value())
            final_settings["noise_gate"] = float(self.noise_gate_spin.value())

        final_settings.update(
            {
                "input_path": self.input_edit.text().strip(),
                "output_path": self.output_edit.text().strip() or output_path_for(self.input_edit.text().strip()),
                "preset_name": preset_name,
                "automatic": bool(automatic and not advanced),
                "advanced": bool(advanced),
                "load_result": bool(self.load_result_checkbox.isChecked() and HAVE_SIRIL),
            }
        )
        return final_settings


class Progress:
    def __init__(self, title, maximum=100):
        self.dialog = QProgressDialog(title, "Cancel", 0, maximum)
        self.dialog.setWindowTitle(APP_NAME)
        self.dialog.setMinimumDuration(0)
        self.dialog.show()
        QApplication.processEvents()

    def set(self, value, text):
        self.dialog.setValue(int(value))
        self.dialog.setLabelText(text)
        QApplication.processEvents()
        if self.dialog.wasCanceled():
            raise RuntimeError("Processing cancelled by user.")

    def close(self):
        self.dialog.setValue(self.dialog.maximum())
        self.dialog.close()
        QApplication.processEvents()


def read_fits(path):
    with fits.open(path, memmap=False) as hdul:
        hdu = None
        for candidate in hdul:
            if getattr(candidate, "data", None) is not None:
                if isinstance(candidate.data, np.ndarray) and candidate.data.ndim >= 2:
                    hdu = candidate
                    break
        if hdu is None:
            raise RuntimeError("No image data found in FITS file.")
        data = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header.copy()
    data = np.nan_to_num(data, copy=False)
    return data, header


def write_fits(path, data, header, settings):
    out_header = header.copy()
    out_header["HISTORY"] = "Processed by DWARF Mini Planetary Sharpen Processor"
    out_header["PYPOST"] = (APP_NAME, "Post-processing script")
    out_header["PYPOSTV"] = (APP_VERSION, "Post-processing script version")
    out_header["PYPRESET"] = (settings["preset_name"], "Sharpening preset")
    out_header["PYOUTNAM"] = (os.path.basename(path)[:68], "Output filename")
    out_header["PYAUTO"] = (int(bool(settings["automatic"])), "Automatic mode")
    out_header["PYADV"] = (int(bool(settings["advanced"])), "Advanced mode")
    out_header["PYDENOIS"] = (settings["denoise_strength"], "Denoise strength")
    out_header["PYRLITER"] = (settings["rl_iterations"], "Richardson-Lucy iterations")
    out_header["PYPSF"] = (settings["psf_radius"], "PSF radius")
    out_header["PYDBLEND"] = (settings["deconv_blend"], "Deconvolution blend")
    out_header["PYDETAIL"] = (settings["detail_strength"], "Detail strength")
    out_header["PYLIMB"] = (settings["limb_protection"], "Limb protection")
    out_header["PYNGATE"] = (settings["noise_gate"], "Noise gate")
    out_header["PYCONT"] = (settings["final_contrast"], "Final local contrast")
    out_header["DATE"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fits.writeto(path, data.astype(np.float32), out_header, overwrite=True)


def output_path_for(input_path):
    root, ext = os.path.splitext(input_path)
    if ext.lower() not in (".fit", ".fits"):
        ext = ".fit"

    lower_root = root.lower()
    if lower_root.endswith("_base"):
        return root[:-5] + "_final" + ext

    if lower_root.endswith("_raw"):
        return root[:-4] + "_final_from_raw" + ext

    if lower_root.endswith("_processed"):
        return root + ext

    return root + "_processed" + ext


def channel_first_rgb(data):
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (3, 4):
        return arr[:3]
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return np.moveaxis(arr[..., :3], -1, 0)
    return None


def luminance(data):
    arr = np.asarray(data, dtype=np.float32)
    rgb = channel_first_rgb(arr)
    if rgb is not None:
        return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
    if arr.ndim == 2:
        return arr
    return arr[0].astype(np.float32)


def replace_luminance(original, new_lum):
    arr = np.asarray(original, dtype=np.float32)
    rgb = channel_first_rgb(arr)
    if rgb is None:
        return new_lum.astype(np.float32)

    old_lum = luminance(rgb)
    scale = new_lum / np.maximum(old_lum, 1e-6)
    out = rgb * scale[None, :, :]
    return np.nan_to_num(out, copy=False).astype(np.float32)


def robust_normalise(img):
    img = np.asarray(img, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.zeros_like(img, dtype=np.float32), 0.0, 1.0

    lo, hi = np.percentile(img[finite], [0.2, 99.85])
    if hi <= lo:
        lo = float(np.min(img[finite]))
        hi = float(np.max(img[finite]))

    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32), lo, hi

    out = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32), float(lo), float(hi)


def restore_scale(img01, lo, hi):
    return (np.asarray(img01, dtype=np.float32) * (hi - lo) + lo).astype(np.float32)


def gaussian_kernel1d(radius):
    sigma = max(float(radius), 0.1)
    half = max(1, int(round(sigma * 3.0)))
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


def convolve_axis_reflect(img, kernel, axis):
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(img, ((pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(img, dtype=np.float32)
        for i, weight in enumerate(kernel):
            out += weight * padded[i : i + img.shape[0], :]
        return out

    padded = np.pad(img, ((0, 0), (pad, pad)), mode="reflect")
    out = np.zeros_like(img, dtype=np.float32)
    for i, weight in enumerate(kernel):
        out += weight * padded[:, i : i + img.shape[1]]
    return out


def gaussian_blur(img, radius):
    if radius <= 0:
        return np.asarray(img, dtype=np.float32)
    kernel = gaussian_kernel1d(radius)
    tmp = convolve_axis_reflect(np.asarray(img, dtype=np.float32), kernel, axis=0)
    return convolve_axis_reflect(tmp, kernel, axis=1)


def gradient_magnitude(img):
    img = np.asarray(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx = np.zeros_like(img, dtype=np.float32)
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def soft_surface_mask(img01, limb_protection):
    img = np.asarray(img01, dtype=np.float32)
    finite = np.isfinite(img)
    if not np.any(finite):
        return np.ones_like(img, dtype=np.float32)

    p10, p995 = np.percentile(img[finite], [10.0, 99.5])
    threshold = p10 + 0.025 * max(float(p995 - p10), 1e-6)
    body = (img > threshold).astype(np.float32)
    body = gaussian_blur(body, 2.0)
    body = np.clip(body, 0.0, 1.0)

    if limb_protection <= 0:
        return body.astype(np.float32)

    grad = gradient_magnitude(img)
    g99 = np.percentile(grad[finite], 99.5)
    if g99 <= 0:
        return body.astype(np.float32)

    edge = np.clip(grad / g99, 0.0, 1.0)
    edge = gaussian_blur(edge, 2.0)
    protection = 1.0 - limb_protection * edge
    protection = np.clip(protection, 0.0, 1.0)

    return (body * protection).astype(np.float32)


def edge_aware_denoise(img, strength):
    if strength <= 0:
        return np.asarray(img, dtype=np.float32)

    img = np.asarray(img, dtype=np.float32)
    blur_large = gaussian_blur(img, 1.3)

    grad = gradient_magnitude(img)
    finite = np.isfinite(grad)
    g95 = np.percentile(grad[finite], 95.0) if np.any(finite) else 0.0

    if g95 <= 0:
        smooth_weight = np.ones_like(img, dtype=np.float32)
    else:
        smooth_weight = 1.0 - np.clip(grad / g95, 0.0, 1.0)
        smooth_weight = smooth_weight ** 1.5

    denoised = img * (1.0 - strength * smooth_weight) + blur_large * (strength * smooth_weight)
    return denoised.astype(np.float32)


def richardson_lucy_gaussian(observed, radius, iterations, progress=None, progress_start=25, progress_end=70):
    if iterations <= 0:
        return np.asarray(observed, dtype=np.float32)

    observed = np.clip(np.asarray(observed, dtype=np.float32), 0.0, 1.0)
    estimate = np.maximum(observed.copy(), 1e-6)

    for i in range(iterations):
        blurred = gaussian_blur(estimate, radius)
        ratio = observed / np.maximum(blurred, 1e-6)
        correction = gaussian_blur(ratio, radius)
        estimate *= correction
        estimate = np.clip(estimate, 0.0, 1.5)

        if progress is not None:
            span = progress_end - progress_start
            value = progress_start + span * (i + 1) / max(iterations, 1)
            progress.set(value, f"Deconvolution iteration {i + 1}/{iterations}")

    return np.clip(estimate, 0.0, 1.0).astype(np.float32)


def noise_gate_detail(detail, img, gate_strength):
    if gate_strength <= 0:
        return detail.astype(np.float32)

    smooth = gaussian_blur(img, 1.2)
    noise = img - smooth
    sigma = 1.4826 * np.median(np.abs(noise - np.median(noise)))
    sigma = max(float(sigma), 1e-6)

    threshold = gate_strength * sigma
    magnitude = np.abs(detail)
    keep = np.clip((magnitude - threshold) / max(threshold, 1e-6), 0.0, 1.0)
    keep = keep ** 0.7
    return (detail * keep).astype(np.float32)


def local_contrast_layer(img, radius):
    blur = gaussian_blur(img, radius)
    return (img - blur).astype(np.float32)


def final_local_contrast(img, amount):
    if amount <= 0:
        return np.asarray(img, dtype=np.float32)
    base = gaussian_blur(img, 3.0)
    contrast = img - base
    return np.clip(img + amount * contrast, 0.0, 1.0).astype(np.float32)


def process_luminance(lum, settings, progress):
    progress.set(10, "Normalising luminance")
    img01, lo, hi = robust_normalise(lum)

    progress.set(18, "Light edge-aware denoise")
    denoised = edge_aware_denoise(img01, settings["denoise_strength"])

    progress.set(25, "Deconvolving")
    deconv = richardson_lucy_gaussian(
        denoised,
        radius=settings["psf_radius"],
        iterations=settings["rl_iterations"],
        progress=progress,
        progress_start=25,
        progress_end=70,
    )

    progress.set(72, "Blending deconvolved structure")
    deconv_blend = settings["deconv_blend"]
    structural_base = denoised * (1.0 - deconv_blend) + deconv * deconv_blend

    progress.set(76, "Extracting controlled multi-scale detail")
    fine_detail = local_contrast_layer(deconv, 0.8)
    medium_detail = local_contrast_layer(deconv, 1.8)
    detail = 0.70 * fine_detail + 0.30 * medium_detail
    detail = noise_gate_detail(detail, denoised, settings["noise_gate"])

    progress.set(82, "Building limb-safe surface mask")
    mask = soft_surface_mask(denoised, settings["limb_protection"])

    progress.set(88, "Applying detail enhancement")
    enhanced = structural_base + settings["detail_strength"] * detail * mask
    enhanced = np.clip(enhanced, 0.0, 1.0).astype(np.float32)

    progress.set(91, "Final local contrast")
    enhanced = final_local_contrast(enhanced, settings["final_contrast"])

    progress.set(94, "Restoring FITS scale")
    return restore_scale(enhanced, lo, hi)


def main():
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication(sys.argv)
        created_app = True

    siril = None
    if HAVE_SIRIL:
        try:
            siril = s.SirilInterface()
            siril.connect()
            siril.log(f"Connected to Siril for {APP_NAME}")
        except Exception:
            siril = None

    try:
        dialog = SettingsDialog()
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        settings = dialog.values()
        progress = Progress("Sharpening planetary stack...", 100)

        progress.set(5, "Reading FITS")
        data, header = read_fits(settings["input_path"])
        lum = luminance(data)

        processed_lum = process_luminance(lum, settings, progress)

        progress.set(96, "Recombining channels")
        output_data = replace_luminance(data, processed_lum)

        output_path = settings["output_path"]
        progress.set(98, "Writing FITS")
        write_fits(output_path, output_data, header, settings)

        if settings["load_result"] and siril is not None:
            progress.set(99, "Loading result in Siril")
            siril.cmd("load", '"' + output_path.replace("\\", "/") + '"')

        progress.close()

        summary = (
            "Post-processing complete.\n\n"
            f"Saved:\n{output_path}\n\n"
            f"Preset: {settings['preset_name']}\n"
            f"Automatic: {'Yes' if settings['automatic'] else 'No'}\n"
            f"Advanced: {'Yes' if settings['advanced'] else 'No'}"
        )
        QMessageBox.information(None, "Complete", summary)

    except Exception as exc:
        try:
            if siril is not None:
                siril.log(f"Processing failed: {exc}")
        except Exception:
            pass
        QMessageBox.critical(None, "Processing failed", str(exc))
        raise

    finally:
        try:
            if siril is not None:
                siril.disconnect()
        except Exception:
            pass
        if created_app:
            app.quit()


if __name__ == "__main__":
    main()
