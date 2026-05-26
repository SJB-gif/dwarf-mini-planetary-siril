# DWARF Mini Planetary Siril Scripts

Siril Python scripts for preprocessing and sharpening DWARF Mini lunar, solar, and planetary FITS sequences.

These scripts were developed for DWARF Mini OSC FITS captures of bright solar-system targets where normal deep-sky star registration is not appropriate, especially Moon sequences.

## What this repository contains

```text
preprocessing/
  DWARF_Mini_Planetary_Preprocess.py

processing/
  DWARF_Mini_Planetary_Sharpen_Process.py

docs/
  workflow.md
  siril-installation.md

examples/
  README.md
```

## Scripts

### 1. Preprocessing

`preprocessing/DWARF_Mini_Planetary_Preprocess.py`

Takes a folder of DWARF Mini FITS light frames and creates a clean stack.

Main features:

- uses Siril OSC/CFA debayering
- registers lunar/solar/planetary frames without relying on stars
- uses local feature tracking and affine alignment
- uses weighted stacking across valid frames
- copies the selected reference frame FITS header into the outputs
- embeds processing metadata in the FITS headers
- supports target-based output names such as `DWARF_Mini_Moon_stack_base.fit`

Expected input folder structure:

```text
YourCaptureFolder/
  lights/
    *.fit
    *.fits
    *.fit.fz
    *.fits.fz
```

Typical outputs:

```text
YourCaptureFolder/
  result/
    DWARF_Mini_Moon_stack_raw.fit
    DWARF_Mini_Moon_stack_base.fit
```

### 2. Processing / sharpening

`processing/DWARF_Mini_Planetary_Sharpen_Process.py`

Takes a preprocessed stack, normally the `_base.fit` file, and creates a final sharpened FITS.

Main features:

- preset-based sharpening UI: Soft, Balanced, Strong
- Automatic mode for simple use
- Basic manual controls for:
  - Deconvolution blend
  - Detail strength
  - Noise gate
- Advanced mode for full tuning
- Richardson-Lucy style deconvolution
- edge-aware denoise
- limb protection to reduce lunar limb ringing
- output filename selector
- converts `_base.fit` to `_final.fit` by default

Typical input:

```text
DWARF_Mini_Moon_stack_base.fit
```

Typical output:

```text
DWARF_Mini_Moon_stack_final.fit
```

## Recommended first workflow

1. Put your DWARF Mini FITS files in a folder called `lights`.
2. Run the preprocessing script from Siril's **Preprocessing** scripts menu.
3. Use the `_base.fit` output for sharpening.
4. Run the processing script from Siril's **Processing** scripts menu.
5. Start with the **Balanced** preset and **Automatic** enabled.
6. Adjust Detail Strength only if needed.

Recommended preprocessing settings:

```text
Target: Moon / Sun / Jupiter / Saturn / etc.
Filter: Astro filter (UV/IR), unless you actually used Dual-Band
Output: Monochrome luminance
Stack mode: Weighted all valid frames
Reference: Median-quality sampled frame
Registration downsample: 2
Outside-limb display cleanup: enabled
```

Recommended sharpening settings:

```text
Preset: Balanced
Automatic: enabled
```

For a stronger result:

```text
Preset: Strong
Automatic: enabled
```

## Installing in Siril

Create a user-owned scripts folder, for example:

```text
Documents/
  SirilScripts/
    preprocessing/
      DWARF_Mini_Planetary_Preprocess.py
    processing/
      DWARF_Mini_Planetary_Sharpen_Process.py
```

Then add the two folders in Siril preferences:

```text
Documents/SirilScripts/preprocessing
Documents/SirilScripts/processing
```

After refreshing/restarting Siril, the scripts should appear under the matching Preprocessing and Processing categories.

See `docs/siril-installation.md` for more detail.

## Requirements

- Siril 1.4 or newer
- Python support in Siril
- Python packages:
  - PyQt6
  - numpy
  - astropy

The scripts call Siril's `sirilpy.ensure_installed(...)` to install required Python packages where Siril allows it.

## Known limitations

- Developed and tested primarily with DWARF Mini Moon sequences.
- Solar and planetary workflows should work in principle, but may need different sharpening presets.
- These scripts are not intended for deep-sky processing.
- They do not perform plate solving or star registration.
- Very aggressive sharpening can create halos around the lunar limb.
- Large FITS sequences can require significant disk space and processing time.


## License

MIT License. See `LICENSE`.

## Disclaimer

These scripts are experimental community tools. Always keep your original FITS captures unchanged.
