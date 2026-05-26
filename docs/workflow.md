# Workflow

## Overview

The project is split into two stages:

```text
DWARF Mini FITS sequence
→ preprocessing script
→ base FITS stack
→ processing/sharpening script
→ final FITS
```

This separation is deliberate.

The preprocessing stage should create a reliable data product. The processing stage can then make visual choices such as denoise, deconvolution, local contrast, and sharpening.

## Stage 1: Preprocessing

Use:

```text
preprocessing/DWARF_Mini_Planetary_Preprocess.py
```

Input:

```text
CaptureFolder/
  lights/
    frame_0001.fit
    frame_0002.fit
    ...
```

Process:

1. Siril debayers the OSC/CFA FITS frames.
2. The script chooses a reference frame from an evenly spaced sample.
3. Local lunar/solar/planetary features are detected.
4. Each frame is aligned using feature-based affine registration.
5. Valid frames are stacked with quality weights.
6. FITS headers are copied from the selected reference frame.
7. Processing metadata is added to FITS headers.

Outputs:

```text
result/DWARF_Mini_<Target>_stack_raw.fit
result/DWARF_Mini_<Target>_stack_base.fit
```

Use `_raw.fit` for diagnostics or archival comparison.

Use `_base.fit` as the usual input to the processing script.

## Stage 2: Processing / sharpening

Use:

```text
processing/DWARF_Mini_Planetary_Sharpen_Process.py
```

Input:

```text
result/DWARF_Mini_<Target>_stack_base.fit
```

Process:

1. Load FITS and preserve metadata.
2. Apply conservative edge-aware denoise.
3. Apply Richardson-Lucy style deconvolution.
4. Blend deconvolved structure.
5. Add controlled multi-scale detail.
6. Use limb protection to reduce ringing.
7. Save final FITS.

Output:

```text
result/DWARF_Mini_<Target>_stack_final.fit
```

## Recommended starting settings

### Preprocessing

```text
Output mode: Monochrome luminance
Stack mode: Weighted all valid frames
Reference mode: Median-quality sampled frame
Registration downsample: 2
Outside-limb display cleanup: enabled
```

### Processing

Start with:

```text
Preset: Balanced
Automatic: enabled
```

Try `Strong` for cleaner, sharper datasets.

Try `Soft` for noisy datasets or where the bright limb is ringing.

## Avoiding file clutter

The intended workflow produces only a small number of key files:

```text
*_raw.fit
*_base.fit
*_final.fit
```

Temporary process files are created in the `process/` folder and can be deleted after a successful run if not needed.

## Notes on lunar halos

Some haloing around the bright lunar limb can come from stacking interpolation, deconvolution, or over-sharpening. If the halo becomes obvious:

- use the Balanced or Soft preset
- reduce Detail Strength
- reduce Deconvolution Blend
- increase Limb Protection
- avoid additional sharpening after the script
