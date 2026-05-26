# Siril installation

## Recommended folder layout

Do not place custom scripts in `Program Files` unless you specifically want to manage global installed scripts with administrator permissions.

A user-owned folder is easier to manage:

```text
C:\Users\<you>\Documents\SirilScripts\
```

Create category folders inside it:

```text
C:\Users\<you>\Documents\SirilScripts\preprocessing\
C:\Users\<you>\Documents\SirilScripts\processing\
```

Copy the scripts into those folders:

```text
preprocessing\
  DWARF_Mini_Planetary_Preprocess.py

processing\
  DWARF_Mini_Planetary_Sharpen_Process.py
```

## Add folders in Siril

In Siril:

1. Open Preferences.
2. Go to Scripts.
3. Add the preprocessing folder as a script storage directory.
4. Add the processing folder as a script storage directory.
5. Refresh scripts or restart Siril.

Add these folders directly:

```text
C:\Users\<you>\Documents\SirilScripts\preprocessing
C:\Users\<you>\Documents\SirilScripts\processing
```

This helps Siril show the scripts under the correct Preprocessing and Processing menu categories.

## Common issue: scripts appear under a generic 'scripts' folder

If Siril shows a menu folder called `scripts`, it is usually because the folder added to Siril preferences is literally named `scripts`.

Fix:

1. Create explicit `preprocessing` and `processing` folders.
2. Move the scripts into those folders.
3. Add those folders directly in Siril Preferences.
4. Refresh/restart Siril.

## Recommended capture folder layout

Each capture should have a `lights` folder:

```text
Moon_2026_05_25\
  lights\
    frame_0001.fit
    frame_0002.fit
```

The preprocessing script will create:

```text
Moon_2026_05_25\
  process\
  result\
```

## Windows Developer Mode note

Siril may copy files instead of creating symbolic links if Windows Developer Mode is not enabled. The scripts still work, but linking/copying may be slower and more verbose.
