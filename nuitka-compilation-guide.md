# Compiling Python with Nuitka: A Step-by-Step Guide

This guide outlines the process to compile a Python `main.py` file using Nuitka, creating a standalone executable with an embedded icon.

## Prerequisites

- Visual Studio Build Tools 2022 installed
- Python environment with Nuitka installed
- Your Python project with `main.py` as the entry point
- Icon file (pickaxe.ico) and splash screen image (pickaxe.png) in the project directory

## Steps

1. **Open Command Prompt as Administrator**
   
   Right-click on Command Prompt and select "Run as administrator".

2. **Set Up Visual Studio Environment**

   Run the following command to set up the Visual Studio environment for x64 architecture:
   ```
   "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
   ```

3. **Navigate to Your Project Directory**

   Change to your project directory:
   ```
   cd "path\to\your\project\directory"
   ```

4. **Activate Your Python Virtual Environment**

   If you're using a virtual environment, activate it:
   ```
   .venv\Scripts\activate
   ```

5. **Run Nuitka Compilation Command**

   Execute the following Nuitka command to compile your `main.py`:
   ```
   nuitka --standalone --onefile --follow-imports --assume-yes-for-downloads --msvc=latest --jobs=6 --remove-output --full-compat --plugin-enable=pyside6 --windows-console-mode=disable --windows-icon-from-ico=pickaxe.ico --windows-icon-from-ico=pickaxe.ico --include-data-file=pickaxe.ico=pickaxe.ico --include-data-file=pickaxe.png=pickaxe.png --onefile-windows-splash-screen-image=pickaxe.png --output-filename=PickAxe.exe starter.py
   ```

   This command includes the following options:
   - `--standalone`: Creates a standalone executable with all dependencies
   - `--onefile`: Packages everything into a single executable file
   - `--follow-imports`: Follows and includes all imports
   - `--assume-yes-for-downloads`: Automatically downloads required components
   - `--msvc=latest`: Uses the latest MSVC compiler
   - `--jobs=6`: Utilizes 6 CPU cores for compilation
   - `--remove-output`: Removes output after creating the final executable
   - `--full-compat`: Ensures full compatibility mode
   - `--plugin-enable=pyside6`: Enables PySide6 plugin (for Qt applications)
   - `--windows-console-mode=disable`: Disables console window for GUI applications
   - `--windows-icon-from-ico=pickaxe.ico`: Sets the application icon (used twice for redundancy)
   - `--include-data-file=pickaxe.ico=pickaxe.ico`: Includes the icon file in the executable
   - `--include-data-file=pickaxe.png=pickaxe.png`: Includes the splash screen image in the executable
   - `--onefile-windows-splash-screen-image=pickaxe.png`: Sets a splash screen image
   - `--output-filename=PickAxe.exe`: Names the output executable

## Notes

- Ensure all paths in the commands are correct for your system.
- The compilation process may take some time depending on your project size and computer specifications.
- Adjust the `--jobs` parameter based on your CPU core count for optimal performance.
- The `--windows-icon-from-ico` option is used twice and the icon file is included as data to ensure it's properly embedded.
- Both the icon file (pickaxe.ico) and the splash screen image (pickaxe.png) are included in the executable, so they don't need to be in the same folder as the .exe after compilation.

