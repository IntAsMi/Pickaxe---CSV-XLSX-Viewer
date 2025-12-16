import sys
from cx_Freeze import setup, Executable

# base="Win32GUI" is used for a GUI application on Windows to hide the console window
base = None
if sys.platform == "win32":
    base = "Win32GUI"

build_exe_options = {
    "packages": ["os", "sys", "re", "polars", "PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore", "threading", "json", "platform", "numpy"],
    "zip_include_packages": ["PySide6", "shiboken6"],
    "includes": ["main", "starter", "data_transformer", "visual_workshop", "operation_logger"],
    "include_files": ["pickaxe.ico", "settings.ico", "vw.ico", "magic-wand.ico"],
    "excludes": ["tkinter"]
}

setup(
    name="Pickaxe",
    version="2.0.1",
    description="A powerful data tool",
    options={"build_exe": build_exe_options},
    executables=[Executable("starter.py", base=base, target_name="Pickaxe", icon="pickaxe.ico")]
)

# to build, run:
# python setup.py build