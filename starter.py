import os
import sys
import tempfile
from main import Pickaxe, resource_path # Assuming Pickaxe is in main.py
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
import time

if "NUITKA_ONEFILE_PARENT" in os.environ:
   splash_filename = os.path.join(
      tempfile.gettempdir(),
      "onefile_%d_splash_feedback.tmp" % int(os.environ["NUITKA_ONEFILE_PARENT"]),
   )

   if os.path.exists(splash_filename):
      os.unlink(splash_filename)

def is_running_in_debugger():
    """Check if we're running in a debugging session"""
    try:
        # Check for VSCode debugger
        return ('debugpy' in sys.modules) + sys.modules['debugpy'].__file__.find('/.vscode/extensions/') > -1
    except Exception:
        return False

# from main import *

if __name__ == "__main__":
    app = QApplication(sys.argv) # sys.argv contains command line arguments
    app.setStyle('Fusion')
    app.desktopSettingsAware()
    
    icon_path = resource_path("pickaxe.ico")
    app_icon = QIcon(icon_path)           
    app.setWindowIcon(app_icon)            
    
    viewer = Pickaxe() # Create the main window instance

    # Check for file paths passed as command-line arguments
    # sys.argv[0] is the script name/path itself.
    # Files passed by "Open With" will be sys.argv[1], sys.argv[2], etc.
    file_to_open_on_startup = None
    if len(sys.argv) > 1:
        # Check if the argument is a valid file (could be other flags too)
        potential_file = sys.argv[1]
        if os.path.isfile(potential_file) and \
           potential_file.lower().endswith(('.csv', '.xlsx', '.xls', '.xlsm', '.xlsb')):
            file_to_open_on_startup = potential_file
            print(f"Attempting to open file from command line: {file_to_open_on_startup}")

    viewer.show()

    if file_to_open_on_startup:
        viewer.load_file(file_to_open_on_startup)
    elif is_running_in_debugger(): # Your existing debugger logic
        # Make sure your sample path is correct or use an absolute path for debugging
        # sample_debug_file = r"data samples scrambled\policy_data_sample.csv" 
        # For robustness in debugging, use an absolute path or ensure CWD is correct
        sample_debug_file = os.path.join(os.path.dirname(__file__), "data samples scrambled", "policy_data_sample.csv")
        if os.path.exists(sample_debug_file):
            viewer.load_file(sample_debug_file)
        else:
            print(f"Debug sample file not found: {sample_debug_file}")
            # viewer.load_file() # Ask user if debug sample not found
    else:
        viewer.load_file() # Prompts user if no command-line file and not debugging

    sys.exit(app.exec())
