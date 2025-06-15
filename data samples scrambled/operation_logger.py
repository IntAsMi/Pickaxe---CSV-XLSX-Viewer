# operation_logger.py (Minor update to header in set_log_file)
from datetime import datetime
import os
import threading
import json 
import platform 

if platform.system() == "Windows":
    import ctypes
    FILE_ATTRIBUTE_HIDDEN = 0x02

class OperationLogger:
    _instance = None
    _lock = threading.Lock() 

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance: 
                    cls._instance = super(OperationLogger, cls).__new__(cls)
                    cls._instance._initialized_once = False 
                    cls._instance.log_filename = "default_activity_log.md" 
                    cls._instance.current_app_context = "UnsetApp" 
        return cls._instance

    def __init__(self): 
        if self._initialized_once:
            return
        self._initialized_once = True

    def _set_file_hidden_windows(self, filepath):
        if platform.system() == "Windows":
            try:
                ret = ctypes.windll.kernel32.SetFileAttributesW(str(filepath), FILE_ATTRIBUTE_HIDDEN)
                if ret == 0: 
                    print(f"Warning: Could not set hidden attribute on {filepath}. Error code: {ctypes.GetLastError()}")
            except Exception as e:
                print(f"Warning: Failed to set hidden attribute on {filepath}: {e}")

    def set_log_file(self, log_filepath, app_name_for_header="Application", associated_data_file="N/A"):
        with self._lock: 
            self.log_filename = log_filepath
            self.current_app_context = app_name_for_header 

            log_dir = os.path.dirname(self.log_filename)
            if log_dir and not os.path.exists(log_dir): # Ensure directory exists
                try:
                    os.makedirs(log_dir, exist_ok=True) # exist_ok=True for safety
                except OSError as e:
                    print(f"Error: Could not create log directory {log_dir}: {e}")
                    return

            file_existed = os.path.exists(self.log_filename)
            # Always open in append mode. Header is only written if file is truly new.
            # This prevents header duplication if set_log_file is called multiple times for the same file.
            if not file_existed:
                try:
                    with open(self.log_filename, "w", encoding="utf-8") as f: # Use 'w' for new file to ensure clean start
                        f.write(f"# Activity Log\n")
                        f.write(f"## Data File: {os.path.basename(associated_data_file)}\n")
                        f.write(f"## Logged by: {app_name_for_header}\n\n")
                        f.write(f"Log session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("-" * 30 + "\n\n")
                    self._set_file_hidden_windows(self.log_filename)
                except IOError as e:
                    print(f"Error: Could not initialize log file {self.log_filename}: {e}")
            # If file existed, we just append to it without rewriting the header.

    def _format_details(self, details):
        
        if not details:
            return ""
        
        formatted_details = "**Details:**\n"
        if isinstance(details, dict):
            for key, value in details.items():
                value_str = str(value)
                if key != "Expression" and not ("error" in value_str.lower()):
                    if len(value_str) > 1000 and "\n" not in value_str:
                        value_str = value_str[:1000] + "..."
                
                if "\n" in value_str:
                    indented_value = "\n".join(["  " + line for line in value_str.splitlines()])
                    formatted_details += f"  - **{key}:**\n```\n{indented_value}\n```\n"
                else:
                    formatted_details += f"  - **{key}:** `{value_str}`\n"
        elif isinstance(details, list):
            for item in details:
                item_str = str(item)
                if len(item_str) > 1000 and "\n" not in item_str:
                     item_str = item_str[:1000] + "..."
                if "\n" in item_str:
                    indented_item = "\n".join(["    " + line for line in item_str.splitlines()])
                    formatted_details += f"  - List Item:\n```\n{indented_item}\n```\n"
                else:
                    formatted_details += f"  - `{item_str}`\n"
        else: 
            details_str = str(details)
            if len(details_str) > 2000: 
                details_str = details_str[:2000] + "..."
            formatted_details += "```\n" + details_str + "\n```\n"
        return formatted_details


    def log_action(self, app_name, category, description, details=None, df_shape_before=None, df_shape_after=None):
        # ... (same as before, ensures log_filename is checked) ...
        if not hasattr(self, 'log_filename') or not self.log_filename:
            print(f"Logger file not set. Action not logged: [{app_name}] {category} - {description}")
            return
            
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"## [{timestamp}] {app_name} - {category}\n\n" 
        log_entry += f"**Description:** {description}\n\n"

        if df_shape_before:
            log_entry += f"**DataFrame Shape Before:** {df_shape_before[0]}R x {df_shape_before[1]}C\n"
        if df_shape_after:
            log_entry += f"**DataFrame Shape After:** {df_shape_after[0]}R x {df_shape_after[1]}C\n"
        if df_shape_before or df_shape_after:
            log_entry += "\n" 

        log_entry += self._format_details(details)
        
        log_entry += "\n---\n\n"
        self._write_log(log_entry)

    def _write_log(self, message):
        
        try:
            with self._lock: 
                with open(self.log_filename, "a", encoding="utf-8") as f:
                    f.write(message)
        except IOError as e:
            print(f"Error: Could not write to log file {self.log_filename}: {e}")
        except AttributeError: 
            print(f"Error: Log filename not set. Message not written: {message[:100]}")

    def log_dataframe_load(self, app_name, loaded_filename, sheet_name=None, rows=0, cols=0, load_time_sec=0):
        
        details = {
            "File": os.path.basename(loaded_filename),
            "Full Path": loaded_filename, 
        }
        if sheet_name:
            details["Sheet"] = sheet_name
        details["Initial Rows"] = rows
        details["Initial Columns"] = cols
        if load_time_sec > 0:
            details["Load Time (s)"] = f"{load_time_sec:.2f}"
        
        self.log_action(app_name, "File Operation", f"Loaded data from '{os.path.basename(loaded_filename)}'", details)

    def log_dataframe_save(self, app_name, saved_filename, rows=0, cols=0, source_data_name="current data"):
        
        details = {
            "Saved To File": os.path.basename(saved_filename),
            "Full Path": saved_filename,
            "Source Data Name/Hint": source_data_name, 
            "Rows Saved": rows,
            "Columns Saved": cols,
        }
        self.log_action(app_name, "File Operation", f"Saved data to '{os.path.basename(saved_filename)}'", details)

logger = OperationLogger()