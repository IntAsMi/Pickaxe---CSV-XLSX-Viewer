import sys
import os
import re
import polars as pl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableView, QHeaderView, QFileDialog, QRadioButton,
    QComboBox, QMenu, QInputDialog, QMessageBox, QProgressBar, QGridLayout,
    QScrollArea, QFrame, QDateEdit, QSizePolicy, QCheckBox, QAbstractItemView,
    QCompleter, QDialog, QDialogButtonBox, QButtonGroup, QTextEdit, QTabWidget,
    QListWidget, QListWidgetItem, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QTextBrowser
)
from PySide6.QtCore import Qt, QAbstractTableModel, QDate, QThread, Signal, Slot, QTimer, QUrl, QStringListModel, QSize, QRegularExpression
from PySide6.QtGui import (
    QPainter, QAction, QIcon, QFont, QColor, QDesktopServices, QPixmap,
    QSyntaxHighlighter, QTextCharFormat
)

from datetime import datetime, date
# Assuming main.py is in the same directory or accessible via PYTHONPATH
from main import FileLoaderWorker, ContainerFileNavigator, PolarsModel, resource_path
from operation_logger import logger
import traceback

# For bij_match
try:
    from fuzzywuzzy import fuzz, process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    print("Warning: fuzzywuzzy library not found. Advanced column name matching for concatenation will be limited.")
import collections

# --- Polars Syntax Highlighter --- ( 그대로 유지 )
class PolarsSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        # 1. Polars module functions (e.g., pl.col, pl.lit)
        pl_function_format = QTextCharFormat()
        pl_function_format.setForeground(QColor("#0000CD")) # MediumBlue
        pl_function_format.setFontWeight(QFont.Bold)
        pl_functions = [
            "pl\\.col", "pl\\.lit", "pl\\.when", "pl\\.all", "pl\\.sum", "pl\\.mean", "pl\\.min", "pl\\.max",
            "pl\\.count", "pl\\.first", "pl\\.last", "pl\\.head", "pl\\.tail", "pl\\.select", "pl\\.filter",
            "pl\\.with_columns", "pl\\.group_by", "pl\\.agg", "pl\\.sort", "pl\\.concat",
            "pl\\.DataFrame", "pl\\.Series", "pl\\.Expr", "pl\\.arange", "pl\\.concat_str",
            "pl\\.coalesce", "pl\\.format", "pl\\.fold", "pl\\.repeat"
        ]
        for word in pl_functions:
            pattern = QRegularExpression(f"\\b{word}\\b")
            self.highlighting_rules.append({"pattern": pattern, "format": pl_function_format})

        # 2. Expression methods (e.g., .alias(), .cast())
        method_format = QTextCharFormat()
        method_format.setForeground(QColor("#8A2BE2")) # BlueViolet
        method_pattern = QRegularExpression("\\.([a-zA-Z_][a-zA-Z0-9_]*)\\b(?=\\s*\\()") 
        self.highlighting_rules.append({"pattern": method_pattern, "format": method_format, "capture_group": 1})


        # 3. Keywords (True, False, None, and common operators as words)
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("darkblue")) 
        keyword_format.setFontWeight(QFont.Bold)
        keywords = ["True", "False", "None", "and", "or", "not", "in", "is", "as", "if", "else", "then", "otherwise"] 
        for word in keywords:
            pattern = QRegularExpression(f"\\b{word}\\b")
            self.highlighting_rules.append({"pattern": pattern, "format": keyword_format})

        # 4. Polars Data Types
        datatype_format = QTextCharFormat()
        datatype_format.setForeground(QColor("#2E8B57")) # SeaGreen
        datatypes = [
            "pl\\.Int8", "pl\\.Int16", "pl\\.Int32", "pl\\.Int64",
            "pl\\.UInt8", "pl\\.UInt16", "pl\\.UInt32", "pl\\.UInt64",
            "pl\\.Float32", "pl\\.Float64", "pl\\.Boolean", "pl\\.Utf8",
            "pl\\.Date", "pl\\.Datetime", "pl\\.Duration", "pl\\.Time",
            "pl\\.List", "pl\\.Struct", "pl\\.Categorical", "pl\\.Object"
        ]
        for dtype_str in datatypes:
            pattern = QRegularExpression(f"\\b{dtype_str}\\b")
            self.highlighting_rules.append({"pattern": pattern, "format": datatype_format})


        # 5. Operators
        operator_format = QTextCharFormat()
        operator_format.setForeground(QColor("#B22222")) # Firebrick
        operator_format.setFontWeight(QFont.Normal) 
        operators = ["=", "==", "!=", "<", "<=", ">", ">=", "\\+", "-", "\\*", "/", "%", "&", "\\|", "~", "\\^", "\\*\\*"] 
        for op in operators:
            pattern = QRegularExpression(re.escape(op) if len(op)==1 and op not in ['&','|','~','^'] else op) 
            self.highlighting_rules.append({"pattern": pattern, "format": operator_format})

        # 6. Strings (single and double quoted)
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000")) # DarkGreen
        self.highlighting_rules.append({"pattern": QRegularExpression(r'"[^"\\]*(\\.[^"\\]*)*"'), "format": string_format})
        self.highlighting_rules.append({"pattern": QRegularExpression(r"'[^'\\]*(\\.[^'\\]*)*'"), "format": string_format})
        self.highlighting_rules.append({"pattern": QRegularExpression(r'r"[^"]*"'), "format": string_format})
        self.highlighting_rules.append({"pattern": QRegularExpression(r"r'[^']*'"), "format": string_format})


        # 7. Numbers (integers, floats, scientific notation)
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#1E90FF")) # DodgerBlue
        self.highlighting_rules.append({"pattern": QRegularExpression("\\b[0-9]+\\.?[0-9]*([eE][-+]?[0-9]+)?\\b"), "format": number_format})


        # 8. Comments
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("gray"))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append({"pattern": QRegularExpression("#[^\n]*"), "format": comment_format})
        
        # 9. Parentheses, Brackets, Braces
        punctuation_format = QTextCharFormat()
        punctuation_format.setForeground(QColor("darkGray")) 
        punctuation_format.setFontWeight(QFont.Bold)
        for punc in ["\\(", "\\)", "\\[", "\\]", "\\{", "\\}"]: # Escaped for regex
            pattern = QRegularExpression(punc)
            self.highlighting_rules.append({"pattern": pattern, "format": punctuation_format})


    def highlightBlock(self, text):
        for rule in self.highlighting_rules:
            regex = rule["pattern"]
            iterator = regex.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                start_index = match.capturedStart(rule.get("capture_group", 0)) 
                length = match.capturedLength(rule.get("capture_group", 0))
                if start_index != -1 and length > 0: 
                    self.setFormat(start_index, length, rule["format"])
        self.setCurrentBlockState(0)


class AsyncFileLoaderThreadDT(QThread):
    finished_signal = Signal(object, dict)
    error_signal = Signal(str)
    request_npz_array_selection = Signal(list, str)
    request_pickle_item_selection = Signal(list, str, str)

    def __init__(self, file_path, sheet_name=None, parent_gui=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.parent_gui = parent_gui

        self.worker_instance = FileLoaderWorker(file_path, sheet_name, parent_gui=parent_gui)

        self.worker_instance.request_npz_array_selection.connect(self.request_npz_array_selection)
        self.worker_instance.request_pickle_item_selection.connect(self.request_pickle_item_selection)

    def run(self):
        try:
            df, file_info = self.worker_instance.run()
            self.finished_signal.emit(df, file_info)
        except Exception as e:
            self.error_signal.emit(f"Error loading file: {str(e)}\nTraceback:\n{traceback.format_exc()}")

    @Slot(str)
    def on_npz_array_selected_relayed(self, array_name):
        if hasattr(self.worker_instance, 'on_npz_array_selected'):
            self.worker_instance.on_npz_array_selected(array_name)

    @Slot(str)
    def on_pickle_item_selected_relayed(self, item_path):
        if hasattr(self.worker_instance, 'on_pickle_item_selected'):
            self.worker_instance.on_pickle_item_selected(item_path)

def bij_match(reference_cols, target_cols, ignore_not_found=False, score_threshold=75):
    if not reference_cols or not target_cols:
        return {ref_col: None for ref_col in reference_cols} if not ignore_not_found else {}

    def normalize(name):
        return re.sub(r'[^a-zA-Z0-9]', '', str(name)).lower()

    norm_to_orig_ref = {normalize(col): col for col in reference_cols}
    norm_to_orig_target = {normalize(col): col for col in target_cols}

    norm_ref_list = list(norm_to_orig_ref.keys())
    norm_target_list = list(norm_to_orig_target.keys())

    final_mapping = {}
    used_norm_target_cols = set()

    for norm_ref_key in norm_ref_list:
        orig_ref_col = norm_to_orig_ref[norm_ref_key]
        if norm_ref_key in norm_target_list and norm_ref_key not in used_norm_target_cols:
            final_mapping[orig_ref_col] = norm_to_orig_target[norm_ref_key]
            used_norm_target_cols.add(norm_ref_key)

    unmatched_orig_ref_cols = [
        ref_col for ref_col in reference_cols if ref_col not in final_mapping
    ]
    
    available_orig_target_cols_for_fuzzy = [
        norm_to_orig_target[norm_target_key] for norm_target_key in norm_target_list 
        if norm_target_key not in used_norm_target_cols
    ]
    
    if FUZZYWUZZY_AVAILABLE and unmatched_orig_ref_cols and available_orig_target_cols_for_fuzzy:
        for orig_ref_col in unmatched_orig_ref_cols:
            if not available_orig_target_cols_for_fuzzy:
                final_mapping[orig_ref_col] = None
                continue

            best_match_info = process.extractOne(
                orig_ref_col, 
                available_orig_target_cols_for_fuzzy, 
                scorer=fuzz.token_sort_ratio
            )
            
            if best_match_info and best_match_info[1] >= score_threshold:
                best_target_orig_name = best_match_info[0]
                final_mapping[orig_ref_col] = best_target_orig_name
                available_orig_target_cols_for_fuzzy.remove(best_target_orig_name)
                used_norm_target_cols.add(normalize(best_target_orig_name))
            else:
                final_mapping[orig_ref_col] = None
    else: 
        for orig_ref_col in unmatched_orig_ref_cols:
            final_mapping[orig_ref_col] = None

    if ignore_not_found:
        return {k: v for k, v in final_mapping.items() if v is not None}
    else:
        for ref_c in reference_cols:
            if ref_c not in final_mapping:
                final_mapping[ref_c] = None
        return final_mapping


class DataTransformer(QMainWindow):
    MAX_PREVIEW_ROWS = 500

    def __init__(self, source_app=None, initial_df=None, df_name_hint=None, log_file_to_use=None, parent=None):
        super().__init__(parent)
        self.source_app = source_app
        self.current_df = None
        self.original_df_for_reset = None
        self.df_name_hint = df_name_hint or "Untitled Data"
        self.current_log_file_path = log_file_to_use

        self.other_df_for_combine = None
        self.other_df_name_hint = None
        self.other_df_file_info = {}
        
        self.applied_expressions_history = {} 

        self.undo_stack = []
        self.redo_stack = []

        self.other_df_conversion_radio_button_groups = []
        self._temp_cols_for_batch_other_df = []

        if self.current_log_file_path:
            logger.set_log_file(self.current_log_file_path, "DataTransformer (Continuing Session)", associated_data_file=self.df_name_hint)
        else:
            base_name = re.sub(r'\W+', '_', os.path.splitext(os.path.basename(self.df_name_hint))[0] if self.df_name_hint else "dt_data")
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_log_file_path = os.path.join(os.getcwd(), f".__log__{base_name}_{timestamp_str}_dt.md")
            logger.set_log_file(self.current_log_file_path, "DataTransformer", associated_data_file=self.df_name_hint or "N/A")

        self.setWindowTitle(f"Data Transformer - [{os.path.basename(self.df_name_hint)}]")
        self.setGeometry(200, 200, 1450, 900)
        icon_path = resource_path("settings.ico")
        if os.path.exists(icon_path): self.setWindowIcon(QIcon(icon_path))

        self._create_actions()
        self._create_menubar()
        self._setup_main_layout()

        if initial_df is not None:
            self.receive_dataframe(initial_df, self.df_name_hint, log_file_path_from_source=log_file_to_use)
        else:
            self._update_ui_for_data_state()
        
        if hasattr(self, 'create_modify_tab'): 
            self.tab_widget.setCurrentWidget(self.create_modify_tab)

        if hasattr(self, 'combine_data_tab') and hasattr(self, 'concat_sub_tab') and \
           hasattr(self, 'combine_operations_tab_widget') and hasattr(self, 'concat_type_combo'):
            # Only set combine tab's sub-tab if combine tab itself is not the default
            if self.tab_widget.currentWidget() != self.combine_data_tab:
                 self.combine_operations_tab_widget.setCurrentWidget(self.concat_sub_tab)
            
            idx = self.concat_type_combo.findData("custom_vertical_common")
            if idx != -1:
                self.concat_type_combo.setCurrentIndex(idx)
            else: 
                idx = self.concat_type_combo.findData("vertical_relaxed")
                if idx != -1: self.concat_type_combo.setCurrentIndex(idx)


        self.statusBar().showMessage("Data Transformer ready.")

    def _create_actions(self):
        self.open_action = QAction(QIcon.fromTheme("document-open"), "&Open Primary Data File...", self)
        self.open_action.triggered.connect(self._handle_open_file_directly)
        
        self.save_action = QAction(QIcon.fromTheme("document-save-as"), "&Save Transformed Data As...", self)
        self.save_action.triggered.connect(self._handle_save_file)
        
        self.describe_df_action = QAction(QIcon.fromTheme("document-properties"), "&Describe Current DataFrame...", self)
        self.describe_df_action.triggered.connect(self._handle_describe_df)
        
        self.exit_action = QAction(QIcon.fromTheme("application-exit"), "&Exit", self)
        self.exit_action.triggered.connect(self.close)

        self.undo_action = QAction(QIcon.fromTheme("edit-undo"), "&Undo", self)
        self.undo_action.triggered.connect(self._handle_undo)
        self.undo_action.setShortcut("Ctrl+Z")
        
        self.redo_action = QAction(QIcon.fromTheme("edit-redo"), "&Redo", self)
        self.redo_action.triggered.connect(self._handle_redo)
        self.redo_action.setShortcut("Ctrl+Y")

        self.reset_action = QAction(QIcon.fromTheme("edit-delete"), "&Reset All Transformations", self)
        self.reset_action.triggered.connect(self._handle_reset_transformations)

        self.send_to_source_action = QAction(QIcon.fromTheme("document-send"), "&Send Transformed Data to Source App", self)
        self.send_to_source_action.triggered.connect(self._handle_send_to_source)
        self.send_to_source_action.setEnabled(self.source_app is not None)

    def _create_menubar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.reset_action)

        data_menu = menubar.addMenu("&Data")
        data_menu.addAction(self.describe_df_action)
        data_menu.addSeparator()
        data_menu.addAction(self.send_to_source_action)

    def _setup_main_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_v_layout = QVBoxLayout(self.central_widget)

        self.df_info_label = QLabel(f"Current Data: {os.path.basename(self.df_name_hint)} | Dimensions: N/A")
        main_v_layout.addWidget(self.df_info_label)

        self.splitter = QSplitter(Qt.Vertical)
        main_v_layout.addWidget(self.splitter)

        self.tab_widget = QTabWidget()
        self._setup_create_fields_tab() 
        self._setup_reshape_tab()
        self._setup_combine_tab() 
        self.splitter.addWidget(self.tab_widget)

        preview_group = QGroupBox("DataFrame Preview (Max 500 rows)")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableView()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.preview_table.horizontalHeader().setStretchLastSection(False)
        preview_layout.addWidget(self.preview_table)
        self.splitter.addWidget(preview_group)

        self.splitter.setSizes([int(self.height() * 0.65), int(self.height() * 0.35)])

        self.status_bar = self.statusBar()


    def _setup_create_fields_tab(self):
        self.create_modify_tab = QWidget() 
        main_layout = QHBoxLayout(self.create_modify_tab) 

        self.create_fields_main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.create_fields_main_splitter)

        self.helper_groupbox = QGroupBox("Polars Expression Helper & Examples")
        helper_left_layout = QVBoxLayout(self.helper_groupbox) 
        self.polars_helper_text = QTextBrowser()
        self.polars_helper_text.setOpenExternalLinks(True)
        self._populate_polars_helper_text() 
        helper_left_layout.addWidget(self.polars_helper_text)
        
        self.create_fields_main_splitter.addWidget(self.helper_groupbox)

        right_pane_main_widget = QWidget()
        right_pane_main_layout = QVBoxLayout(right_pane_main_widget)
        right_pane_main_layout.setContentsMargins(0,0,0,0) 
        
        self.toggle_helper_button = QPushButton("Hide Helper")
        self.toggle_helper_button.setCheckable(True)
        self.toggle_helper_button.setChecked(True) 
        self.toggle_helper_button.clicked.connect(self._toggle_polars_helper)
        right_pane_main_layout.addWidget(self.toggle_helper_button, 0, Qt.AlignRight)


        self.create_fields_right_pane_splitter = QSplitter(Qt.Vertical) # Renamed for clarity
        right_pane_main_layout.addWidget(self.create_fields_right_pane_splitter)

        top_right_widget = QWidget()
        top_right_h_layout = QHBoxLayout(top_right_widget)

        self.available_cols_group_create = QGroupBox("Available Columns (Current DF)")
        col_list_layout = QVBoxLayout(self.available_cols_group_create)
        self.available_cols_list_create = self._create_checkbox_list_widget() 
        self.available_cols_list_create.itemDoubleClicked.connect(self._insert_col_into_expression_from_checkbox_list)
        col_list_layout.addWidget(self.available_cols_list_create)
        self.delete_fields_button = QPushButton("Delete Selected Field(s)")
        self.delete_fields_button.clicked.connect(self._handle_delete_fields)
        col_list_layout.addWidget(self.delete_fields_button)
        top_right_h_layout.addWidget(self.available_cols_group_create)

        self.applied_expressions_group = QGroupBox("Applied Expressions Log (Latest per Field)")
        applied_expr_layout = QVBoxLayout(self.applied_expressions_group)
        self.applied_expressions_log_table = QTableWidget()
        self.applied_expressions_log_table.setColumnCount(3)
        self.applied_expressions_log_table.setHorizontalHeaderLabels(["Field Name", "Expression", "Use"])
        self.applied_expressions_log_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.applied_expressions_log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.applied_expressions_log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.applied_expressions_log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.applied_expressions_log_table.setSelectionBehavior(QAbstractItemView.SelectRows) # Allow row selection for potential copy
        applied_expr_layout.addWidget(self.applied_expressions_log_table)
        top_right_h_layout.addWidget(self.applied_expressions_group)
        
        self.create_fields_right_pane_splitter.addWidget(top_right_widget)


        self.expression_input_group = QGroupBox("Expression Details") 
        input_layout = QGridLayout(self.expression_input_group)
        
        input_layout.addWidget(QLabel("New/Existing Field Name:"), 0, 0)
        self.new_field_name_edit = QLineEdit()
        self.new_field_name_edit.setPlaceholderText("e.g., 'new_col' or 'existing_col_to_overwrite'")
        input_layout.addWidget(self.new_field_name_edit, 0, 1)

        input_layout.addWidget(QLabel("Polars Expression:"), 1, 0, Qt.AlignTop)
        self.polars_expression_edit = QTextEdit()
        self.polars_expression_edit.setPlaceholderText("e.g., pl.col('some_col') + pl.col('other_col')")
        self.polars_expression_edit.setMinimumHeight(120) 
        self.polars_syntax_highlighter = PolarsSyntaxHighlighter(self.polars_expression_edit.document())
        input_layout.addWidget(self.polars_expression_edit, 1, 1)
        
        self.apply_expression_button = QPushButton("Apply Expression")
        self.apply_expression_button.clicked.connect(self._handle_apply_expression)
        input_layout.addWidget(self.apply_expression_button, 2, 0, 1, 2)
        self.create_fields_right_pane_splitter.addWidget(self.expression_input_group)
        
        self.create_fields_main_splitter.addWidget(right_pane_main_widget)
        
        self.create_fields_main_splitter.setSizes([int(self.width() * 0.35), int(self.width() * 0.65)])
        self.create_fields_right_pane_splitter.setSizes([int(self.height() * 0.4), int(self.height() * 0.25)]) # Adjusted for more space for expression box


        self.tab_widget.addTab(self.create_modify_tab, "Create/Modify Fields")

    def _populate_polars_helper_text(self):
        html_content = """
        <html><head><style>
            body { font-family: sans-serif; font-size: 9pt; margin: 5px; }
            h4 { color: #2c3e50; margin-bottom: 3px; margin-top: 10px; border-bottom: 1px solid #ccc; padding-bottom: 2px;}
            p { margin-top: 2px; margin-bottom: 6px; }
            code { background-color: #e9e9e9; padding: 2px 4px; border-radius: 3px; font-family: Consolas, 'Courier New', monospace;}
            .code-block { background-color: #f4f4f4; border: 1px solid #ddd; padding: 6px; margin: 6px 0; border-radius: 3px; font-family: Consolas, 'Courier New', monospace; white-space: pre-wrap; font-size: 8.5pt; }
            ul { padding-left: 20px; margin-top: 2px; margin-bottom: 8px;}
            li { margin-bottom: 3px; }
            a { color: #3498db; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style></head><body>
        <h4>Core Concepts:</h4>
        <ul>
            <li>Select Column: <code>pl.col("ColumnName")</code> (Excel: <code>A1</code> or named range)</li>
            <li>Literal Value: <code>pl.lit("hello")</code>, <code>pl.lit(123)</code>, <code>pl.lit(True)</code></li>
            <li>Alias (Rename in expression): <code>pl.col("A").alias("NewLabel")</code></li>
        </ul>

        <h4>Arithmetic & Comparisons:</h4>
        <div class="code-block">pl.col("Price") * pl.col("Quantity") * (1 - pl.col("Discount"))<br>pl.col("Score") > 80<br>pl.col("ValueA") == pl.col("ValueB")</div>

        <h4>String Operations (<code>.str</code>):</h4>
        <ul>
            <li>Contains: <code>pl.col("Notes").str.contains("important")</code> (Excel: <code>ISNUMBER(SEARCH("important", A1))</code>)</li>
            <li>Replace: <code>pl.col("Category").str.replace_all("TypeA", "GroupA")</code> (Excel: <code>SUBSTITUTE(A1, "TypeA", "GroupA")</code>)</li>
            <li>Length: <code>pl.col("Name").str.len_chars()</code> (Excel: <code>LEN(A1)</code>)</li>
            <li>To Uppercase: <code>pl.col("Code").str.to_uppercase()</code> (Excel: <code>UPPER(A1)</code>)</li>
            <li>Concatenate: <code>pl.format("{} - {}", pl.col("FirstName"), pl.col("LastName"))</code> or <br><code>pl.concat_str([pl.col("FirstName"), pl.lit(" "), pl.col("LastName")])</code> (Excel: <code>A1 & " " & B1</code>)</li>
            <li>Split & Get Part: <code>pl.col("FullName").str.split(" ").list.get(0).alias("FirstName")</code></li>
            <li>Extract with Regex: <code>pl.col("ID").str.extract(r"ITEM-(\\d+)", 1).cast(pl.Int64)</code> (extracts number after "ITEM-")</li>
        </ul>

        <h4>Date/Datetime Operations (<code>.dt</code>):</h4>
        <p><em>First, ensure column is a date/datetime type. If it's a string:</em><br>
        <code>pl.col("DateString").str.to_date("%Y-%m-%d", strict=False)</code> (e.g., "2023-12-31")<br>
        <code>pl.col("DateTimeString").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)</code></p>
        <ul>
            <li>Year: <code>pl.col("OrderDate").dt.year()</code> (Excel: <code>YEAR(A1)</code>)</li>
            <li>Month: <code>pl.col("OrderDate").dt.month()</code> (Excel: <code>MONTH(A1)</code>)</li>
            <li>Day: <code>pl.col("OrderDate").dt.day()</code> (Excel: <code>DAY(A1)</code>)</li>
            <li>Format to String: <code>pl.col("EventDate").dt.strftime("%A, %B %d, %Y")</code></li>
            <li>Add Duration: <code>pl.col("StartDate") + pl.duration(days=7)</code></li>
        </ul>

        <h4>Conditional Logic (When-Then-Otherwise):</h4>
        <p>Excel equivalent: Nested <code>IF()</code> statements.</p>
        <div class="code-block">
        pl.when(pl.col("Value") > 100).then(pl.lit("A"))<br>
        &nbsp;&nbsp;.when(pl.col("Value") > 50).then(pl.lit("B"))<br>
        &nbsp;&nbsp;.otherwise(pl.lit("C"))
        </div>
        <p>With multiple conditions (AND):</p>
        <div class="code-block">
        pl.when( (pl.col("Type") == "X") & (pl.col("Amount") > 1000) )<br>
        &nbsp;&nbsp;.then(pl.lit("HighValueX"))<br>
        &nbsp;&nbsp;.otherwise(pl.lit("Other"))
        </div>

        <h4>Window Functions (<code>.over("group_col")</code>):</h4>
        <p>Calculations within groups/partitions. Excel: SUMIFS, COUNTIFS, or array formulas with conditions.</p>
        <ul>
            <li>Sum per category: <code>pl.col("Sales").sum().over("ProductCategory")</code></li>
            <li>Rank: <code>pl.col("Score").rank("descending").over("Department")</code></li>
            <li>Get first value in group: <code>pl.col("Timestamp").first().over("UserID")</code></li>
             <li>Cumulative Sum: <code>pl.col("DailySales").cumsum().over("Month")</code></li>
        </ul>

        <h4>Type Casting:</h4>
        <ul>
            <li>To Integer: <code>pl.col("ValueAsString").cast(pl.Int64, strict=False)</code></li>
            <li>To Float: <code>pl.col("NumString").cast(pl.Float64, strict=False)</code></li>
            <li>To String: <code>pl.col("NumericColumn").cast(pl.Utf8)</code></li>
        </ul>
        <p><em>For more, see the <a href="https://pola-rs.github.io/polars-book/user-guide/expressions/column_selection.html">Polars Expressions Guide</a> and <a href="https://pola-rs.github.io/polars-book/user-guide/cheat_sheet.html">Cheat Sheet</a>.</em></p>
        </body></html>
        """
        self.polars_helper_text.setHtml(html_content)

    def _toggle_polars_helper(self):
        is_checked = self.toggle_helper_button.isChecked()
        self.helper_groupbox.setVisible(is_checked) # This is correct
        self.toggle_helper_button.setText("Hide Helper" if is_checked else "Show Helper")
        # No need to manipulate splitter sizes directly if setVisible works as expected on a top-level widget in splitter.
        # If self.helper_groupbox is the widget directly added to splitter:
        # current_sizes = self.create_fields_main_splitter.sizes()
        # if is_checked: # Show
        #     if current_sizes[0] < 50: # If it was effectively hidden
        #         self.create_fields_main_splitter.setSizes([300, max(50, current_sizes[1])]) # Restore or set default
        # else: # Hide
        #     self.create_fields_main_splitter.setSizes([0, sum(current_sizes)])


    def _insert_col_into_expression(self, item_widget): 
        # item is QListWidgetItem from a _create_checkbox_list_widget
        # The text is on the checkbox, which is the item's widget
        if isinstance(item_widget, QListWidgetItem): # Check if it's the item itself
            widget = self.available_cols_list_create.itemWidget(item_widget) # Get the checkbox
            if isinstance(widget, QCheckBox):
                col_name_with_type = widget.text()
                col_name = col_name_with_type.split(" (")[0]
                self.polars_expression_edit.insertPlainText(f"pl.col('{col_name}')")

    def _insert_col_into_expression_from_checkbox_list(self, item): # item is QListWidgetItem
        widget = self.available_cols_list_create.itemWidget(item)
        if isinstance(widget, QCheckBox):
            col_name_with_type = widget.text()
            col_name = col_name_with_type.split(" (")[0]
            self.polars_expression_edit.insertPlainText(f"pl.col('{col_name}')")


    def _setup_reshape_tab(self):
        tab = QWidget() 
        main_reshape_layout = QVBoxLayout(tab)

        reshape_help_layout = QHBoxLayout()
        reshape_help_layout.addStretch() 
        reshape_help_button = QPushButton("?")
        reshape_help_button.setFixedSize(25,25)
        reshape_help_button.setToolTip("Help on Pivoting and Melting Data")
        reshape_help_button.clicked.connect(self._show_reshape_help)
        reshape_help_layout.addWidget(reshape_help_button)
        main_reshape_layout.addLayout(reshape_help_layout)

        self.reshape_sub_tab_widget = QTabWidget()
        main_reshape_layout.addWidget(self.reshape_sub_tab_widget)

        # --- Pivot Sub-Tab ---
        pivot_sub_tab = QWidget()
        pivot_main_layout = QVBoxLayout(pivot_sub_tab) 
        
        pivot_instruction = QLabel("<b>Pivot:</b> Transforms data from a 'long' format to a 'wider' format by turning unique values from one column into new column headers.")
        pivot_instruction.setWordWrap(True)
        pivot_main_layout.addWidget(pivot_instruction)

        pivot_group = QGroupBox("Pivot Configuration") 
        pivot_layout = QGridLayout(pivot_group)
        
        pivot_layout.addWidget(QLabel("Index Column(s) (Rows in Excel Pivot):"), 0, 0, Qt.AlignTop)
        self.pivot_index_cols_list = self._create_checkbox_list_widget()
        pivot_layout.addWidget(self.pivot_index_cols_list, 0, 1, 1, 2) # Span 1 row, 2 cols for list

        pivot_layout.addWidget(QLabel("Columns (from values of) (Columns in Excel Pivot):"), 1, 0)
        self.pivot_columns_col_combo = QComboBox()
        pivot_layout.addWidget(self.pivot_columns_col_combo, 1, 1)

        pivot_layout.addWidget(QLabel("Values (populate new cols) (Values in Excel Pivot):"), 2, 0)
        self.pivot_values_col_combo = QComboBox()
        pivot_layout.addWidget(self.pivot_values_col_combo, 2, 1)

        pivot_agg_group = QGroupBox("Aggregation for Duplicates")
        pivot_agg_layout = QHBoxLayout(pivot_agg_group)
        pivot_agg_layout.addWidget(QLabel("Function:"))
        self.pivot_agg_func_combo = QComboBox()
        self.pivot_agg_func_combo.addItems(["first", "sum", "mean", "median", "min", "max", "count", "list"])
        pivot_agg_layout.addWidget(self.pivot_agg_func_combo)
        pivot_layout.addWidget(pivot_agg_group, 1, 2, 2, 1) # Span 2 rows

        self.apply_pivot_button = QPushButton("Apply Pivot")
        self.apply_pivot_button.clicked.connect(self._handle_apply_pivot)
        pivot_layout.addWidget(self.apply_pivot_button, 3, 0, 1, 3) # Span all columns in grid
        pivot_main_layout.addWidget(pivot_group)
        pivot_main_layout.addStretch()
        self.reshape_sub_tab_widget.addTab(pivot_sub_tab, "Pivot")

        # --- Melt Sub-Tab ---
        melt_sub_tab = QWidget()
        melt_main_layout = QVBoxLayout(melt_sub_tab) 
        
        melt_instruction = QLabel("<b>Melt (Unpivot):</b> Transforms data from a 'wide' format to a 'longer' format by stacking specified columns into key-value pairs.")
        melt_instruction.setWordWrap(True)
        melt_main_layout.addWidget(melt_instruction)

        melt_group = QGroupBox("Melt Configuration") 
        melt_layout = QGridLayout(melt_group)

        id_vars_group = QGroupBox("Identifier Columns (to keep as is)")
        id_vars_layout = QVBoxLayout(id_vars_group)
        self.melt_id_vars_list = self._create_checkbox_list_widget()
        id_vars_layout.addWidget(self.melt_id_vars_list)
        melt_layout.addWidget(id_vars_group, 0, 0, 1, 2) # Span 2 cols

        value_vars_group = QGroupBox("Columns to Unpivot (Value Variables - Optional)")
        value_vars_layout = QVBoxLayout(value_vars_group)
        self.melt_value_vars_list = self._create_checkbox_list_widget()
        self.melt_value_vars_list.setToolTip("If empty, all columns NOT selected as ID Variables will be unpivoted.")
        value_vars_layout.addWidget(self.melt_value_vars_list)
        melt_layout.addWidget(value_vars_group, 0, 2, 1, 2) # Span 2 cols
        
        output_names_group = QGroupBox("Output Column Names")
        output_names_layout = QGridLayout(output_names_group)
        output_names_layout.addWidget(QLabel("New 'Variable' Column Name:"), 0, 0)
        self.melt_variable_name_edit = QLineEdit("variable")
        output_names_layout.addWidget(self.melt_variable_name_edit, 0, 1)
        output_names_layout.addWidget(QLabel("New 'Value' Column Name:"), 1, 0)
        self.melt_value_name_edit = QLineEdit("value")
        output_names_layout.addWidget(self.melt_value_name_edit, 1, 1)
        melt_layout.addWidget(output_names_group, 1, 0, 1, 4) # Span all

        self.apply_melt_button = QPushButton("Apply Melt")
        self.apply_melt_button.clicked.connect(self._handle_apply_melt)
        melt_layout.addWidget(self.apply_melt_button, 2, 0, 1, 4) 
        melt_main_layout.addWidget(melt_group)
        melt_main_layout.addStretch()
        self.reshape_sub_tab_widget.addTab(melt_sub_tab, "Melt (Unpivot)")
        
        self.tab_widget.addTab(tab, "Reshape Data (Pivot/Melt)")


    def _setup_combine_tab(self):
        self.combine_data_tab = QWidget() 
        main_combine_layout = QVBoxLayout(self.combine_data_tab)

        load_other_group = QGroupBox("Load 'Other' DataFrame for Combining")
        load_other_layout = QGridLayout(load_other_group)
        self.other_df_load_button = QPushButton("Load Other DataFrame from File...")
        self.other_df_load_button.clicked.connect(self._handle_load_other_df)
        load_other_layout.addWidget(self.other_df_load_button, 0, 0)
        self.other_df_info_label = QLabel("Other DF: Not loaded")
        self.other_df_info_label.setWordWrap(True)
        load_other_layout.addWidget(self.other_df_info_label, 0, 1, 1, 2)
        self.other_df_progress_bar = QProgressBar()
        self.other_df_progress_bar.setVisible(False)
        load_other_layout.addWidget(self.other_df_progress_bar, 1, 0, 1, 3)
        main_combine_layout.addWidget(load_other_group)

        manage_other_group = QGroupBox("Manage 'Other' DataFrame Columns & Types")
        manage_other_layout = QHBoxLayout(manage_other_group)
        self.other_df_cols_list = self._create_checkbox_list_widget() 
        manage_other_layout.addWidget(self.other_df_cols_list, 2)

        type_management_vlayout = QVBoxLayout()
        type_management_vlayout.addWidget(QLabel("Cast Selected to:"))
        self.other_df_type_conversion_combo = QComboBox()
        self.other_df_type_conversion_combo.addItems(["String", "Integer", "Float", "Datetime", "Categorical", "Boolean"])
        type_management_vlayout.addWidget(self.other_df_type_conversion_combo)
        self.other_df_cast_selected_button = QPushButton("Cast Selected Column(s)")
        self.other_df_cast_selected_button.clicked.connect(self._handle_cast_other_df_columns)
        type_management_vlayout.addWidget(self.other_df_cast_selected_button)
        self.other_df_suggest_types_button = QPushButton("Run Type Suggestion")
        self.other_df_suggest_types_button.clicked.connect(self._handle_suggest_types_other_df)
        type_management_vlayout.addWidget(self.other_df_suggest_types_button)
        type_management_vlayout.addStretch()
        manage_other_layout.addLayout(type_management_vlayout, 1)
        main_combine_layout.addWidget(manage_other_group)

        self.combine_operations_tab_widget = QTabWidget()
        self._setup_merge_sub_tab()
        self._setup_concat_sub_tab() 
        main_combine_layout.addWidget(self.combine_operations_tab_widget)

        self.tab_widget.addTab(self.combine_data_tab, "Combine Data (Merge/Concatenate)")

    def _setup_merge_sub_tab(self):
        sub_tab = QWidget()
        layout = QVBoxLayout(sub_tab)

        merge_header_layout = QHBoxLayout()
        merge_header_layout.addWidget(QLabel("Join (Merge): Combines columns from the Current DataFrame and the 'Other' DataFrame based on common key(s)."), 1)
        merge_help_button = QPushButton("?")
        merge_help_button.setFixedSize(25, 25)
        merge_help_button.setToolTip("Help on Merge/Join Types")
        merge_help_button.clicked.connect(self._show_merge_help)
        merge_header_layout.addWidget(merge_help_button)
        layout.addLayout(merge_header_layout)


        keys_layout = QGridLayout()
        keys_layout.addWidget(QLabel("Current DF Key(s) (Left):"), 0, 0, Qt.AlignTop)
        self.merge_left_keys_list = self._create_checkbox_list_widget()
        keys_layout.addWidget(self.merge_left_keys_list, 0, 1) 

        keys_layout.addWidget(QLabel("'Other' DF Key(s) (Right):"), 1, 0, Qt.AlignTop)
        self.merge_right_on_list = self._create_checkbox_list_widget()
        keys_layout.addWidget(self.merge_right_on_list, 1, 1) 
        layout.addLayout(keys_layout)

        options_layout = QGridLayout()
        options_layout.addWidget(QLabel("Join Type:"), 0, 0)
        self.merge_how_combo = QComboBox()
        self.merge_how_combo.addItems(["inner", "left", "outer", "semi", "anti", "cross"])
        options_layout.addWidget(self.merge_how_combo, 0, 1)

        options_layout.addWidget(QLabel("Suffix for overlapping columns (from 'Other DF'):"), 1, 0)
        self.merge_suffix_edit = QLineEdit("_other") 
        options_layout.addWidget(self.merge_suffix_edit, 1, 1)
        layout.addLayout(options_layout)
        
        self.merge_select_cols_checkbox = QCheckBox("Select specific columns for output (advanced)")
        self.merge_select_cols_checkbox.setEnabled(False)
        layout.addWidget(self.merge_select_cols_checkbox)

        self.apply_merge_button = QPushButton("Apply Merge/Join")
        self.apply_merge_button.clicked.connect(self._handle_apply_merge)
        layout.addWidget(self.apply_merge_button, alignment=Qt.AlignCenter)
        layout.addStretch()
        self.combine_operations_tab_widget.addTab(sub_tab, "Merge / Join")

    def _setup_concat_sub_tab(self):
        self.concat_sub_tab = QWidget() 
        main_concat_layout = QVBoxLayout(self.concat_sub_tab)

        top_label_layout = QHBoxLayout()
        top_label_layout.addWidget(QLabel("Concatenate: Appends DataFrames vertically or horizontally."), 1)
        concat_help_button = QPushButton("?")
        concat_help_button.setFixedSize(25,25)
        concat_help_button.setToolTip("Help on Concatenating DataFrames")
        concat_help_button.clicked.connect(self._show_concat_help)
        top_label_layout.addWidget(concat_help_button)
        main_concat_layout.addLayout(top_label_layout)

        concat_splitter = QSplitter(Qt.Horizontal)
        main_concat_layout.addWidget(concat_splitter, 1)

        options_widget = QWidget()
        options_layout_v = QVBoxLayout(options_widget)
        
        options_group = QGroupBox("Concatenation Options")
        options_grid_layout = QGridLayout(options_group)

        options_grid_layout.addWidget(QLabel("Concatenation Type (Polars 'how'):"), 0, 0)
        self.concat_type_combo = QComboBox()
        self.concat_type_combo.addItem("Vertical Relaxed (Stack, align common, null fill)", "vertical_relaxed")
        self.concat_type_combo.addItem("Vertical Strict (Stack, exact schema match required)", "vertical")
        self.concat_type_combo.addItem("Vertical Common Columns (Stack, only common columns kept)", "custom_vertical_common")
        self.concat_type_combo.addItem("Diagonal Relaxed (Stack, align common, null fill, supercast)", "diagonal_relaxed")
        self.concat_type_combo.addItem("Horizontal (Side-by-Side)", "horizontal") 
        self.concat_type_combo.addItem("Align (Experimental - Align by 1st col, then horizontal)", "align")
        self.concat_type_combo.addItem("Align Full (Experimental - Outer align by 1st col, then horizontal)", "align_full")
        self.concat_type_combo.addItem("Align Inner (Experimental - Inner align by 1st col, then horizontal)", "align_inner")
        self.concat_type_combo.addItem("Align Left (Experimental - Left align by 1st col, then horizontal)", "align_left")
        self.concat_type_combo.addItem("Align Right (Experimental - Right align by 1st col, then horizontal)", "align_right")
        self.concat_type_combo.currentIndexChanged.connect(self._on_concat_type_changed)
        options_grid_layout.addWidget(self.concat_type_combo, 0, 1)

        self.concat_rechunk_check = QCheckBox("Rechunk after concatenate (recommended)")
        self.concat_rechunk_check.setChecked(True)
        options_grid_layout.addWidget(self.concat_rechunk_check, 1, 0, 1, 2)
        
        options_layout_v.addWidget(options_group)
        options_layout_v.addStretch()
        concat_splitter.addWidget(options_widget)

        self.mapping_group_concat = QGroupBox("Column Mapping")
        concat_mapping_layout = QVBoxLayout(self.mapping_group_concat)
        
        self.concat_column_mapping_table = QTableWidget()
        self.concat_column_mapping_table.setColumnCount(4) 
        self.concat_column_mapping_table.setHorizontalHeaderLabels(["Include", "Current DF Column", "Maps To (Other DF)", "Output Name"])
        self.concat_column_mapping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.concat_column_mapping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.concat_column_mapping_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.concat_column_mapping_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.concat_column_mapping_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        concat_mapping_layout.addWidget(self.concat_column_mapping_table)
        
        self.refresh_mapping_button = QPushButton("Refresh / Auto-Match Column Mapping")
        self.refresh_mapping_button.clicked.connect(self._populate_concat_column_mapping_table)
        concat_mapping_layout.addWidget(self.refresh_mapping_button)
        
        concat_splitter.addWidget(self.mapping_group_concat)
        concat_splitter.setSizes([int(self.width() * 0.35), int(self.width() * 0.65)])


        self.apply_concat_button = QPushButton("Apply Concatenate")
        self.apply_concat_button.clicked.connect(self._handle_apply_concat)
        main_concat_layout.addWidget(self.apply_concat_button, 0, Qt.AlignCenter)
        
        self.combine_operations_tab_widget.addTab(self.concat_sub_tab, "Concatenate")
        self._on_concat_type_changed() 

    def _on_concat_type_changed(self, index=None): 
        how_str = self.concat_type_combo.currentData()
        if how_str is None: 
            current_idx = self.concat_type_combo.currentIndex()
            if current_idx >=0: how_str = self.concat_type_combo.itemData(current_idx)
        
        # Always keep the mapping group visible, but control interactivity
        self.mapping_group_concat.setVisible(True) # Always visible

        is_interactive_mapping_mode = (how_str == "horizontal") 
        is_info_mapping_mode = (how_str == "custom_vertical_common")

        if self.current_df is not None and self.other_df_for_combine is not None:
            self._populate_concat_column_mapping_table() # Populate first
            for row in range(self.concat_column_mapping_table.rowCount()):
                chk_widget = self.concat_column_mapping_table.cellWidget(row, 0)
                if chk_widget: 
                    checkbox = chk_widget.findChild(QCheckBox)
                    if checkbox: checkbox.setEnabled(is_interactive_mapping_mode) # Only editable for horizontal

                combo_widget = self.concat_column_mapping_table.cellWidget(row, 2)
                if combo_widget: 
                    combo_widget.setEnabled(is_interactive_mapping_mode)
                
                edit_widget = self.concat_column_mapping_table.cellWidget(row, 3)
                if edit_widget: 
                    edit_widget.setEnabled(is_interactive_mapping_mode) # Output name mainly for horizontal
        else:
            self.concat_column_mapping_table.setRowCount(0)


    def _populate_concat_column_mapping_table(self):
        self.concat_column_mapping_table.setRowCount(0)
        if self.current_df is None or self.current_df.is_empty() or \
           self.other_df_for_combine is None or self.other_df_for_combine.is_empty():
            return

        current_cols = self.current_df.columns
        other_cols = self.other_df_for_combine.columns
        
        initial_matches_dict = bij_match(current_cols, other_cols, ignore_not_found=False)

        self.concat_column_mapping_table.setRowCount(len(current_cols))

        for row, current_col_name in enumerate(current_cols):
            chk_include_widget = QWidget() 
            chk_include_layout = QHBoxLayout(chk_include_widget)
            chk_include = QCheckBox()
            chk_include.setChecked(True) 
            chk_include_layout.addWidget(chk_include)
            chk_include_layout.setAlignment(Qt.AlignCenter)
            chk_include_layout.setContentsMargins(0,0,0,0)
            self.concat_column_mapping_table.setCellWidget(row, 0, chk_include_widget)

            item_current_col = QTableWidgetItem(current_col_name)
            item_current_col.setFlags(item_current_col.flags() & ~Qt.ItemIsEditable)
            self.concat_column_mapping_table.setItem(row, 1, item_current_col)

            combo_maps_to = QComboBox(self.concat_column_mapping_table)
            combo_maps_to.addItem("<None>") 
            combo_maps_to.addItems(other_cols)
            
            matched_other_col = initial_matches_dict.get(current_col_name)
            if matched_other_col and matched_other_col in other_cols:
                idx = combo_maps_to.findText(matched_other_col)
                if idx !=-1: combo_maps_to.setCurrentIndex(idx)
                else: combo_maps_to.setCurrentIndex(0) 
            else:
                combo_maps_to.setCurrentIndex(0) 
            
            self.concat_column_mapping_table.setCellWidget(row, 2, combo_maps_to)

            output_name_edit = QLineEdit(self.concat_column_mapping_table)
            output_name_edit.setText(current_col_name) 
            output_name_edit.setPlaceholderText(f"Default: '{current_col_name}'")
            self.concat_column_mapping_table.setCellWidget(row, 3, output_name_edit)
            
        self.concat_column_mapping_table.resizeColumnToContents(0)


    def _get_concat_column_mapping_from_table(self):
        mapping_config = []
        for row in range(self.concat_column_mapping_table.rowCount()):
            include_widget = self.concat_column_mapping_table.cellWidget(row, 0)
            include_current = include_widget.findChild(QCheckBox).isChecked() if include_widget else False

            current_df_col_item = self.concat_column_mapping_table.item(row, 1)
            maps_to_combo = self.concat_column_mapping_table.cellWidget(row, 2)
            output_name_edit = self.concat_column_mapping_table.cellWidget(row, 3)

            if not all([current_df_col_item, maps_to_combo, output_name_edit]):
                continue 

            current_col = current_df_col_item.text()
            other_col_mapped = maps_to_combo.currentText()
            if other_col_mapped == "<None>":
                other_col_mapped = None
            
            output_name = output_name_edit.text().strip()
            if not output_name: 
                output_name = current_col 

            mapping_config.append({
                'include_current': include_current,
                'current_col': current_col,
                'map_to_other': other_col_mapped,
                'output_name': output_name
            })
        return mapping_config

    def receive_dataframe(self, df, df_name_hint, log_file_path_from_source=None):
        if df is None:
            self._show_error_message("Data Error", "Received empty DataFrame.")
            return

        self.current_df = df.clone()
        self.original_df_for_reset = df.clone()
        self.df_name_hint = df_name_hint or "Untitled Data"
        
        if log_file_path_from_source:
            self.current_log_file_path = log_file_path_from_source
            logger.set_log_file(self.current_log_file_path, "DataTransformer (Continuing Session)", associated_data_file=self.df_name_hint)
        
        self.setWindowTitle(f"Data Transformer - [{os.path.basename(self.df_name_hint)}]")
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.applied_expressions_history.clear() # Clear for new primary DF
        self._update_ui_for_data_state()
        self.update_preview_table()
        self.update_df_info_label()
        self._update_all_column_lists()
        self._update_applied_expressions_log() 
        
        logger.log_action("DataTransformer", "Primary Data Loaded/Received",
                          f"Data '{os.path.basename(self.df_name_hint)}' set as current.",
                          details={"Source": "Source App" if self.source_app else "Direct Load/Receive",
                                   "Shape": self.current_df.shape,
                                   "Log File": self.current_log_file_path})


    def update_preview_table(self):
        if self.current_df is not None and not self.current_df.is_empty():
            df_for_preview = self.current_df.slice(0, self.MAX_PREVIEW_ROWS)
            model = PolarsModel(df_for_preview) 
            self.preview_table.setModel(model)
        else:
            self.preview_table.setModel(None)
        self.preview_table.resizeColumnsToContents()


    def update_df_info_label(self):
        if self.current_df is not None:
            dims = f"{self.current_df.height}R x {self.current_df.width}C"
            self.df_info_label.setText(f"Current Data: {os.path.basename(self.df_name_hint)} | Dimensions: {dims}")
        else:
            self.df_info_label.setText(f"Current Data: {os.path.basename(self.df_name_hint)} | Dimensions: N/A")
            
        if self.other_df_for_combine is not None:
            dims_other = f"{self.other_df_for_combine.height}R x {self.other_df_for_combine.width}C"
            self.other_df_info_label.setText(f"Other DF: {os.path.basename(self.other_df_name_hint)} | {dims_other}")
        else:
            self.other_df_info_label.setText("Other DF: Not loaded")

    def _update_ui_for_data_state(self):
        has_current_data = self.current_df is not None and not self.current_df.is_empty()
        has_other_data = self.other_df_for_combine is not None and not self.other_df_for_combine.is_empty()

        self.save_action.setEnabled(has_current_data)
        self.describe_df_action.setEnabled(has_current_data)
        self.reset_action.setEnabled(has_current_data and len(self.undo_stack) > 0)
        self.send_to_source_action.setEnabled(has_current_data and self.source_app is not None)
        
        self.undo_action.setEnabled(len(self.undo_stack) > 0)
        self.redo_action.setEnabled(len(self.redo_stack) > 0)

        self.apply_expression_button.setEnabled(has_current_data)
        if hasattr(self, 'delete_fields_button'): self.delete_fields_button.setEnabled(has_current_data)
        self.apply_pivot_button.setEnabled(has_current_data)
        self.apply_melt_button.setEnabled(has_current_data)
        
        self.apply_merge_button.setEnabled(has_current_data and has_other_data)
        self.apply_concat_button.setEnabled(has_current_data and has_other_data)
        
        self.other_df_cast_selected_button.setEnabled(has_other_data)
        self.other_df_suggest_types_button.setEnabled(has_other_data)
        
        if hasattr(self, 'refresh_mapping_button'): 
            self.refresh_mapping_button.setEnabled(has_current_data and has_other_data)


        if not has_current_data:
            self._clear_checkbox_list_widget(self.available_cols_list_create)
            self._clear_checkbox_list_widget(self.pivot_index_cols_list)
            if hasattr(self, 'pivot_columns_col_combo'): self.pivot_columns_col_combo.clear()
            if hasattr(self, 'pivot_values_col_combo'): self.pivot_values_col_combo.clear()
            self._clear_checkbox_list_widget(self.melt_id_vars_list)
            self._clear_checkbox_list_widget(self.melt_value_vars_list)
            self._clear_checkbox_list_widget(self.merge_left_keys_list)
            if hasattr(self, 'concat_column_mapping_table'): self.concat_column_mapping_table.setRowCount(0)
        
        if not has_other_data:
            self._clear_checkbox_list_widget(self.other_df_cols_list)
            self._clear_checkbox_list_widget(self.merge_right_on_list)
            if hasattr(self, 'concat_column_mapping_table'): self.concat_column_mapping_table.setRowCount(0)
        
        if hasattr(self, 'concat_type_combo'): 
            self._on_concat_type_changed()


    def _update_all_column_lists(self):
        self._populate_checkbox_list_widget(self.available_cols_list_create, self.current_df)
        
        self._populate_checkbox_list_widget(self.pivot_index_cols_list, self.current_df)
        self._populate_qcombobox(self.pivot_columns_col_combo, self.current_df, include_none=False) 
        self._populate_qcombobox(self.pivot_values_col_combo, self.current_df, include_none=False) 
        
        self._populate_checkbox_list_widget(self.melt_id_vars_list, self.current_df)
        self._populate_checkbox_list_widget(self.melt_value_vars_list, self.current_df)

        self._populate_checkbox_list_widget(self.merge_left_keys_list, self.current_df)
        
        self._populate_checkbox_list_widget(self.other_df_cols_list, self.other_df_for_combine)
        self._populate_checkbox_list_widget(self.merge_right_on_list, self.other_df_for_combine)
        
        if hasattr(self, 'mapping_group_concat') and self.mapping_group_concat.isVisible(): 
             self._populate_concat_column_mapping_table()

    def _create_checkbox_list_widget(self):
        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.NoSelection) 
        return list_widget

    def _populate_checkbox_list_widget(self, list_widget: QListWidget, df: pl.DataFrame = None, check_all=False):
        if not list_widget: return # Guard if called before UI fully ready
        list_widget.clear()
        if df is not None and not df.is_empty():
            for col_name, dtype in df.schema.items():
                item = QListWidgetItem(list_widget) # Create item first
                checkbox = QCheckBox(f"{col_name} ({dtype})")
                checkbox.setChecked(check_all)
                # list_widget.addItem(item) # This might be problematic if item itself isn't directly added
                list_widget.setItemWidget(item, checkbox) # Set the widget for the item
    
    def _clear_checkbox_list_widget(self, list_widget: QListWidget):
        if list_widget: 
            list_widget.clear()


    def _get_checked_items_from_checkbox_list(self, list_widget: QListWidget, get_name_only=True):
        checked_items = []
        if not list_widget : return checked_items # Guard
        for i in range(list_widget.count()):
            item = list_widget.item(i) # Get the QListWidgetItem
            widget = list_widget.itemWidget(item) # Get the widget (QCheckBox) associated with the item
            if isinstance(widget, QCheckBox) and widget.isChecked():
                text = widget.text()
                if get_name_only:
                    checked_items.append(text.split(" (")[0])
                else:
                    checked_items.append(text)
        return checked_items


    def _populate_qcombobox(self, combo_box: QComboBox, df: pl.DataFrame = None, include_none=True):
        current_text = combo_box.currentText() 
        
        combo_box.blockSignals(True)
        combo_box.clear()
        items = []
        if include_none:
            items.append("None")
        if df is not None and not df.is_empty():
            items.extend(df.columns)
        combo_box.addItems(items)
        
        idx = combo_box.findText(current_text)
        if idx != -1:
            combo_box.setCurrentIndex(idx)
        elif not include_none and combo_box.count() > 0 : 
             combo_box.setCurrentIndex(0)
        elif include_none and "None" in items: 
             combo_box.setCurrentIndex(combo_box.findText("None"))
        elif combo_box.count() > 0: 
            combo_box.setCurrentIndex(0)

        combo_box.blockSignals(False)


    def _get_selected_items_from_qlistwidget(self, list_widget: QListWidget, get_name_only=True):
        # This is now primarily for QListWidgets that DON'T use checkboxes for selection
        # The checkbox lists will use _get_checked_items_from_checkbox_list
        selected = []
        if not list_widget : return selected # Guard
        for item in list_widget.selectedItems():
            text = item.text()
            if get_name_only:
                selected.append(text.split(" (")[0]) 
            else:
                selected.append(text)
        return selected

    def _handle_open_file_directly(self):
        initial_path = os.path.join(os.path.expanduser('~'), 'Documents')
        file_dialog_filter = (
            "All Supported Data Files (*.csv *.xlsx *.xls *.xlsb *.xlsm *.parquet *.npz *.pkl *.pickle *.hkl *.hickle *.json *.jsonl *.ndjson *.dat *.txt);;"
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls *.xlsb *.xlsm);;Parquet Files (*.parquet);;"
            "NumPy Archives (*.npz *.npy);;Pickle Files (*.pkl *.pickle);;"
            "Hickle Files (*.hkl *.hickle);;JSON Files (*.json *.jsonl *.ndjson);;"
            "Text/DAT Files (*.dat *.txt);;All Files (*)"
        )
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Primary Data File", initial_path, file_dialog_filter)

        if file_name:
            self.df_name_hint = file_name 
            base_name = re.sub(r'\W+', '_', os.path.splitext(os.path.basename(self.df_name_hint))[0])
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_log_file_path = os.path.join(os.path.dirname(file_name), f".__log__{base_name}_{timestamp_str}_dt.md")
            logger.set_log_file(self.current_log_file_path, "DataTransformer", associated_data_file=self.df_name_hint)
            
            sheet_name = None
            if file_name.lower().endswith(('.xlsx', '.xlsm', '.xlsb', '.xls')):
                try:
                    from main import get_sheet_names 
                    sheet_names_list = get_sheet_names(file_name)
                    if len(sheet_names_list) > 1:
                        sheet_name_selected, ok = QInputDialog.getItem(self, "Select Sheet", "Choose a sheet:", sheet_names_list, 0, False)
                        if not ok: return
                        sheet_name = sheet_name_selected
                    elif sheet_names_list: sheet_name = sheet_names_list[0]
                except Exception as e:
                    self._show_error_message("Sheet Selection Error", f"Could not get sheet names: {e}")
                    return
            
            self.status_bar.showMessage(f"Loading primary data: {os.path.basename(file_name)}...")
            if hasattr(self, 'primary_file_loader_thread'):
                try:
                    self.primary_file_loader_thread.finished_signal.disconnect(self._handle_primary_df_loaded)
                    self.primary_file_loader_thread.error_signal.disconnect(self._handle_primary_df_load_error)
                except RuntimeError: pass 
            
            self.primary_file_loader_thread = AsyncFileLoaderThreadDT(file_name, sheet_name, parent_gui=self)
            self.primary_file_loader_thread.finished_signal.connect(self._handle_primary_df_loaded)
            self.primary_file_loader_thread.error_signal.connect(self._handle_primary_df_load_error)
            self.primary_file_loader_thread.request_npz_array_selection.connect(self._handle_prompt_npz_array_selection_primary)
            self.primary_file_loader_thread.request_pickle_item_selection.connect(self._handle_prompt_pickle_item_selection_primary)
            self.primary_file_loader_thread.start()


    def _handle_primary_df_loaded(self, df, file_info):
        self.status_bar.showMessage(f"Primary data '{os.path.basename(file_info.get('filename', self.df_name_hint))}' loaded.", 3000)
        if df is None or df.is_empty():
            self._show_error_message("Load Error", f"Loaded file '{os.path.basename(file_info.get('filename', 'N/A'))}' is empty or unreadable.")
            logger.log_action("DataTransformer", "File Load Error (Primary)", f"Failed to load or file empty: {file_info.get('filename', 'N/A')}", details=file_info)
            return

        self.receive_dataframe(df, file_info.get('filename', self.df_name_hint)) 
        logger.log_action("DataTransformer", "File Load Success (Primary)",
                          f"Details for loaded primary file: {os.path.basename(file_info.get('filename', 'N/A'))}",
                          details=file_info)
        if hasattr(self, 'primary_file_loader_thread'):
            self.primary_file_loader_thread.quit()
            self.primary_file_loader_thread.wait()


    def _handle_primary_df_load_error(self, error_message):
        self.status_bar.showMessage(f"Error loading primary data: {error_message}", 5000)
        self._show_error_message("Primary File Load Error", error_message)
        logger.log_action("DataTransformer", "File Load Error (Primary)", "Failed to load primary data file.", details={"Error": error_message})
        if hasattr(self, 'primary_file_loader_thread'):
            self.primary_file_loader_thread.quit()
            self.primary_file_loader_thread.wait()


    def _handle_save_file(self):
        if self.current_df is None:
            self._show_error_message("Save Error", "No data to save.")
            return
        
        original_name = os.path.splitext(os.path.basename(self.df_name_hint if self.df_name_hint else "transformed_data"))[0]
        suggested_name = f"{original_name}_transformed.csv"
        
        file_name_to_save, selected_filter = QFileDialog.getSaveFileName(self, "Save Transformed Data As", suggested_name, 
                                                                         "CSV Files (*.csv);;Excel Files (*.xlsx);;Parquet Files (*.parquet)")
        if file_name_to_save:
            try:
                df_to_save = self.current_df.clone()
                if "__original_index__" in df_to_save.columns:
                    df_to_save = df_to_save.drop("__original_index__")

                self.status_bar.showMessage(f"Saving to {os.path.basename(file_name_to_save)}...", 3000)
                QApplication.processEvents()

                if file_name_to_save.lower().endswith('.csv'):
                    df_to_save.write_csv(file_name_to_save)
                elif file_name_to_save.lower().endswith('.xlsx'):
                    df_to_save.write_excel(file_name_to_save)
                elif file_name_to_save.lower().endswith('.parquet'):
                    df_to_save.write_parquet(file_name_to_save)
                else:
                    self._show_error_message("Save Error", "Unsupported file extension. Please choose .csv, .xlsx, or .parquet.")
                    return

                self.status_bar.showMessage(f"Data saved successfully to {os.path.basename(file_name_to_save)}.", 5000)
                logger.log_dataframe_save("DataTransformer", file_name_to_save,
                                          rows=df_to_save.height, cols=df_to_save.width,
                                          source_data_name=os.path.basename(self.df_name_hint))
            except Exception as e:
                self._show_error_message("Save Error", f"Could not save file: {e}")
                logger.log_action("DataTransformer", "File Save Error", f"Error saving to {os.path.basename(file_name_to_save)}", details={"Error": str(e)})

    def _handle_describe_df(self):
        if self.current_df is None or self.current_df.is_empty():
            self._show_error_message("Describe Error", "No data loaded to describe.")
            return
        try:
            description_df = self.current_df.describe()
            
            html_output = "<h3>DataFrame Description:</h3>"
            try:
                # Try to use to_html first
                html_output += description_df.to_html(notebook=False, max_rows=-1)
            except AttributeError: # Fallback for older Polars or if to_html fails
                html_output += f"<pre>{str(description_df)}</pre>"
            except Exception as e_html: # Catch other potential errors with to_html
                html_output += f"<p>Error generating HTML table for describe: {e_html}</p><pre>{str(description_df)}</pre>"


            desc_dialog = QDialog(self)
            desc_dialog.setWindowTitle(f"Description of '{os.path.basename(self.df_name_hint)}'")
            desc_dialog.setMinimumSize(700, 500) # Increased size
            layout = QVBoxLayout(desc_dialog)
            
            text_browser = QTextBrowser()
            text_browser.setHtml(html_output)
            layout.addWidget(text_browser)
            
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(desc_dialog.accept)
            layout.addWidget(button_box)
            desc_dialog.exec()
            logger.log_action("DataTransformer", "DataFrame Described", f"Displayed description for current data.")
        except Exception as e:
            self._show_error_message("Describe Error", f"Could not describe DataFrame: {e}")


    def _push_to_undo_stack(self):
        if self.current_df is not None:
            self.undo_stack.append(self.current_df.clone())
            self.redo_stack.clear()
            self._update_ui_for_data_state()

    def _handle_undo(self):
        if self.undo_stack:
            df_shape_before_undo = self.current_df.shape if self.current_df is not None else None
            self.redo_stack.append(self.current_df.clone())
            self.current_df = self.undo_stack.pop()
            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists()
            self._update_ui_for_data_state()
            # After undo, the expression history might be out of sync.
            # For simplicity, we'll clear it. A more complex system would track expressions per undo state.
            self.applied_expressions_history.clear() 
            self._update_applied_expressions_log()
            self.status_bar.showMessage("Undo successful.", 2000)
            logger.log_action("DataTransformer", "Undo Operation", "Reverted to previous data state.",
                              df_shape_before=df_shape_before_undo, df_shape_after=self.current_df.shape)


    def _handle_redo(self):
        if self.redo_stack:
            df_shape_before_redo = self.current_df.shape if self.current_df is not None else None
            self.undo_stack.append(self.current_df.clone())
            self.current_df = self.redo_stack.pop()
            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists()
            self._update_ui_for_data_state()
            # Redoing also means expression history might change. For simplicity, clear.
            self.applied_expressions_history.clear() 
            self._update_applied_expressions_log() # This will be empty. A more robust way is needed.
            self.status_bar.showMessage("Redo successful.", 2000)
            logger.log_action("DataTransformer", "Redo Operation", "Re-applied next data state.",
                              df_shape_before=df_shape_before_redo, df_shape_after=self.current_df.shape)

    def _handle_reset_transformations(self):
        if self.original_df_for_reset is not None:
            if self._confirm_action("Reset Transformations", "Are you sure you want to discard all changes and revert to the original data?"):
                self._push_to_undo_stack() 
                df_shape_before_reset = self.current_df.shape
                self.current_df = self.original_df_for_reset.clone()
                self.applied_expressions_history.clear()
                self._update_applied_expressions_log()
                self.update_preview_table()
                self.update_df_info_label()
                self._update_all_column_lists()
                self._update_ui_for_data_state()
                self.status_bar.showMessage("All transformations reset.", 2000)
                logger.log_action("DataTransformer", "Reset Transformations", "All changes reverted to original loaded data.",
                                  df_shape_before=df_shape_before_reset, df_shape_after=self.current_df.shape)


    def _handle_send_to_source(self):
        if self.source_app and hasattr(self.source_app, "load_dataframe_from_source") and self.current_df is not None:
            new_name_hint = f"{os.path.splitext(os.path.basename(self.df_name_hint))[0]}_transformed"
            
            try:
                self.source_app.load_dataframe_from_source(self.current_df.clone(), new_name_hint, self.current_log_file_path)
                self.status_bar.showMessage(f"Data sent to {self.source_app.windowTitle()}.", 3000)
                logger.log_action("DataTransformer", "Data Sent to Source",
                                  f"Transformed data sent to '{self.source_app.windowTitle()}'.",
                                  details={"Source App": self.source_app.windowTitle(),
                                           "Data Hint": new_name_hint, "Shape": self.current_df.shape,
                                           "Log File Passed": self.current_log_file_path})
            except AttributeError:
                 if hasattr(self.source_app, "receive_dataframe"): 
                    self.source_app.receive_dataframe(self.current_df.clone(), new_name_hint, log_file_path_from_source=self.current_log_file_path)
                    self.status_bar.showMessage(f"Data sent to {self.source_app.windowTitle()}.", 3000)
                    logger.log_action("DataTransformer", "Data Sent to Source",
                                  f"Transformed data sent to '{self.source_app.windowTitle()}'. (using receive_dataframe)",
                                  details={"Source App": self.source_app.windowTitle(),
                                           "Data Hint": new_name_hint, "Shape": self.current_df.shape,
                                           "Log File Passed": self.current_log_file_path})
                 else:
                    self._show_error_message("Send Error", "Source application does not support receiving data or method name mismatch.")
            except Exception as e:
                self._show_error_message("Send Error", f"Failed to send data: {e}")
                logger.log_action("DataTransformer", "Data Send Error", "Failed to send data to source.", details={"Error": str(e)})
        elif not self.source_app:
            self._show_error_message("Send Error", "No source application to send data to.")
        elif self.current_df is None:
            self._show_error_message("Send Error", "No data to send.")


    def _handle_apply_expression(self):
        if self.current_df is None: return
        field_name = self.new_field_name_edit.text().strip()
        expression_str = self.polars_expression_edit.toPlainText().strip()

        if not field_name or not expression_str:
            self._show_error_message("Input Error", "Field name and expression cannot be empty.")
            return
        
        if field_name in self.current_df.columns:
            if not self._confirm_action("Overwrite Column?", f"Column '{field_name}' already exists. Overwrite it?"):
                return

        df_shape_before = self.current_df.shape
        self._push_to_undo_stack()
        try:
            eval_context = {"pl": pl}
            eval_context.update({col: pl.col(col) for col in self.current_df.columns})
            
            polars_expr_obj = eval(expression_str, eval_context)
            
            self.current_df = self.current_df.with_columns(
                polars_expr_obj.alias(field_name)
            )
            
            self.applied_expressions_history[field_name] = expression_str 
            self._update_applied_expressions_log()


            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists()
            self.status_bar.showMessage(f"Expression applied to field '{field_name}'.", 2000)
            logger.log_action("DataTransformer", "Create/Modify Field",
                              f"Expression applied to '{field_name}'.",
                              details={"Field": field_name, "Expression": expression_str},
                              df_shape_before=df_shape_before, df_shape_after=self.current_df.shape)
        except Exception as e:
            self._show_error_message("Expression Error", f"Error applying expression: {e}\n\n{traceback.format_exc()}")
            if self.undo_stack: 
                 self.current_df = self.undo_stack.pop() 
                 self.redo_stack.clear() 
                 self._update_ui_for_data_state()
            logger.log_action("DataTransformer", "Create/Modify Field Error",
                              f"Error for field '{field_name}'.",
                              details={"Field": field_name, "Expression": expression_str, "Error": str(e)})

    def _handle_delete_fields(self):
        if self.current_df is None or self.current_df.is_empty():
            self._show_error_message("Delete Error", "No data loaded to delete columns from.")
            return
        
        selected_cols_to_delete = self._get_checked_items_from_checkbox_list(self.available_cols_list_create, get_name_only=True)

        if not selected_cols_to_delete:
            self._show_error_message("Delete Error", "No columns selected for deletion.")
            return

        if not self._confirm_action("Confirm Deletion", f"Are you sure you want to permanently delete the selected column(s): {', '.join(selected_cols_to_delete)}?"):
            return

        df_shape_before = self.current_df.shape
        self._push_to_undo_stack()
        try:
            self.current_df = self.current_df.drop(selected_cols_to_delete)
            
            for col_name in selected_cols_to_delete:
                if col_name in self.applied_expressions_history:
                    del self.applied_expressions_history[col_name]
            self._update_applied_expressions_log()


            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists()
            self.status_bar.showMessage(f"Columns deleted: {', '.join(selected_cols_to_delete)}.", 2000)
            logger.log_action("DataTransformer", "Delete Fields",
                              f"Deleted columns: {', '.join(selected_cols_to_delete)}.",
                              details={"Columns Deleted": selected_cols_to_delete},
                              df_shape_before=df_shape_before, df_shape_after=self.current_df.shape)
        except Exception as e:
            self._show_error_message("Delete Error", f"Error deleting columns: {e}\n\n{traceback.format_exc()}")
            if self.undo_stack: self.current_df = self.undo_stack.pop(); self.redo_stack.clear(); self._update_ui_for_data_state()
            logger.log_action("DataTransformer", "Delete Fields Error", "Error during column deletion.", details={"Error": str(e)})

    def _update_applied_expressions_log(self):
        self.applied_expressions_log_table.setRowCount(0) 
        
        # To show latest at bottom, iterate through history keys
        # (Python dicts maintain insertion order from 3.7+)
        for field_name, expression in self.applied_expressions_history.items():
            row_position = self.applied_expressions_log_table.rowCount()
            self.applied_expressions_log_table.insertRow(row_position)
            
            self.applied_expressions_log_table.setItem(row_position, 0, QTableWidgetItem(field_name))
            self.applied_expressions_log_table.setItem(row_position, 1, QTableWidgetItem(expression))
            
            use_btn = QPushButton("Use")
            use_btn.setFixedSize(40,22) # Small button
            use_btn.clicked.connect(lambda checked=False, fn=field_name, expr=expression: self._populate_expression_fields_from_log(fn, expr))
            self.applied_expressions_log_table.setCellWidget(row_position, 2, use_btn)
        self.applied_expressions_log_table.scrollToBottom()


    def _populate_expression_fields_from_log(self, field_name, expression):
        self.new_field_name_edit.setText(field_name)
        self.polars_expression_edit.setPlainText(expression)


    def _handle_apply_pivot(self):
        if self.current_df is None: return
        
        index_cols = self._get_checked_items_from_checkbox_list(self.pivot_index_cols_list)
        columns_col = self.pivot_columns_col_combo.currentText()
        values_col = self.pivot_values_col_combo.currentText()
        agg_func_str = self.pivot_agg_func_combo.currentText()

        if not index_cols or columns_col == "None" or values_col == "None":
            self._show_error_message("Pivot Error", "Index, Columns, and Values columns must be selected.")
            return
        if columns_col in index_cols or values_col in index_cols or columns_col == values_col :
             self._show_error_message("Pivot Error", "Index, Columns, and Values columns must be distinct.")
             return

        df_shape_before = self.current_df.shape
        self._push_to_undo_stack()
        try:
            temp_df = self.current_df 
            if agg_func_str not in ["first", "list", "count"]:
                try:
                    if temp_df.schema[values_col] not in [pl.Int64, pl.Float64, pl.Int32, pl.Float32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8]:
                        temp_df = temp_df.with_columns(pl.col(values_col).cast(pl.Float64, strict=False))
                except Exception as cast_e:
                    self.status_bar.showMessage(f"Warning: Could not cast values column '{values_col}' to numeric for aggregation: {cast_e}", 4000)
            
            agg_func_polars = agg_func_str 

            self.current_df = temp_df.pivot( 
                index=index_cols,
                columns=columns_col,
                values=values_col,
                aggregate_function=agg_func_polars
            )
            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists()
            self.status_bar.showMessage("Pivot applied.", 2000)
            logger.log_action("DataTransformer", "Pivot Operation", "Data pivoted.",
                              details={"Index": index_cols, "Columns From": columns_col, "Values From": values_col, "Aggregation": agg_func_str},
                              df_shape_before=df_shape_before, df_shape_after=self.current_df.shape)
        except Exception as e:
            self._show_error_message("Pivot Error", f"Error applying pivot: {e}\n\n{traceback.format_exc()}")
            if self.undo_stack: self.current_df = self.undo_stack.pop(); self.redo_stack.clear(); self._update_ui_for_data_state()
            logger.log_action("DataTransformer", "Pivot Operation Error", "Error during pivot.", details={"Error": str(e)})


    def _handle_apply_melt(self):
        if self.current_df is None: return

        id_vars = self._get_checked_items_from_checkbox_list(self.melt_id_vars_list)
        value_vars = self._get_checked_items_from_checkbox_list(self.melt_value_vars_list)
        variable_name = self.melt_variable_name_edit.text().strip() or "variable"
        value_name = self.melt_value_name_edit.text().strip() or "value"

        if not id_vars and not value_vars : 
            if not self._confirm_action("Melt Warning", "No ID variables and no Value variables selected. This will melt ALL columns. Are you sure?"):
                 return
        elif not id_vars and value_vars: 
            pass 
        elif not id_vars: 
            pass
        
        df_shape_before = self.current_df.shape
        self._push_to_undo_stack()
        try:
            self.current_df = self.current_df.melt(
                id_vars=id_vars if id_vars else None, 
                value_vars=value_vars if value_vars else None, 
                variable_name=variable_name,
                value_name=value_name
            )
            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists()
            self.status_bar.showMessage("Melt applied.", 2000)
            logger.log_action("DataTransformer", "Melt Operation", "Data melted (unpivoted).",
                              details={"ID Vars": id_vars, "Value Vars": value_vars if value_vars else "All non-ID",
                                       "Variable Name": variable_name, "Value Name": value_name},
                              df_shape_before=df_shape_before, df_shape_after=self.current_df.shape)
        except Exception as e:
            self._show_error_message("Melt Error", f"Error applying melt: {e}\n\n{traceback.format_exc()}")
            if self.undo_stack: self.current_df = self.undo_stack.pop(); self.redo_stack.clear(); self._update_ui_for_data_state()
            logger.log_action("DataTransformer", "Melt Operation Error", "Error during melt.", details={"Error": str(e)})

    def _handle_load_other_df(self):
        initial_path = os.path.join(os.path.expanduser('~'), 'Documents')
        file_dialog_filter = ( 
            "All Supported Data Files (*.csv *.xlsx *.xls *.xlsb *.xlsm *.parquet *.npz *.pkl *.pickle *.hkl *.hickle *.json *.jsonl *.ndjson *.dat *.txt);;"
            "CSV Files (*.csv);;Excel Files (*.xlsx *.xls *.xlsb *.xlsm);;Parquet Files (*.parquet);;"
            "NumPy Archives (*.npz *.npy);;Pickle Files (*.pkl *.pickle);;"
            "Hickle Files (*.hkl *.hickle);;JSON Files (*.json *.jsonl *.ndjson);;"
            "Text/DAT Files (*.dat *.txt);;All Files (*)"
        )
        file_name, _ = QFileDialog.getOpenFileName(self, "Load 'Other' DataFrame", initial_path, file_dialog_filter)

        if file_name:
            self.other_df_name_hint = file_name 
            sheet_name = None 
            if file_name.lower().endswith(('.xlsx', '.xlsm', '.xlsb', '.xls')):
                try:
                    from main import get_sheet_names
                    sheet_names_list = get_sheet_names(file_name)
                    if len(sheet_names_list) > 1:
                        sheet_name_selected, ok = QInputDialog.getItem(self, "Select Sheet for 'Other' DF", "Choose a sheet:", sheet_names_list, 0, False)
                        if not ok: return
                        sheet_name = sheet_name_selected
                    elif sheet_names_list: sheet_name = sheet_names_list[0]
                except Exception as e:
                    self._show_error_message("Sheet Selection Error", f"Could not get sheet names for 'Other' DF: {e}")
                    return

            self.other_df_progress_bar.setVisible(True)
            self.other_df_progress_bar.setValue(0) 
            self.status_bar.showMessage(f"Loading 'Other' DF: {os.path.basename(file_name)}...")

            if hasattr(self, 'other_df_loader_thread'):
                try:
                    self.other_df_loader_thread.finished_signal.disconnect(self._handle_other_df_loaded)
                    self.other_df_loader_thread.error_signal.disconnect(self._handle_other_df_load_error)
                except RuntimeError: pass
            
            self.other_df_loader_thread = AsyncFileLoaderThreadDT(file_name, sheet_name, parent_gui=self)
            self.other_df_loader_thread.finished_signal.connect(self._handle_other_df_loaded)
            self.other_df_loader_thread.error_signal.connect(self._handle_other_df_load_error)
            self.other_df_loader_thread.request_npz_array_selection.connect(self._handle_prompt_npz_array_selection_other)
            self.other_df_loader_thread.request_pickle_item_selection.connect(self._handle_prompt_pickle_item_selection_other)
            self.other_df_loader_thread.start()

    def _handle_other_df_loaded(self, df, file_info):
        self.other_df_progress_bar.setValue(100)
        QTimer.singleShot(500, lambda: self.other_df_progress_bar.setVisible(False))
        self.status_bar.showMessage(f"'Other' DF '{os.path.basename(file_info.get('filename', 'N/A'))}' loaded.", 3000)

        if df is None or df.is_empty():
            self._show_error_message("Load Error", f"Loaded 'Other' DF '{os.path.basename(file_info.get('filename', 'N/A'))}' is empty or unreadable.")
            self.other_df_for_combine = None
            self.other_df_file_info = {}
            logger.log_action("DataTransformer", "File Load Error (Other DF)", f"Failed to load or file empty: {file_info.get('filename', 'N/A')}", details=file_info)
        else:
            self.other_df_for_combine = df
            self.other_df_file_info = file_info 
            logger.log_dataframe_load("DataTransformer (Other DF)", file_info.get('filename'), 
                                      sheet_name=file_info.get('sheet_name'),
                                      rows=file_info.get('rows', df.height), 
                                      cols=file_info.get('columns', df.width),
                                      load_time_sec=file_info.get('load_time', 0))
            
        self.update_df_info_label()
        self._populate_checkbox_list_widget(self.other_df_cols_list, self.other_df_for_combine) 
        self._populate_checkbox_list_widget(self.merge_right_on_list, self.other_df_for_combine) 
        self._update_ui_for_data_state() 
        if hasattr(self, 'other_df_loader_thread'):
            self.other_df_loader_thread.quit()
            self.other_df_loader_thread.wait()

    def _handle_other_df_load_error(self, error_message):
        self.other_df_progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Error loading 'Other' DF: {error_message}", 5000)
        self._show_error_message("'Other' DF Load Error", error_message)
        self.other_df_for_combine = None
        self.other_df_file_info = {}
        self._update_ui_for_data_state()
        logger.log_action("DataTransformer", "File Load Error (Other DF)", "Failed to load 'Other' data file.", details={"Error": error_message})
        if hasattr(self, 'other_df_loader_thread'):
            self.other_df_loader_thread.quit()
            self.other_df_loader_thread.wait()

    @Slot(list, str)
    def _handle_prompt_npz_array_selection_other(self, items_info, base_filename):
        item_display_list = [f"{item['name']} (Shape: {item['shape']}, Dtype: {item['dtype']})" for item in items_info]
        item_text, ok = QInputDialog.getItem(self, f"Select Array from {base_filename} (Other DF)",
                                             "Choose an array to load:", item_display_list, 0, False)
        selected_array_name = item_text.split(" (Shape:")[0] if ok and item_text else ""
        if hasattr(self.other_df_loader_thread, 'on_npz_array_selected_relayed'):
            self.other_df_loader_thread.on_npz_array_selected_relayed(selected_array_name)

    @Slot(list, str, str)
    def _handle_prompt_pickle_item_selection_other(self, items_info, base_filename, file_type):
        item_display_list = [f"{item['name']} (Type: {item['type']}, Shape: {item.get('shape', 'N/A')})" for item in items_info]
        item_text, ok = QInputDialog.getItem(self, f"Select Item from {file_type.capitalize()} (Other DF): {base_filename}",
                                             "Choose dataset:", item_display_list, 0, False)
        selected_item_path = ""
        if ok and item_text:
            for item_detail in items_info:
                if item_text.startswith(item_detail['name']):
                    selected_item_path = item_detail['object_path']; break
        if hasattr(self.other_df_loader_thread, 'on_pickle_item_selected_relayed'):
            self.other_df_loader_thread.on_pickle_item_selected_relayed(selected_item_path)

    @Slot(list, str)
    def _handle_prompt_npz_array_selection_primary(self, items_info, base_filename):
        item_display_list = [f"{item['name']} (Shape: {item['shape']}, Dtype: {item['dtype']})" for item in items_info]
        item_text, ok = QInputDialog.getItem(self, f"Select Array from {base_filename} (Primary DF)",
                                             "Choose an array to load:", item_display_list, 0, False)
        selected_array_name = item_text.split(" (Shape:")[0] if ok and item_text else ""
        if hasattr(self.primary_file_loader_thread, 'on_npz_array_selected_relayed'):
            self.primary_file_loader_thread.on_npz_array_selected_relayed(selected_array_name)

    @Slot(list, str, str)
    def _handle_prompt_pickle_item_selection_primary(self, items_info, base_filename, file_type):
        item_display_list = [f"{item['name']} (Type: {item['type']}, Shape: {item.get('shape', 'N/A')})" for item in items_info]
        item_text, ok = QInputDialog.getItem(self, f"Select Item from {file_type.capitalize()} (Primary DF): {base_filename}",
                                             "Choose dataset:", item_display_list, 0, False)
        selected_item_path = ""
        if ok and item_text:
            for item_detail in items_info:
                if item_text.startswith(item_detail['name']):
                    selected_item_path = item_detail['object_path']; break
        if hasattr(self.primary_file_loader_thread, 'on_pickle_item_selected_relayed'):
            self.primary_file_loader_thread.on_pickle_item_selected_relayed(selected_item_path)


    def _handle_cast_other_df_columns(self):
        if self.other_df_for_combine is None or self.other_df_for_combine.is_empty(): return
        selected_col_names = self._get_checked_items_from_checkbox_list(self.other_df_cols_list, get_name_only=True)
        target_type_str = self.other_df_type_conversion_combo.currentText()

        if not selected_col_names:
            self._show_error_message("Type Cast Error", "No columns selected in 'Other DF' to cast.")
            return

        type_map = {
            "String": pl.Utf8, "Integer": pl.Int64, "Float": pl.Float64,
            "Datetime": pl.Datetime, "Categorical": pl.Categorical, "Boolean": pl.Boolean
        }
        target_polars_type = type_map.get(target_type_str)
        if not target_polars_type:
            self._show_error_message("Type Cast Error", f"Unknown target type: {target_type_str}")
            return

        df_shape_before = self.other_df_for_combine.shape
        try:
            expressions = []
            for col_name in selected_col_names:
                if target_polars_type == pl.Datetime:
                    if self.other_df_for_combine.schema[col_name] == pl.Utf8:
                        expressions.append(pl.col(col_name).str.to_datetime(strict=False, time_unit='ns', ambiguous='earliest').dt.replace_time_zone(None).alias(col_name))
                    else: 
                        expressions.append(pl.col(col_name).cast(pl.Datetime, strict=False).dt.replace_time_zone(None).alias(col_name))
                else:
                    expressions.append(pl.col(col_name).cast(target_polars_type, strict=False).alias(col_name))
            
            self.other_df_for_combine = self.other_df_for_combine.with_columns(expressions)
            self._populate_checkbox_list_widget(self.other_df_cols_list, self.other_df_for_combine) 
            self._populate_checkbox_list_widget(self.merge_right_on_list, self.other_df_for_combine)
            self.status_bar.showMessage(f"Selected columns in 'Other DF' cast to {target_type_str}.", 2000)
            logger.log_action("DataTransformer", "Type Cast (Other DF)",
                              f"Casted {len(selected_col_names)} cols to {target_type_str}.",
                              details={"Columns": selected_col_names, "Target Type": target_type_str, "DF Name": self.other_df_name_hint},
                              df_shape_before=df_shape_before, df_shape_after=self.other_df_for_combine.shape)
        except Exception as e:
            self._show_error_message("Type Cast Error", f"Error casting types for 'Other DF': {e}")
            logger.log_action("DataTransformer", "Type Cast Error (Other DF)", "Error during type casting.", details={"Error": str(e), "DF Name": self.other_df_name_hint})

    def _handle_apply_merge(self):
        if self.current_df is None or self.current_df.is_empty() or \
           self.other_df_for_combine is None or self.other_df_for_combine.is_empty():
            self._show_error_message("Merge Error", "Both Current DF and Other DF must be loaded and not empty.")
            return

        left_keys_orig = self._get_checked_items_from_checkbox_list(self.merge_left_keys_list)
        right_keys_orig = self._get_checked_items_from_checkbox_list(self.merge_right_on_list)
        how = self.merge_how_combo.currentText() 
        suffix = self.merge_suffix_edit.text() # Use the single suffix field
        
        if not left_keys_orig or not right_keys_orig:
            if how != "cross": 
                self._show_error_message("Merge Error", "Key(s) must be selected for both DataFrames (unless it's a cross join).")
                return
        elif len(left_keys_orig) != len(right_keys_orig):
            if how != "cross":
                self._show_error_message("Merge Error", "The number of keys selected for Current DF and Other DF must match.")
                return
        
        df_shape_before = self.current_df.shape
        other_df_shape = self.other_df_for_combine.shape
        self._push_to_undo_stack()

        current_df_to_join = self.current_df.clone()
        other_df_to_join = self.other_df_for_combine.clone()

        try:
            # Cast join keys to string for robustness
            left_key_exprs = [pl.col(k).cast(pl.Utf8).alias(k) for k in left_keys_orig]
            right_key_exprs = [pl.col(k).cast(pl.Utf8).alias(k) for k in right_keys_orig]

            if left_key_exprs: # Only apply if keys are actually selected
                current_df_to_join = current_df_to_join.with_columns(left_key_exprs)
            if right_key_exprs:
                other_df_to_join = other_df_to_join.with_columns(right_key_exprs)

            join_args = {
                "left_on": left_keys_orig if left_keys_orig else None,
                "right_on": right_keys_orig if right_keys_orig else None,
                "how": how,
                "suffix": suffix # Polars join uses a single suffix for columns from the right DF
            }
            
            self.current_df = current_df_to_join.join(
                other_df_to_join,
                **join_args
            )
            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists()
            self.status_bar.showMessage(f"'{how.capitalize()}' merge applied.", 2000)
            logger.log_action("DataTransformer", "Merge/Join Operation", f"Applied '{how}' join.",
                              details={"Left Keys": left_keys_orig, "Right Keys": right_keys_orig, "How": how, 
                                       "Suffix (for Other DF)": suffix, 
                                       "Other DF Shape": other_df_shape, "Other DF Name": self.other_df_name_hint},
                              df_shape_before=df_shape_before, df_shape_after=self.current_df.shape)
        except Exception as e:
            self._show_error_message("Merge Error", f"Error applying merge: {e}\n\n{traceback.format_exc()}")
            if self.undo_stack: self.current_df = self.undo_stack.pop(); self.redo_stack.clear(); self._update_ui_for_data_state()
            logger.log_action("DataTransformer", "Merge/Join Error", "Error during merge.", details={"Error": str(e)})


    def _handle_apply_concat(self):
        if self.current_df is None or self.current_df.is_empty() or \
           self.other_df_for_combine is None or self.other_df_for_combine.is_empty():
            self._show_error_message("Concatenate Error", "Both Current DF and Other DF must be loaded and not empty.")
            return

        polars_how_from_ui = self.concat_type_combo.currentData()
        rechunk = self.concat_rechunk_check.isChecked()
        
        if polars_how_from_ui is None:
            self._show_error_message("Concatenate Error", "Invalid concatenation type selected.")
            return

        current_df_to_concat = self.current_df.clone()
        other_df_to_concat = self.other_df_for_combine.clone()
        
        mapping_config = self._get_concat_column_mapping_from_table()
        processed_mapping_for_log = mapping_config # Log the user's choices
        
        actual_polars_how = polars_how_from_ui

        if polars_how_from_ui == "horizontal":
            if current_df_to_concat.height != other_df_to_concat.height:
                if not self._confirm_action("Concat Warning", 
                                            "Horizontal concatenation: Row counts differ. Polars will fill shorter DF with nulls up to the height of the taller one. Continue?"):
                    return
            
            # Prepare current_df based on "Include" and "Output Name"
            current_df_select_exprs = []
            for row_map in mapping_config:
                if row_map['include_current']:
                    if row_map['output_name'] != row_map['current_col']:
                        current_df_select_exprs.append(pl.col(row_map['current_col']).alias(row_map['output_name']))
                    else:
                        current_df_select_exprs.append(pl.col(row_map['current_col']))
            
            if current_df_select_exprs:
                current_df_to_concat = current_df_to_concat.select(current_df_select_exprs)
            elif mapping_config : # All current_df cols unchecked
                current_df_to_concat = pl.DataFrame().with_height(other_df_to_concat.height if not other_df_to_concat.is_empty() else 0)
            # If mapping_config is empty, current_df_to_concat remains original

            # Prepare other_df based on "Maps To" and "Output Name"
            other_df_select_exprs = []
            # Ensure unique output names for columns from other_df, potentially adding suffix if clash with current_df's selected output names
            current_df_final_names = [expr.meta.output_name() for expr in current_df_select_exprs]

            for row_map in mapping_config:
                if row_map['map_to_other'] and row_map['map_to_other'] in other_df_to_concat.columns:
                    other_col_original = row_map['map_to_other']
                    output_name_for_other = row_map['output_name']
                    
                    # If the intended output name for an 'other' column already exists in the selected 'current' columns,
                    # Polars' default horizontal concat would suffix it. Here we don't pre-suffix,
                    # we just select and alias as per user's "Output Name" for the 'other' column.
                    # Polars will handle final suffixing if names still clash after user's explicit renaming.
                    if output_name_for_other != other_col_original:
                        other_df_select_exprs.append(pl.col(other_col_original).alias(output_name_for_other))
                    else: # Output name is same as original 'other' column name
                        # Avoid duplicate selection if this other_col_original was already processed (e.g. mapped from multiple current_df rows)
                        if not any(expr.meta.output_name() == other_col_original for expr in other_df_select_exprs) and \
                           not any(expr.meta.root_names() == [other_col_original] and expr.meta.output_name() == other_col_original for expr in other_df_select_exprs):
                             other_df_select_exprs.append(pl.col(other_col_original))
            
            if other_df_select_exprs:
                other_df_to_concat = other_df_to_concat.select(other_df_select_exprs)
            else:
                other_df_to_concat = pl.DataFrame().with_height(current_df_to_concat.height if not current_df_to_concat.is_empty() else 0)


        elif polars_how_from_ui == "custom_vertical_common":
            common_cols_set = set(current_df_to_concat.columns) & set(other_df_to_concat.columns)
            common_cols = [col for col in current_df_to_concat.columns if col in common_cols_set] 
            if not common_cols:
                self._show_error_message("Concatenate Error", "No common columns found for 'Vertical Common Columns' concatenation.")
                return
            current_df_to_concat = current_df_to_concat.select(common_cols)
            other_df_to_concat = other_df_to_concat.select(common_cols) 
            actual_polars_how = "vertical" 
            processed_mapping_for_log.append({"action": "selected_common_columns_for_vertical_stack", "columns": common_cols})
        else:
            actual_polars_how = polars_how_from_ui 
        
        df_shape_before = self.current_df.shape 
        other_df_original_shape = self.other_df_for_combine.shape 
        self._push_to_undo_stack()
        try:
            # Ensure DFs are not empty before concat, as Polars might error with empty+non-empty
            dfs_to_concat_final = []
            if not current_df_to_concat.is_empty():
                dfs_to_concat_final.append(current_df_to_concat)
            if not other_df_to_concat.is_empty():
                dfs_to_concat_final.append(other_df_to_concat)
            
            if not dfs_to_concat_final: # Both ended up empty
                self.current_df = pl.DataFrame() # Result is empty
            elif len(dfs_to_concat_final) == 1: # Only one DF has data
                 self.current_df = dfs_to_concat_final[0]
            else: # Both have data
                self.current_df = pl.concat(
                    dfs_to_concat_final,
                    how=actual_polars_how, 
                    rechunk=rechunk
                )

            self.update_preview_table()
            self.update_df_info_label()
            self._update_all_column_lists() 
            self.status_bar.showMessage(f"Concatenation (Polars how='{actual_polars_how}') applied.", 2000)
            
            log_details = {"UI Choice How": self.concat_type_combo.currentText(),
                           "Polars How Used": actual_polars_how, 
                           "Rechunk": rechunk, 
                           "Other DF Original Shape": other_df_original_shape, 
                           "Other DF Name": self.other_df_name_hint,
                           "Column Mapping Config (if horizontal)": processed_mapping_for_log if polars_how_from_ui == "horizontal" else "N/A"}
            if actual_polars_how == "vertical" and self.concat_type_combo.currentData() == "custom_vertical_common":
                 log_details["Common Columns Used (custom_vertical_common)"] = common_cols


            logger.log_action("DataTransformer", "Concatenate Operation", 
                              f"Applied concatenation. UI Choice: '{self.concat_type_combo.currentText()}', Polars How: '{actual_polars_how}'.",
                              details=log_details,
                              df_shape_before=df_shape_before, df_shape_after=self.current_df.shape)
        except Exception as e:
            self._show_error_message("Concatenate Error", f"Error applying concatenate: {e}\n\n{traceback.format_exc()}")
            if self.undo_stack: self.current_df = self.undo_stack.pop(); self.redo_stack.clear(); self._update_ui_for_data_state()
            logger.log_action("DataTransformer", "Concatenate Error", "Error during concatenation.", details={"Error": str(e)})


    def _handle_suggest_types_other_df(self):
        if self.other_df_for_combine is None or self.other_df_for_combine.is_empty():
            self._show_error_message("Type Suggestion", "Load 'Other DataFrame' first and ensure it's not empty.")
            return
        self._suggest_and_convert_types_for_other_df()


    def _suggest_and_convert_types_for_other_df(self):
        df_to_analyze = self.other_df_for_combine
        if df_to_analyze is None or df_to_analyze.is_empty(): 
            return

        self.other_df_progress_bar.setVisible(True)
        self.other_df_progress_bar.setRange(0,100) 
        self.other_df_progress_bar.setValue(0)
        self.status_bar.showMessage("Analyzing 'Other DF' for type conversions...", 0) 
        QApplication.processEvents()

        suggested_conversions = [] 
        MAX_UNIQUE_CATEGORICAL_ABS = 50 
        MAX_UNIQUE_CATEGORICAL_REL = 0.1 

        columns_to_analyze = [col for col in df_to_analyze.columns if col != "__original_index__"]

        for i, col_name in enumerate(columns_to_analyze):
            progress_val = int(((i + 1) / len(columns_to_analyze)) * 50) if columns_to_analyze else 0
            self.other_df_progress_bar.setValue(progress_val)
            self.status_bar.showMessage(f"Analyzing (Other DF): {col_name} ({i+1}/{len(columns_to_analyze)})")
            QApplication.processEvents()
            
            series = df_to_analyze.get_column(col_name)
            non_null_series = series.drop_nulls()
            
            current_suggestion = "string_key" 

            if non_null_series.is_empty():
                suggested_conversions.append((col_name, current_suggestion)) 
                continue

            if series.dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
                current_suggestion = "integer"
            elif series.dtype in [pl.Float32, pl.Float64]:
                current_suggestion = "numeric_float"
            elif series.dtype == pl.Categorical:
                current_suggestion = "category" 
            elif series.dtype in [pl.Date, pl.Datetime]:
                current_suggestion = "datetime"
            elif series.dtype == pl.Boolean: 
                current_suggestion = "category" 
            
            elif series.dtype == pl.Utf8:
                try:
                    sample_size_dt = min(1000, non_null_series.len())
                    if sample_size_dt > 0:
                        datetime_cast_sample_null_ratio = non_null_series.head(sample_size_dt).str.to_datetime(
                            strict=False, time_unit='us', ambiguous='earliest', format=None
                        ).is_null().sum() / sample_size_dt

                        if datetime_cast_sample_null_ratio <= 0.1:
                            casted_dt = non_null_series.str.to_datetime(strict=False, time_unit='us', ambiguous='earliest', format=None)
                            if casted_dt.null_count() / non_null_series.len() <= 0.05: 
                                current_suggestion = "datetime"
                except Exception: pass

                if current_suggestion == "string_key": 
                    try:
                        _ = non_null_series.cast(pl.Int64, strict=True) 
                        current_suggestion = "integer"
                    except pl.PolarsError: 
                        try:
                            casted_float_for_int_check = non_null_series.cast(pl.Float64, strict=False)
                            original_string_non_null_count = non_null_series.len() 
                            if not casted_float_for_int_check.is_null().all() and \
                               (casted_float_for_int_check.drop_nulls() % 1 == 0).all():
                                if casted_float_for_int_check.null_count() <= original_string_non_null_count * 0.05 :
                                    current_suggestion = "integer"
                            elif current_suggestion == "string_key": 
                                if casted_float_for_int_check.null_count() <= original_string_non_null_count * 0.05:
                                    current_suggestion = "numeric_float"
                        except Exception: pass

                if current_suggestion == "string_key": 
                    n_unique = non_null_series.n_unique()
                    if n_unique <= MAX_UNIQUE_CATEGORICAL_ABS or \
                       (non_null_series.len() > 0 and (n_unique / non_null_series.len()) <= MAX_UNIQUE_CATEGORICAL_REL):
                        current_suggestion = "category"
            
            suggested_conversions.append((col_name, current_suggestion))

        self.other_df_progress_bar.setValue(50)
        self.status_bar.showMessage("Presenting suggestions for 'Other DF'...")
        QApplication.processEvents()

        if not suggested_conversions:
            self.other_df_progress_bar.setVisible(False)
            self.status_bar.showMessage("No columns found to analyze or suggest conversions for 'Other DF'.", 3000)
            QMessageBox.information(self, "No Suggestions (Other DF)", "No columns available for conversion suggestions in 'Other DF'.")
            return

        dialog = QDialog(self) 
        dialog.setWindowTitle("Suggested Type Conversions for 'Other DataFrame'")
        dialog_layout = QVBoxLayout(dialog)

        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("<b>Column Name</b>"), 2)
        header_layout.addWidget(QLabel("<b>String/Key</b>"), 1, alignment=Qt.AlignCenter)
        header_layout.addWidget(QLabel("<b>Category</b>"), 1, alignment=Qt.AlignCenter)
        header_layout.addWidget(QLabel("<b>Numeric (Float)</b>"), 1, alignment=Qt.AlignCenter)
        header_layout.addWidget(QLabel("<b>Integer</b>"), 1, alignment=Qt.AlignCenter)
        header_layout.addWidget(QLabel("<b>Datetime</b>"), 1, alignment=Qt.AlignCenter)
        dialog_layout.addLayout(header_layout)

        scroll_area = QScrollArea()
        scroll_widget = QWidget()
        grid_layout = QGridLayout(scroll_widget)
        grid_layout.setColumnStretch(0, 2) 
        for i in range(1, 6): grid_layout.setColumnStretch(i, 1)

        self.other_df_conversion_radio_button_groups.clear() 

        for row_idx, (col_name, suggested_type) in enumerate(suggested_conversions):
            col_name_label = QLabel(col_name)
            string_rb = QRadioButton(); category_rb = QRadioButton()
            float_rb = QRadioButton(); integer_rb = QRadioButton()
            datetime_rb = QRadioButton()

            button_group = QButtonGroup(dialog) 
            button_group.addButton(string_rb); button_group.addButton(category_rb)
            button_group.addButton(float_rb); button_group.addButton(integer_rb)
            button_group.addButton(datetime_rb)

            if suggested_type == "string_key": string_rb.setChecked(True)
            elif suggested_type == "category": category_rb.setChecked(True)
            elif suggested_type == "numeric_float": float_rb.setChecked(True)
            elif suggested_type == "integer": integer_rb.setChecked(True)
            elif suggested_type == "datetime": datetime_rb.setChecked(True)
            else: string_rb.setChecked(True) 

            grid_layout.addWidget(col_name_label, row_idx, 0)
            grid_layout.addWidget(string_rb, row_idx, 1, alignment=Qt.AlignCenter)
            grid_layout.addWidget(category_rb, row_idx, 2, alignment=Qt.AlignCenter)
            grid_layout.addWidget(float_rb, row_idx, 3, alignment=Qt.AlignCenter)
            grid_layout.addWidget(integer_rb, row_idx, 4, alignment=Qt.AlignCenter)
            grid_layout.addWidget(datetime_rb, row_idx, 5, alignment=Qt.AlignCenter)

            self.other_df_conversion_radio_button_groups.append(
                (col_name_label, string_rb, category_rb, float_rb, integer_rb, datetime_rb)
            )
        
        scroll_widget.setLayout(grid_layout)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)
        dialog_layout.addWidget(scroll_area)

        button_box_layout = QHBoxLayout() 
        help_button = QPushButton("?")
        help_button.setFixedSize(25, 25)
        help_button.setToolTip("Help on Data Types")
        help_button.clicked.connect(self._show_type_conversion_help_for_other_df) 
        
        dialog_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        dialog_buttons.accepted.connect(dialog.accept)
        dialog_buttons.rejected.connect(dialog.reject)

        button_box_layout.addWidget(help_button)
        button_box_layout.addStretch() 
        button_box_layout.addWidget(dialog_buttons)
        dialog_layout.addLayout(button_box_layout) 

        dialog.setMinimumWidth(700) 
        dialog.setMinimumHeight(400)

        if dialog.exec() == QDialog.Accepted:
            cols_to_float, cols_to_int, cols_to_datetime, cols_to_category, cols_to_string = [], [], [], [], []

            for col_label, str_rb, cat_rb, flt_rb, int_rb, dt_rb in self.other_df_conversion_radio_button_groups:
                col_name_to_convert = col_label.text()
                if col_name_to_convert not in self.other_df_for_combine.columns:
                    continue
                current_col_dtype = self.other_df_for_combine.schema[col_name_to_convert]


                if flt_rb.isChecked() and current_col_dtype not in [pl.Float32, pl.Float64]:
                    cols_to_float.append(col_name_to_convert)
                elif int_rb.isChecked() and current_col_dtype not in [pl.Int8,pl.Int16,pl.Int32,pl.Int64,pl.UInt8,pl.UInt16,pl.UInt32,pl.UInt64]:
                    cols_to_int.append(col_name_to_convert)
                elif dt_rb.isChecked() and current_col_dtype not in [pl.Date, pl.Datetime]:
                    cols_to_datetime.append(col_name_to_convert)
                elif cat_rb.isChecked() and current_col_dtype != pl.Categorical:
                    cols_to_category.append(col_name_to_convert)
                elif str_rb.isChecked() and current_col_dtype != pl.Utf8: 
                    cols_to_string.append(col_name_to_convert)
            
            any_conversion_done = False
            if cols_to_float:
                self._temp_cols_for_batch_other_df = cols_to_float
                self._batch_convert_columns_for_other_df("numeric_float", target_polars_type=pl.Float64)
                any_conversion_done = True
            if cols_to_int:
                self._temp_cols_for_batch_other_df = cols_to_int
                self._batch_convert_columns_for_other_df("integer", target_polars_type=pl.Int64)
                any_conversion_done = True
            if cols_to_datetime:
                self._temp_cols_for_batch_other_df = cols_to_datetime
                self._batch_convert_columns_for_other_df("datetime", target_polars_type=pl.Datetime)
                any_conversion_done = True
            if cols_to_category:
                self._temp_cols_for_batch_other_df = cols_to_category
                self._batch_convert_columns_for_other_df("categorical", target_polars_type=pl.Categorical)
                any_conversion_done = True
            if cols_to_string: 
                self._temp_cols_for_batch_other_df = cols_to_string
                self._batch_convert_columns_for_other_df("string_key", target_polars_type=pl.Utf8)
                any_conversion_done = True
            
            if any_conversion_done:
                self.status_bar.showMessage("Selected type conversions applied to 'Other DF'.", 3000)
            else:
                self.status_bar.showMessage("No changes made to 'Other DF' data types.", 3000)
            self.other_df_progress_bar.setValue(100)

        else: 
            self.status_bar.showMessage("Conversion suggestions for 'Other DF' cancelled.", 3000)
            self.other_df_progress_bar.setValue(100) 
        
        QTimer.singleShot(500, lambda: self.other_df_progress_bar.setVisible(False))
        self.other_df_conversion_radio_button_groups.clear() 


    def _batch_convert_columns_for_other_df(self, type_name, target_polars_type=None):
        if self.other_df_for_combine is None or self.other_df_for_combine.is_empty():
            self._show_error_message("Batch Conversion Error (Other DF)", "No 'Other DataFrame' loaded or it's empty.")
            return
        
        selected_names = self._temp_cols_for_batch_other_df 
        if not selected_names:
            return

        num_selected = len(selected_names)
        
        self.other_df_progress_bar.setVisible(True)
        self.other_df_progress_bar.setValue(50) 
        
        converted_count = 0
        skipped_details = []
        error_details = []
        user_cancelled_all = False
        yes_to_all_warnings = False
        
        df_shape_before_batch_op = self.other_df_for_combine.shape

        for i, col_name in enumerate(selected_names):
            current_progress = 50 + int(((i + 0.5) / num_selected) * 50)
            self.other_df_progress_bar.setValue(current_progress)
            self.status_bar.showMessage(f"Converting (Other DF) {i+1}/{num_selected}: '{col_name}' to {type_name}...")
            QApplication.processEvents()

            try:
                if col_name not in self.other_df_for_combine.columns:
                    error_details.append(f"'{col_name}': Column not found in 'Other DataFrame'.")
                    continue

                original_column_series = self.other_df_for_combine.get_column(col_name)
                original_nulls = original_column_series.is_null().sum()
                current_col_schema_dtype = self.other_df_for_combine.schema[col_name]

                final_conversion_expr = None
                actual_target_polars_type = target_polars_type 

                if type_name == "numeric_float":
                    final_conversion_expr = pl.col(col_name).cast(pl.Float64, strict=False)
                elif type_name == "integer":
                    final_conversion_expr = pl.col(col_name).cast(pl.Int64, strict=False)
                elif type_name == "datetime":
                    if current_col_schema_dtype == pl.Utf8:
                        final_conversion_expr = pl.col(col_name).str.to_datetime(strict=False, time_unit='ns', ambiguous='earliest').dt.replace_time_zone(None)
                    else:
                        final_conversion_expr = pl.col(col_name).cast(pl.Datetime, strict=False).dt.replace_time_zone(None)
                elif type_name == "categorical":
                    final_conversion_expr = pl.col(col_name).cast(pl.Categorical, strict=False)
                elif type_name == "string_key":
                    final_conversion_expr = pl.col(col_name).cast(pl.Utf8)
                
                if final_conversion_expr is None:
                    error_details.append(f"'{col_name}': No conversion logic for type '{type_name}'.")
                    continue
                
                temp_converted_series = self.other_df_for_combine.select(final_conversion_expr.alias("___temp_batch_check___")).get_column("___temp_batch_check___")
                converted_nulls = temp_converted_series.is_null().sum()
                newly_created_nulls = converted_nulls - original_nulls

                significant_new_nans = False
                original_non_null_count = self.other_df_for_combine.height - original_nulls
                if original_non_null_count > 0 and (newly_created_nulls / original_non_null_count > 0.1):
                    significant_new_nans = True
                elif newly_created_nulls > 0.05 * self.other_df_for_combine.height:
                    significant_new_nans = True
                if type_name == "string_key": significant_new_nans = False

                if significant_new_nans and not yes_to_all_warnings:
                    msg_box = QMessageBox(self); msg_box.setIcon(QMessageBox.Warning)
                    msg_box.setWindowTitle(f"Conversion Warning for '{col_name}' (Other DF)")
                    msg_box.setText(f"Converting '{col_name}' to {type_name} resulted in {newly_created_nulls} new empty/null values. Proceed with this column?")
                    yes_button = msg_box.addButton("Yes", QMessageBox.YesRole); no_button = msg_box.addButton("No", QMessageBox.NoRole)
                    yes_all_button = msg_box.addButton("Yes to All", QMessageBox.AcceptRole); cancel_all_button = msg_box.addButton("Cancel All", QMessageBox.RejectRole)
                    msg_box.setDefaultButton(yes_button); msg_box.exec()
                    
                    clicked_btn = msg_box.clickedButton()
                    if clicked_btn == no_button: 
                        skipped_details.append(f"'{col_name}': Skipped by user due to new nulls."); continue
                    elif clicked_btn == cancel_all_button: 
                        user_cancelled_all = True; skipped_details.append(f"'{col_name}': Batch cancelled by user."); break
                    elif clicked_btn == yes_all_button: 
                        yes_to_all_warnings = True

                self.other_df_for_combine = self.other_df_for_combine.with_columns(final_conversion_expr.alias(col_name))
                converted_count += 1
            except Exception as e: 
                error_details.append(f"'{col_name}': {str(e)}")
                print(f"Error converting (Other DF) {col_name} to {type_name}: {e}\n{traceback.format_exc()}")

        self.other_df_progress_bar.setValue(100)
        self.status_bar.showMessage(f"Batch conversion for 'Other DF' processing...", 2000)
        QApplication.processEvents()

        self._populate_checkbox_list_widget(self.other_df_cols_list, self.other_df_for_combine)
        self._populate_checkbox_list_widget(self.merge_right_on_list, self.other_df_for_combine)
        
        summary_parts = [f"{converted_count} of {num_selected} columns in 'Other DF' processed for conversion to '{type_name}'." if not user_cancelled_all else f"Batch conversion for 'Other DF' to '{type_name}' cancelled. {converted_count} processed."]
        if skipped_details: summary_parts.append("\nSkipped:\n" + "\n".join(skipped_details))
        if error_details: summary_parts.append("\nErrors:\n" + "\n".join(error_details))
        
        if skipped_details or error_details or (converted_count < num_selected and not user_cancelled_all) :
            QMessageBox.information(self, "Batch Conversion Summary (Other DF)", "\n".join(summary_parts))
        
        self.status_bar.showMessage(f"Batch {type_name} conversion for 'Other DF' finished.", 3000)
        
        logger.log_action("DataTransformer", "Batch Type Conversion (Other DF)", 
                          f"Attempted to convert {num_selected} columns in '{self.other_df_name_hint}' to {type_name}.",
                          details={"Columns Selected": selected_names, 
                                   "Target Type Name": type_name,
                                   "Target Polars Type": str(actual_target_polars_type),
                                   "Converted Count": converted_count,
                                   "Skipped": skipped_details, "Errors": error_details,
                                   "DF Name": self.other_df_name_hint},
                          df_shape_before=df_shape_before_batch_op,
                          df_shape_after=self.other_df_for_combine.shape)
        
        self._temp_cols_for_batch_other_df = [] 

    def _show_type_conversion_help_for_other_df(self):
        help_text = """
        <html><head><style> body { font-family: sans-serif; font-size: 10pt; } h3 { color: #333; } p { margin-bottom: 8px; } ul { margin-top: 0px; padding-left: 20px; } li { margin-bottom: 4px; } code { background-color: #f0f0f0; padding: 1px 3px; border-radius: 3px; font-family: monospace;} </style></head><body>
        <h3>Understanding Data Types for Conversion:</h3>
        <p><b>String/Key:</b><ul><li>Text data (words, codes, non-math numbers like IDs).</li></ul></p>
        <p><b>Category:</b><ul><li>Text/numbers for distinct groups from a limited set (e.g., Gender, Country Code). Good for few unique values.</li></ul></p>
        <p><b>Numeric (Float):</b><ul><li>Numbers with decimals for measurements, amounts, calculations.</li></ul></p>
        <p><b>Integer (Whole Number):</b><ul><li>Whole numbers for counts or quantities.</li></ul></p>
        <p><b>Datetime:</b><ul><li>Specific dates and/or times. Allows date-based calculations.</li></ul></p>
        <p><i>Choosing the right type helps with analysis and memory. Suggestions are based on data appearance.</i></p>
        </body></html>
        """
        help_dialog = QMessageBox(self) 
        help_dialog.setWindowTitle("Data Type Conversion Help (Other DF)")
        help_dialog.setTextFormat(Qt.RichText)
        help_dialog.setText(help_text)
        help_dialog.setIcon(QMessageBox.Information)
        help_dialog.setStandardButtons(QMessageBox.Ok)
        help_dialog.exec()

    def _create_help_dialog(self, title, html_content):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumSize(650, 450) 
        layout = QVBoxLayout(dialog)
        text_browser = QTextBrowser()
        text_browser.setHtml(html_content)
        text_browser.setOpenExternalLinks(True)
        layout.addWidget(text_browser)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(dialog.accept)
        layout.addWidget(button_box)
        dialog.exec()

    def _show_concat_help(self):
        html_content = """
        <html><head><style> body { font-family: sans-serif; font-size: 10pt; } h3 { color: #333377; margin-top:1em; } code { background-color: #f0f0f0; padding: 1px 3px; border-radius: 3px; font-family: monospace;} pre { background-color: #f8f8f8; border: 1px solid #ddd; padding: 5px; border-radius: 3px; overflow-x: auto; } table { border-collapse: collapse; margin-top: 5px; margin-bottom:10px;} th, td { border: 1px solid #ccc; padding: 4px; text-align: left;} </style></head><body>
        <h2>Concatenation (<code>pl.concat</code>) Strategies</h2>
        <p>Concatenation combines multiple DataFrames. The <code>how</code> parameter determines the strategy.</p>

        <h3>Vertical Strategies (Stacking Rows)</h3>
        <p>These strategies append rows from one DataFrame below another.</p>
        <ul>
            <li><strong><code>Vertical Relaxed</code>:</strong>
                <ul>
                    <li>Stacks DataFrames vertically. Columns are matched by name (order doesn't matter).</li>
                    <li>If a column exists in one DF but not another, it's added and filled with nulls.</li>
                    <li>Data types of common columns are supercasted if necessary (e.g., Int64 and Float64 become Float64).</li>
                    <li>Generally recommended for robust vertical stacking when schemas might differ slightly. (Corresponds to Polars <code>how="vertical_relaxed"</code>)</li>
                </ul>
            </li>
            <li><strong><code>Vertical Strict</code>:</strong>
                <ul>
                    <li>Stacks DataFrames vertically.</li>
                    <li>Requires schemas (column names, order, and data types) to match <em>exactly</em>.</li>
                    <li>Fastest vertical method if schemas align perfectly. (Corresponds to Polars <code>how="vertical"</code>)</li>
                </ul>
            </li>
             <li><strong><code>Vertical Common Columns</code> (Custom Logic):</strong>
                <ul>
                    <li>This UI option first identifies columns with identical names in both DataFrames.</li>
                    <li>Both DataFrames are temporarily subsetted to <em>only these common columns</em>.</li>
                    <li>Then, a strict <code>vertical</code> concatenation is performed on these subsetted DataFrames.</li>
                    <li>Ensures only shared schema parts are stacked, discarding non-common columns from both before stacking. Data types must match for common columns or an error may occur.</li>
                </ul>
            </li>
            <li><strong><code>Diagonal Relaxed</code>:</strong>
                <ul>
                    <li>Conceptually interleaves rows, then aligns by column name, fills nulls, and supercasts types.</li>
                    <li>Often produces the same result as <code>vertical_relaxed</code> for two DataFrames. (Corresponds to Polars <code>how="diagonal_relaxed"</code>)</li>
                </ul>
            </li>
        </ul>

        <h3>Horizontal Strategy (Side-by-Side)</h3>
        <ul>
            <li><strong><code>Horizontal</code>:</strong>
                <ul>
                    <li>Joins DataFrames side-by-side (column-wise).</li>
                    <li>Polars attempts to align rows if possible (e.g. if DataFrames have the same height). If heights differ, the behavior might involve null-padding up to the height of the tallest DataFrame. It's best to ensure row counts are compatible.</li>
                    <li>Use the <strong>Column Mapping UI</strong> to:
                        <ul>
                        <li>Select which columns from 'Current DF' to include (via checkbox).</li>
                        <li>For each 'Current DF' column, select which 'Other DF' column it maps to (or '&lt;None&gt;').</li>
                        <li>Specify the 'Output Column Name'. If this name clashes with another chosen output name or an existing column name, Polars may still auto-suffix (e.g., <code>_other</code>). Explicit unique output names are best.</li>
                        </ul>
                     (Corresponds to Polars <code>how="horizontal"</code>)</li>
                </ul>
            </li>
        </ul>
        
        <h3>Alignment Strategies (Experimental/Advanced - Horizontal Focus)</h3>
        <p>These strategies first attempt to align rows based on the values in the <strong>first column</strong> of each DataFrame, and then perform a horizontal concatenation. DataFrames should ideally be sorted by their first column for predictable results.</p>
        <ul>
            <li><strong><code>Align</code>:</strong> (Polars <code>how="align"</code>)</li>
            <li><strong><code>Align Full</code> (Outer):</strong> (Polars <code>how="align_full"</code>)</li>
            <li><strong><code>Align Inner</code> (Inner):</strong> (Polars <code>how="align_inner"</code>)</li>
            <li><strong><code>Align Left</code> (Left):</strong> (Polars <code>how="align_left"</code>)</li>
            <li><strong><code>Align Right</code> (Right):</strong> (Polars <code>how="align_right"</code>)</li>
        </ul>
        
        <p><strong>Rechunk:</strong> The 'Rechunk' option (if checked) performs a rechunk operation after concatenation. This can improve performance for subsequent operations. Usually recommended.</p>
        </body></html>
        """
        self._create_help_dialog("Concatenation Strategies Help", html_content)

    def _show_merge_help(self):
        html_content = """
        <html><head><style> body { font-family: sans-serif; font-size: 10pt; } h3 { color: #333377; margin-top:1em; } code { background-color: #f0f0f0; padding: 1px 3px; border-radius: 3px; font-family: monospace;} pre { background-color: #f8f8f8; border: 1px solid #ddd; padding: 5px; border-radius: 3px; overflow-x: auto; } table { border-collapse: collapse; margin-top: 5px; } th, td { border: 1px solid #ccc; padding: 4px; text-align: left;} </style></head><body>
        <h2>Merge/Join Types (<code>df.join()</code>)</h2>
        <p>Joining combines DataFrames based on common key columns or by row index.</p>

        <ul>
            <li><strong><code>inner</code>:</strong> Returns only the rows where the key(s) exist in <em>both</em> DataFrames.</li>
            <li><strong><code>left</code>:</strong> Returns all rows from the <em>left</em> DataFrame and the matched rows from the <em>right</em> DataFrame. Nulls fill where no match in right.</li>
            <li><strong><code>outer</code>:</strong> Returns all rows from <em>both</em> DataFrames. Nulls fill where no match in the other DataFrame. (Polars also accepts <code>full</code> as an alias for <code>outer</code>).</li>
            <li><strong><code>semi</code>:</strong> Returns only rows from the <em>left</em> DataFrame for which there is a matching key in the <em>right</em> DataFrame. Only columns from the left DataFrame are included.</li>
            <li><strong><code>anti</code>:</strong> Returns only rows from the <em>left</em> DataFrame for which there is <em>no</em> matching key in the <em>right</em> DataFrame. Only columns from the left DataFrame are included.</li>
            <li><strong><code>cross</code>:</strong> Returns the Cartesian product (all combinations of rows). No keys needed. <strong>Use with extreme caution on large DataFrames!</strong></li>
        </ul>
        
        <p><strong>Suffix:</strong> If both DataFrames have non-key columns with the same name, the specified suffix (e.g., <code>_other</code>) is added to distinguish the columns coming from the 'Other' (right) DataFrame in the result.</p>
        <h3>Example:</h3>
        <pre>
left_df = pl.DataFrame({"ID": [1, 2, 3, 5], "Name": ["Alice", "Bob", "Charlie", "David"], "Value": [10,20,30,40]})
right_df = pl.DataFrame({"ID": [1, 2, 4, 5], "City": ["NY", "LA", "SF", "CHI"], "Value": [15,25,45,55]})

# Inner Join with suffix="_other" for clashing 'Value' column
inner_join = left_df.join(right_df, on="ID", how="inner", suffix="_other")
# Output:
# │ ID  ┆ Name  ┆ Value ┆ City ┆ Value_other │
# │ 1   ┆ Alice ┆ 10    ┆ NY   ┆ 15          │
# │ 2   ┆ Bob   ┆ 20    ┆ LA   ┆ 25          │
# │ 5   ┆ David ┆ 40    ┆ CHI  ┆ 55          │
        </pre>
        </body></html>
        """
        self._create_help_dialog("Merge/Join Types Help", html_content)

    def _show_reshape_help(self):
        html_content = """
        <html><head><style> body { font-family: sans-serif; font-size: 10pt; } h3 { color: #333377; margin-top:1em; } code { background-color: #f0f0f0; padding: 1px 3px; border-radius: 3px; font-family: monospace;} pre { background-color: #f8f8f8; border: 1px solid #ddd; padding: 5px; border-radius: 3px; overflow-x: auto; } table { border-collapse: collapse; margin-top: 5px; margin-bottom:10px; } th, td { border: 1px solid #ccc; padding: 4px; text-align: left;} </style></head><body>
        <h2>Reshaping Data: Pivot vs. Melt</h2>
        <p>Reshaping changes the structure of your DataFrame from "wide" to "long" format, or vice-versa.</p>

        <h3>Melt (Unpivot / Wide to Long)</h3>
        <p><strong>Purpose:</strong> Makes data "taller" and "narrower". Converts columns into rows.</p>
        <ul>
            <li><strong>ID Variable(s) (<code>id_vars</code>):</strong> Columns to keep as they are. (Excel: Rows in a PivotTable Report Filter / Row Labels that are not part of the unpivot action).</li>
            <li><strong>Value Variables (<code>value_vars</code>):</strong> Columns to unpivot. Their names go into a new 'variable' column, their values into a new 'value' column. If empty, all non-ID vars are melted. (Excel: These are like multiple columns you'd drag into the 'Values' area of a PivotTable one by one if they represented different time periods or categories of the same measure).</li>
        </ul>
        <p><strong>Example:</strong></p>
        <p><em>Input:</em></p> <table><tr><th>Prod</th><th>Q1_Sales</th><th>Q2_Sales</th></tr><tr><td>A</td><td>10</td><td>12</td></tr></table>
        <p><em>Melt (ID Vars=["Prod"]):</em></p>
        <table><tr><th>Prod</th><th>variable</th><th>value</th></tr><tr><td>A</td><td>Q1_Sales</td><td>10</td></tr><tr><td>A</td><td>Q2_Sales</td><td>12</td></tr></table>

        <hr style="margin: 20px 0;">
        <h3>Pivot (Long to Wide)</h3>
        <p><strong>Purpose:</strong> Makes data "wider" and "shorter". Converts unique values from one column into new column headers.</p>
        <ul>
            <li><strong>Index Column(s) (<code>index</code>):</strong> Column(s) forming unique rows in the new wide DF. (Excel: Fields dragged to 'Rows' area of a PivotTable).</li>
            <li><strong>Columns (from values of) (<code>columns</code>):</strong> Column whose unique values become new column headers. (Excel: Fields dragged to 'Columns' area).</li>
            <li><strong>Values (populate new cols) (<code>values</code>):</strong> Column whose values fill the new cells. (Excel: Fields dragged to 'Values' area).</li>
            <li><strong>Aggregation (<code>aggregate_function</code>):</strong> Handles multiple values for the same index/columns combination (e.g., "first", "sum", "mean"). (Excel: Summarize Values By option in PivotTable).</li>
        </ul>
        <p><strong>Example (using melted output from above):</strong></p>
        <p><em>Input:</em></p> <table><tr><th>Prod</th><th>Quarter</th><th>Sales</th></tr><tr><td>A</td><td>Q1_Sales</td><td>10</td></tr><tr><td>A</td><td>Q2_Sales</td><td>12</td></tr></table>
        <p><em>Pivot (Index="Prod", Columns="Quarter", Values="Sales", Aggregation="first"):</em></p>
        <table><tr><th>Prod</th><th>Q1_Sales</th><th>Q2_Sales</th></tr><tr><td>A</td><td>10</td><td>12</td></tr></table>
        </body></html>
        """
        self._create_help_dialog("Reshaping Data: Pivot vs. Melt Help", html_content)


    # --- Utility Methods ---
    def _show_error_message(self, title, message):
        QMessageBox.critical(self, title, message)
        self.status_bar.showMessage(f"Error: {message[:100]}", 5000)

    def _confirm_action(self, title, message):
        reply = QMessageBox.question(self, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply == QMessageBox.Yes

    def closeEvent(self, event):
        if self.undo_stack:
            reply = QMessageBox.question(self, "Confirm Exit",
                                           "You have unsaved transformations. Do you want to save before exiting?",
                                           QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                                           QMessageBox.Cancel)
            if reply == QMessageBox.Save:
                self._handle_save_file() 
                logger.log_action("DataTransformer", "Application Close", "User chose to save transformations upon exit.")
                event.accept() 
            elif reply == QMessageBox.Discard:
                logger.log_action("DataTransformer", "Application Close", "User chose to discard transformations upon exit.")
                event.accept()
            else: 
                logger.log_action("DataTransformer", "Application Close", "User cancelled closing.")
                event.ignore()
        else:
            logger.log_action("DataTransformer", "Application Close", "Application closed (no unsaved transformations).")
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    class DummySourceApp(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Dummy Source Application")
        def load_dataframe_from_source(self, df, name_hint, source_log_file_path):
            print(f"DummySourceApp: Received data '{name_hint}' (shape: {df.shape}) from DataTransformer.")
            print(f"DummySourceApp: Log file path from DT: {source_log_file_path}")
            QMessageBox.information(self, "Data Received by Dummy", f"Received: {name_hint}\nShape: {df.shape}\nLog: {source_log_file_path}")

    sample_data_current = {
        'ID': [1, 2, 3, 4, 5], 
        'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'], 
        'Value_X': [10, 20, 30, 40, 50], 
        'Common_Col': [100,200,300,400,500],
        'Date_Current': pl.date_range(date(2023,1,1), date(2023,1,5), eager=True)
    }
    initial_polars_df = pl.DataFrame(sample_data_current)
    
    transformer_window = DataTransformer(initial_df=initial_polars_df, df_name_hint="SampleDataCurrent.testing")

    sample_data_other = {
        'ID_other': [2, 3, 4, 6, 7], 
        'Department': ['HR', 'IT', 'Sales', 'RD', 'MKG'], 
        'Value_Y': [100, 200, 300, 600, 700], 
        'Name_Rel': ['Robert', 'Charles', 'Diana', 'Frank', 'Grace'], 
        'Common_Col': [11,22,33,66,77], 
        'Date_Other': pl.date_range(date(2024,1,1), date(2024,1,5), eager=True)
    } 
    other_df = pl.DataFrame(sample_data_other)
    
    transformer_window.other_df_for_combine = other_df
    transformer_window.other_df_name_hint = "SampleDataOther.testing"
    transformer_window.update_df_info_label()
    transformer_window._update_all_column_lists() 
    transformer_window._update_ui_for_data_state()


    transformer_window.show()
    sys.exit(app.exec())
