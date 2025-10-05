# visual_workshop.py
import sys
import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
import plotly.subplots as sp
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QScrollArea, QGroupBox, QSplitter,
    QFileDialog, QListWidget, QAbstractItemView, QCheckBox, QSpinBox, QGridLayout,
    QMessageBox, QDialog, QDialogButtonBox, QSizePolicy, QRadioButton, QButtonGroup,
    QTextEdit, QToolButton
)
from PySide6.QtCore import Qt, Slot, QSize, QUrl, QBuffer, QIODevice
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWebEngineWidgets import QWebEngineView
# Import necessary classes for the scheme handler
from PySide6.QtWebEngineCore import (
    QWebEngineSettings, QWebEngineProfile, QWebEnginePage,
    QWebEngineUrlScheme, QWebEngineUrlSchemeHandler, QWebEngineUrlRequestJob
)
import traceback
import datetime
from scipy.stats import norm
from main import resource_path # Assuming main.py and resource_path are available
from operation_logger import logger # Assuming operation_logger.py is available
from ast import literal_eval
import re

try:
    import kaleido
except ImportError:
    KALEIDO_AVAILABLE = False
    print("Kaleido not found. PNG/Static image export will not be available.")
else:
    KALEIDO_AVAILABLE = True

PICKAXE_LOADERS_AVAILABLE = False
try:
    # from pickaxe.common.file_loaders import FileLoaderRegistry # Example
    PICKAXE_LOADERS_AVAILABLE = False
except ImportError:
    PICKAXE_LOADERS_AVAILABLE = False



class PlotSchemeHandler(QWebEngineUrlSchemeHandler):
    """
    A custom URL scheme handler to serve Plotly figures from memory.
    This avoids the 2MB limit of setHtml and the need for temporary files.
    """
    def __init__(self, app_instance):
        super().__init__()
        # Store a reference to the main app to access the plot figure
        self.app = app_instance

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:
        url = job.requestUrl()
        plot_id = url.host()

        if self.app.current_plotly_fig and plot_id == "current_plot":
            fig = self.app.current_plotly_fig
            
            html = pio.to_html(fig, full_html=True, include_plotlyjs=True)
            
            buf = QBuffer()
            buf.open(QIODevice.WriteOnly)
            buf.write(html.encode('utf-8'))
            buf.seek(0)
            # The buffer does not need to be closed here, as reply() will read from it.
            # job.destroyed will take care of the QBuffer's lifetime if we parent it.
            buf.setParent(job)
            
            job.reply(b"text/html", buf)
        else:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)

class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title="", parent=None, checked=False): # Added checked default
        super().__init__(title, parent)
        self.setCheckable(True)
        
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 10, 5, 5)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0,0,0,0) # Content area usually has no extra margins

        self._main_layout.addWidget(self.content_widget)
        
        self.toggled.connect(self._toggle_content_internal) # Connect to internal slot
        self.setChecked(checked) # Set initial state and trigger toggle

    def _toggle_content_internal(self, checked):
        self.content_widget.setVisible(checked)
        if checked:
            self.content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding) # Allow expansion
            self.content_widget.setMaximumHeight(16777215) # QWIDGETSIZE_MAX
        else:
            # When hiding, it takes minimum space
            self.content_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.content_widget.setMaximumHeight(0)
        
        self.content_widget.adjustSize() 
        self.adjustSize()

        parent = self.parentWidget()
        while parent:
            if parent.layout():
                parent.layout().activate()
            if isinstance(parent, QScrollArea):
                 if parent.widget():
                     parent.widget().adjustSize()
                 parent.updateGeometry()
            parent = parent.parentWidget()
        
        QApplication.processEvents()


    def add_widget_to_content(self, widget):
        self.content_layout.addWidget(widget)
        if self.isChecked():
            self.content_widget.adjustSize()
            self.adjustSize()

    def get_content_layout(self):
        return self.content_layout


class VisualWorkshopApp(QMainWindow):
    def __init__(self, pickaxe_instance=None, log_file_to_use=None):
        super().__init__()
        self.source_app = pickaxe_instance
        self.current_log_file_path = log_file_to_use

        self.current_df = None
        self.current_filename_hint = "No data"
        self.basic_config_widgets = {}
        self.advanced_config_widgets = {}
        self.distplot_advanced_widgets = {}
        self.pie_advanced_widgets = {}
        self.advanced_group = None
        self.distplot_advanced_group = None
        self.pie_advanced_group = None
        
        # --- Scheme Handler Setup ---
        self.PLOT_SCHEME = b"plotly-figure"
        
        # FIX: Check if the scheme is already registered before trying again.
        # This prevents the "already registered" warning in interactive environments.
        scheme = QWebEngineUrlScheme.schemeByName(self.PLOT_SCHEME)
        if not scheme.name(): # A non-registered scheme will have an empty name
            scheme = QWebEngineUrlScheme(self.PLOT_SCHEME)
            scheme.setSyntax(QWebEngineUrlScheme.Syntax.HostAndPort)
            scheme.setDefaultPort(24815)
            scheme.registerScheme(scheme)

        self.profile = QWebEngineProfile(f"vw-profile-{id(self)}", self)
        self.plot_handler = PlotSchemeHandler(self)
        self.profile.installUrlSchemeHandler(self.PLOT_SCHEME, self.plot_handler)
        
        self.hover_data_configs = {}
        
        if not self.current_log_file_path:
            default_log_name = f".__log__vw_session_default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            self.current_log_file_path = os.path.join(os.getcwd(), default_log_name)
            logger.set_log_file(self.current_log_file_path, "VisualWorkshop (Default Session)", associated_data_file="N/A")

        self.setWindowTitle("Visual Workshop - Polars Plotting Studio")
        self.setGeometry(150, 150, 1400, 900)

        pickaxe_icon_path = resource_path("pickaxe.ico")
        if os.path.exists(pickaxe_icon_path):
            self.setWindowIcon(QIcon(pickaxe_icon_path))

        self.current_plotly_fig = None

        self._create_actions()
        self._create_menubar()
        self._init_ui()

        self.statusBar().showMessage("Visual Workshop ready. Load data to begin.")
        self._update_ui_for_data()

    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        self._create_controls_pane()
        self._create_plot_display_pane()

        self.splitter.addWidget(self.controls_widget)
        self.splitter.addWidget(self.plot_display_widget)
        self.splitter.setSizes([450, 950])

        self.statusBar().show()

    def _create_actions(self):
        doc_open_icon = QIcon(resource_path("document-open.png")) if os.path.exists(resource_path("document-open.png")) else QIcon.fromTheme("document-open")
        app_exit_icon = QIcon(resource_path("application-exit.png")) if os.path.exists(resource_path("application-exit.png")) else QIcon.fromTheme("application-exit")
        view_refresh_icon = QIcon(resource_path("view-refresh.png")) if os.path.exists(resource_path("view-refresh.png")) else QIcon.fromTheme("view-refresh")

        self.open_file_action = QAction(doc_open_icon, "&Open File Directly...", self)
        self.open_file_action.triggered.connect(self.open_file_directly)

        self.exit_action = QAction(app_exit_icon, "&Exit", self)
        self.exit_action.triggered.connect(self.close)

        self.refresh_data_action = QAction(view_refresh_icon, "&Refresh Data from Pickaxe", self)
        self.refresh_data_action.triggered.connect(self.refresh_data_from_pickaxe)
        self.refresh_data_action.setEnabled(self.source_app is not None)

    def _create_menubar(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.open_file_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        data_menu = menubar.addMenu("&Data")
        data_menu.addAction(self.refresh_data_action)

    def _create_controls_pane(self):
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(10,10,10,10)

        data_info_hbox = QHBoxLayout()
        print_vw_filename = os.path.basename(self.current_filename_hint)
        if len(print_vw_filename) > 30:
            print_vw_filename = print_vw_filename[:30] + "..."
        self.data_info_label = QLabel(f"<b>Source:</b> {print_vw_filename}\nDimensions: N/A")
        self.data_info_label.setWordWrap(True)
        self.data_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        data_info_hbox.addWidget(self.data_info_label, 1)

        self.plot_data_info_label = QLabel("<b>No. of Rows used for Plotting:</b> N/A")
        self.plot_data_info_label.setWordWrap(True)
        self.plot_data_info_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        data_info_hbox.addWidget(self.plot_data_info_label, 1)

        controls_layout.addLayout(data_info_hbox)
        controls_layout.addSpacing(10)

        sampling_group = QGroupBox("Data Sampling")
        sampling_layout = QVBoxLayout(sampling_group)
        self.sample_all_rb = QRadioButton("Use all data")
        self.sample_random_rb = QRadioButton("Random sample (max 10,000 rows)")
        self.sample_representative_rb = QRadioButton("Representative sample")
        self.sample_all_rb.setChecked(True)
        self.sampling_button_group = QButtonGroup(self)
        self.sampling_button_group.addButton(self.sample_all_rb)
        self.sampling_button_group.addButton(self.sample_random_rb)
        self.sampling_button_group.addButton(self.sample_representative_rb)
        sampling_layout.addWidget(self.sample_all_rb)
        sampling_layout.addWidget(self.sample_random_rb)
        sampling_layout.addWidget(self.sample_representative_rb)
        self.representative_sample_inputs_widget = QWidget()
        representative_sample_layout = QGridLayout(self.representative_sample_inputs_widget)
        representative_sample_layout.setContentsMargins(0,5,0,0)
        self.ci_label = QLabel("Confidence Level (%):")
        self.ci_combo = QComboBox()
        self.ci_combo.addItems(["90", "95", "99"])
        self.ci_combo.setCurrentText("95")
        representative_sample_layout.addWidget(self.ci_label, 0, 0)
        representative_sample_layout.addWidget(self.ci_combo, 0, 1)
        self.margin_error_label_rep = QLabel("Margin of Error (%):")
        self.margin_error_input_rep = QSpinBox()
        self.margin_error_input_rep.setRange(1, 50)
        self.margin_error_input_rep.setValue(5)
        self.margin_error_input_rep.setSuffix("%")
        representative_sample_layout.addWidget(self.margin_error_label_rep, 1, 0)
        representative_sample_layout.addWidget(self.margin_error_input_rep, 1, 1)
        self.sampling_note_label = QLabel(
            "<i><small>Note: Margin of Error (E) and Confidence Level (CI) are inputs that "
            "together determine the required sample size (n).</small></i>"
        )
        self.sampling_note_label.setWordWrap(True)
        self.sampling_note_label.setStyleSheet("QLabel { color: grey; }")
        representative_sample_layout.addWidget(self.sampling_note_label, 2, 0, 1, 2)
        sampling_layout.addWidget(self.representative_sample_inputs_widget)
        self.sample_representative_rb.toggled.connect(self.representative_sample_inputs_widget.setVisible)
        self.representative_sample_inputs_widget.setVisible(self.sample_representative_rb.isChecked())
        controls_layout.addWidget(sampling_group)

        plot_type_group = QGroupBox("Plot Type")
        plot_type_layout = QVBoxLayout(plot_type_group)
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems([
            "Scatter Plot", "Line Plot", "Bar Chart", "Pie Chart",
            "Histogram", "Box Plot", "Violin Plot", "Strip Plot",
            "Density Contour", "Distribution Plot (Distplot)"
        ])
        self.plot_type_combo.currentTextChanged.connect(self._update_plot_config_ui)
        plot_type_layout.addWidget(self.plot_type_combo)
        controls_layout.addWidget(plot_type_group)

        self.config_scroll_area = QScrollArea()
        self.config_scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.config_layout = QVBoxLayout(self.scroll_widget)
        self.config_scroll_area.setWidget(self.scroll_widget)
        controls_layout.addWidget(self.config_scroll_area, 1)

        self.generate_plot_button = QPushButton("Generate Plot")
        self.generate_plot_button.setDefault(True)
        self.generate_plot_button.clicked.connect(self.generate_plot)
        self.generate_plot_button.setFixedHeight(40)
        controls_layout.addWidget(self.generate_plot_button)

        self.save_plot_button = QPushButton("Save Plot")
        self.save_plot_button.clicked.connect(self.save_current_plot)
        self.save_plot_button.setEnabled(False)
        self.save_plot_button.setFixedHeight(30)
        controls_layout.addWidget(self.save_plot_button)

        self._update_plot_config_ui(self.plot_type_combo.currentText())

    def _create_plot_display_pane(self):
        self.plot_display_widget = QWidget()
        plot_display_layout = QVBoxLayout(self.plot_display_widget)
        
        # Create a QWebEnginePage that uses our custom profile
        page = QWebEnginePage(self.profile, self)

        # Create the QWebEngineView using this page
        self.plot_view = QWebEngineView(self)
        self.plot_view.setPage(page)
        
        self.plot_view.settings().setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
        self.plot_view.settings().setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)
        self.plot_view.setHtml("<div style='display:flex;justify-content:center;align-items:center;height:100%;font-family:sans-serif;color:grey;'>Load data and configure a plot to display.</div>")
        plot_display_layout.addWidget(self.plot_view)

    def _update_plot_config_ui(self, plot_type_name):
        # Clear existing advanced groups and widgets
        for group_attr in ['advanced_group', 'distplot_advanced_group', 'pie_advanced_group']:
            group = getattr(self, group_attr, None)
            if group:
                if self.config_layout.indexOf(group) != -1:
                    self.config_layout.removeWidget(group)
                group.deleteLater()
                setattr(self, group_attr, None)

        # Clear basic config widgets from layout
        current_widgets = []
        for i in range(self.config_layout.count()):
            item = self.config_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), QGroupBox) and "Basic" in item.widget().title():
                current_widgets.append(item.widget())
        for widget in current_widgets:
            self.config_layout.removeWidget(widget)
            widget.deleteLater()

        self.basic_config_widgets.clear()
        self.advanced_config_widgets.clear()
        self.distplot_advanced_widgets.clear()
        self.pie_advanced_widgets.clear()
        self.hover_data_configs.clear()

        basic_group = QGroupBox(f"Basic {plot_type_name} Configuration")
        basic_layout = QVBoxLayout(basic_group)
        self._add_basic_config_widgets(plot_type_name, basic_layout)
        self.config_layout.addWidget(basic_group)

        if plot_type_name == "Distribution Plot (Distplot)":
            self.distplot_advanced_group = CollapsibleGroupBox("Advanced Distplot Configuration")
            self._add_distplot_advanced_config(self.distplot_advanced_group.get_content_layout())
            self.config_layout.addWidget(self.distplot_advanced_group)
        elif plot_type_name == "Pie Chart":
            self.pie_advanced_group = CollapsibleGroupBox("Advanced Pie Chart Configuration")
            self._add_pie_advanced_config(self.pie_advanced_group.get_content_layout())
            self.config_layout.addWidget(self.pie_advanced_group)
        else:
            self.advanced_group = CollapsibleGroupBox("Advanced Plot Configuration")
            self._add_advanced_config_widgets(self.advanced_group.get_content_layout(), plot_type_name)
            self.config_layout.addWidget(self.advanced_group)

        self.config_layout.addStretch()
        self._auto_populate_color_discrete_map()


    def _add_basic_config_widgets(self, plot_type, layout):
        widget_adders = {
            "Scatter Plot": self._add_scatter_config,
            "Line Plot": self._add_line_config,
            "Bar Chart": self._add_bar_config,
            "Pie Chart": self._add_pie_config,
            "Histogram": self._add_histogram_config,
            "Box Plot": self._add_box_config,
            "Violin Plot": self._add_violin_config,
            "Strip Plot": self._add_strip_config,
            "Density Contour": self._add_density_contour_config,
            "Distribution Plot (Distplot)": self._add_distplot_config,
        }
        if plot_type in widget_adders:
            widget_adders[plot_type](layout)

        # Connect color_col changes for auto-populating color discrete map
        if "color_col" in self.basic_config_widgets:
            color_combo = self.basic_config_widgets["color_col"]
            if color_combo:
                # FIX: Removed the `try...except` for disconnecting.
                # Qt's `deleteLater()` on the parent widget handles connection cleanup.
                color_combo.currentTextChanged.connect(self._on_basic_color_col_changed_for_map)

    @Slot(str)
    def _on_basic_color_col_changed_for_map(self, _text):
        self._auto_populate_color_discrete_map()

    def _auto_populate_color_discrete_map(self):
        if self.current_df is None or self.current_df.is_empty():
            return

        color_col_widget = self.basic_config_widgets.get("color_col")
        color_discrete_map_edit = self.advanced_config_widgets.get("color_discrete_map_edit")

        if not color_col_widget or not color_discrete_map_edit:
            return # Widgets not present for current plot type or state

        color_col_name = color_col_widget.currentText()
        if color_col_name == "None" or color_col_name not in self.current_df.columns:
            # color_discrete_map_edit.setText("") # Clear if no valid color col
            return

        col_dtype = self.current_df.schema[color_col_name]
        if col_dtype in [pl.Utf8, pl.Categorical, pl.Boolean]: # Categorical type
            try:
                unique_values = self.current_df.get_column(color_col_name).drop_nulls().unique().sort().to_list()
                if not unique_values:
                    color_discrete_map_edit.setText("")
                    return

                # Check if current text is already a valid map for these keys
                # This prevents overwriting user's custom colors if keys match
                try:
                    current_map_text = color_discrete_map_edit.text()
                    if current_map_text:
                        existing_map = literal_eval(current_map_text)
                        if isinstance(existing_map, dict) and set(existing_map.keys()) == set(str(uv) for uv in unique_values):
                            return # User's map already matches keys, don't overwrite
                except:
                    pass # Invalid existing text, proceed to generate

                color_map = {}
                palette = px.colors.qualitative.Plotly # Or any other preferred palette
                for i, value in enumerate(unique_values):
                    color_map[str(value)] = palette[i % len(palette)]
                
                # Format as string: {'CategoryA': 'blue', 'CategoryB': 'red'}
                map_str = "{"+", ".join([f"'{k}': '{v}'" for k, v in color_map.items()])+"}"
                color_discrete_map_edit.setText(map_str)

            except Exception as e:
                print(f"Error auto-populating color discrete map: {e}")
                # color_discrete_map_edit.setText("") # Clear on error
        else:
            # color_discrete_map_edit.setText("") # Not a categorical column
            pass


    def _add_widget_pair(self, label_text, widget, layout, storage_dict, storage_key, help_text=None):
        row_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(120) # Standardized label width
        row_layout.addWidget(label)
        row_layout.addWidget(widget, 1) # Widget takes remaining space

        if help_text:
            help_button = QToolButton()
            help_button.setText("?")
            help_button.setFixedSize(20, 20)
            help_button.clicked.connect(lambda: QMessageBox.information(self, f"Help: {label_text.strip(':')}", help_text))
            row_layout.addWidget(help_button)

        layout.addLayout(row_layout)
        storage_dict[storage_key] = widget
        return widget


    def _populate_column_combobox(self, combobox, include_none=True, data_types=None, default_selection_hint=None):
        combobox.clear()
        available_cols = []
        ordered_columns_to_add = []
        if self.current_df is not None:
            available_cols = self._get_columns_by_type(data_types, self.current_df)
            original_df_columns = self.current_df.columns
            ordered_columns_to_add = [col for col in original_df_columns if col in available_cols]

        if include_none:
            combobox.addItem("None")

        for col_name in ordered_columns_to_add:
            combobox.addItem(col_name)

        selected_idx = -1
        if default_selection_hint and default_selection_hint in ordered_columns_to_add:
            selected_idx = combobox.findText(default_selection_hint)
        elif not include_none:
            sensible_defaults = [col for col in ordered_columns_to_add if col != "__original_index__"]
            if sensible_defaults:
                selected_idx = combobox.findText(sensible_defaults[0])
            elif ordered_columns_to_add:
                 selected_idx = combobox.findText(ordered_columns_to_add[0])
        elif include_none:
            selected_idx = combobox.findText("None")
            if selected_idx == -1 and combobox.count() > 0: selected_idx = 0

        if selected_idx != -1:
            combobox.setCurrentIndex(selected_idx)
        elif combobox.count() > 0 :
             combobox.setCurrentIndex(0)
        elif not include_none:
            combobox.addItem("N/A")
            combobox.setCurrentIndex(0)


    def _get_columns_by_type(self, data_types_filter, df_source):
        if df_source is None: return []

        int_types_pl = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]
        float_types_pl = [pl.Float32, pl.Float64]
        numeric_types_pl = int_types_pl + float_types_pl
        temporal_types_pl = [pl.Date, pl.Datetime, pl.Duration, pl.Time]
        string_types_pl = [pl.Utf8]
        categorical_types_pl = [pl.Categorical]
        boolean_types_pl = [pl.Boolean]

        all_types_pl = numeric_types_pl + temporal_types_pl + string_types_pl + categorical_types_pl + boolean_types_pl

        if data_types_filter is None or 'all' in data_types_filter:
            return [col for col in df_source.columns if col != "__original_index__"]

        selected_columns = []
        for col_name in df_source.columns:
            if col_name == "__original_index__": continue

            col_type = df_source.schema[col_name]
            match = False
            if 'numeric' in data_types_filter and col_type in numeric_types_pl: match = True
            if 'integer' in data_types_filter and col_type in int_types_pl: match = True
            if 'float' in data_types_filter and col_type in float_types_pl: match = True
            if 'temporal' in data_types_filter and col_type in temporal_types_pl: match = True
            if 'categorical' in data_types_filter and \
               (col_type in string_types_pl or col_type in categorical_types_pl or col_type in boolean_types_pl):
                match = True
            if 'string' in data_types_filter and col_type in string_types_pl : match = True
            if 'numeric_temporal' in data_types_filter and (col_type in numeric_types_pl or col_type in temporal_types_pl): match = True
            if 'sortable' in data_types_filter and (col_type in all_types_pl) : match = True # All pl types are generally sortable

            if match: selected_columns.append(col_name)
        return selected_columns

    def _add_scatter_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Color by:", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        self._populate_column_combobox(self._add_widget_pair("Size by:", QComboBox(), layout, self.basic_config_widgets, "size_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Symbol by:", QComboBox(), layout, self.basic_config_widgets, "symbol_col"), data_types=["categorical"])

    def _add_line_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["sortable"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by:", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        self._populate_column_combobox(self._add_widget_pair("Symbol by:", QComboBox(), layout, self.basic_config_widgets, "symbol_col"), data_types=["categorical"])
        cb = self._add_widget_pair("Sort X-axis:", QCheckBox(), layout, self.basic_config_widgets, "sort_x")
        cb.setChecked(True)

    def _add_bar_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["categorical", "all"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by:", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])

        self.bar_mode_label = QLabel("Bar Mode:")
        self.bar_mode_label.setFixedWidth(120)
        self.bar_mode_combo = QComboBox()
        self.bar_mode_combo.addItems(["Group", "Stack", "Relative"])
        self.basic_config_widgets["barmode"] = self.bar_mode_combo
        self.bar_mode_row_layout = QHBoxLayout()
        self.bar_mode_row_layout.addWidget(self.bar_mode_label)
        self.bar_mode_row_layout.addWidget(self.bar_mode_combo)
        layout.addLayout(self.bar_mode_row_layout)
        self.basic_config_widgets["color_col"].currentTextChanged.connect(self._on_bar_color_by_changed)
        self._on_bar_color_by_changed(self.basic_config_widgets["color_col"].currentText())

        self._populate_column_combobox(self._add_widget_pair("Pattern Shape by:", QComboBox(), layout, self.basic_config_widgets, "pattern_col"), data_types=["categorical"])
        agg_combo = self._add_widget_pair("Aggregation:", QComboBox(), layout, self.basic_config_widgets, "agg_func")
        agg_combo.addItems(["sum", "mean", "median", "min", "max", "count", "first", "last"]) # Added first, last
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])

    def _on_bar_color_by_changed(self, color_col_name):
        is_categorical_color = False
        if self.current_df is not None and not self.current_df.is_empty() and color_col_name != "None" and color_col_name in self.current_df.columns:
            dtype = self.current_df.schema[color_col_name]
            if dtype in [pl.Utf8, pl.Categorical, pl.Boolean]:
                is_categorical_color = True
        self.bar_mode_label.setVisible(is_categorical_color)
        self.bar_mode_combo.setVisible(is_categorical_color)


    def _add_pie_config(self, layout):
        self._populate_column_combobox(
            self._add_widget_pair("Names/Labels:", QComboBox(), layout, self.basic_config_widgets, "names_col"),
            include_none=False, data_types=["categorical"]
        )
        mode_combo = self._add_widget_pair("Mode:", QComboBox(), layout, self.basic_config_widgets, "pie_mode")
        mode_combo.addItems(["Count Occurrences (Rows)", "Sum Values from Column"])
        mode_combo.currentTextChanged.connect(self._on_pie_mode_changed)
        self.pie_values_col_label = QLabel("Values Column:")
        self.pie_values_col_label.setFixedWidth(120)
        self.pie_values_col_widget = QComboBox()
        self._populate_column_combobox(self.pie_values_col_widget, include_none=False, data_types=["numeric"])
        self.basic_config_widgets["values_col"] = self.pie_values_col_widget
        self.pie_values_row_layout = QHBoxLayout()
        self.pie_values_row_layout.addWidget(self.pie_values_col_label)
        self.pie_values_row_layout.addWidget(self.pie_values_col_widget)
        layout.addLayout(self.pie_values_row_layout)
        self._on_pie_mode_changed(mode_combo.currentText())
        hole_spin = self._add_widget_pair("Hole Size (%):", QSpinBox(), layout, self.basic_config_widgets, "hole_size")
        hole_spin.setRange(0,90)
        hole_spin.setValue(0)

    def _on_pie_mode_changed(self, mode_text):
        show_values_column = (mode_text == "Sum Values from Column")
        if hasattr(self, 'pie_values_col_label') and hasattr(self, 'pie_values_col_widget'):
            self.pie_values_col_label.setVisible(show_values_column)
            self.pie_values_col_widget.setVisible(show_values_column)

    def _add_histogram_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis/Weights:", QComboBox(), layout, self.basic_config_widgets, "y_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by:", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        self._populate_column_combobox(self._add_widget_pair("Pattern Shape by:", QComboBox(), layout, self.basic_config_widgets, "pattern_col"), data_types=["categorical"]) # Still present, though not used by go.Histogram
        bins_spin = self._add_widget_pair("Number of Bins:", QSpinBox(), layout, self.basic_config_widgets, "nbinsx")
        bins_spin.setRange(0, 1000); bins_spin.setValue(10)
        norm_combo = self._add_widget_pair("Normalization:", QComboBox(), layout, self.basic_config_widgets, "histnorm")
        norm_combo.addItems(["None", "Percent", "Probability", "Density", "Probability Density"])
        self._add_widget_pair("Cumulative:", QCheckBox(), layout, self.basic_config_widgets, "cumulative_enabled")

    def _add_box_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), data_types=["categorical", "temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by:", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        pts_combo = self._add_widget_pair("Points:", QComboBox(), layout, self.basic_config_widgets, "boxpoints")
        pts_combo.addItems(["Outliers", "Suspected Outliers", "All", "False"])
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])
        self._add_widget_pair("Notched:", QCheckBox(), layout, self.basic_config_widgets, "notched")

    def _add_violin_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), data_types=["categorical", "temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by:", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        self._populate_column_combobox(self._add_widget_pair("Split By:", QComboBox(), layout, self.basic_config_widgets, "split_by_col"), data_types=["categorical", "boolean"])
        self._add_widget_pair("Show Box:", QCheckBox(), layout, self.basic_config_widgets, "box_visible")
        pts_combo = self._add_widget_pair("Points:", QComboBox(), layout, self.basic_config_widgets, "points")
        pts_combo.addItems(["Outliers", "Suspected Outliers", "All", "False"])
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])

    def _add_strip_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), data_types=["categorical", "temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by:", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])

    def _add_density_contour_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Z-axis/Weights:", QComboBox(), layout, self.basic_config_widgets, "z_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color (Marginal):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"]) # Typically influences marginal plots
        mx_combo = self._add_widget_pair("Marginal X:", QComboBox(), layout, self.basic_config_widgets, "marginal_x")
        mx_combo.addItems(["None", "Histogram", "Rug", "Box", "Violin"])
        my_combo = self._add_widget_pair("Marginal Y:", QComboBox(), layout, self.basic_config_widgets, "marginal_y")
        my_combo.addItems(["None", "Histogram", "Rug", "Box", "Violin"])

    def _add_distplot_column_selector(self, parent_layout, storage_dict, storage_key):
        """
        Adds a collapsible group box with checkboxes for selecting distplot columns.
        """
        group_box = CollapsibleGroupBox("Data Columns for Distribution", checked=True) # Start expanded
        content_layout = group_box.get_content_layout()

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget)
        scroll_area.setMinimumHeight(80)  # Adjust as needed
        scroll_area.setMaximumHeight(200) # Adjust as needed

        checkbox_list = [] # This will be stored in storage_dict[storage_key]

        if self.current_df is not None and not self.current_df.is_empty():
            # Get numeric and temporal columns
            dist_cols = self._get_columns_by_type(["numeric_temporal"], self.current_df)
            if not dist_cols:
                scroll_layout.addWidget(QLabel("No suitable (numeric/temporal) columns found in the data."))
            else:
                for col_name in dist_cols:
                    if col_name == "__original_index__": # Skip internal columns
                        continue
                    cb = QCheckBox(col_name)
                    scroll_layout.addWidget(cb)
                    checkbox_list.append(cb)
        else:
            scroll_layout.addWidget(QLabel("No data loaded to select columns."))
        
        scroll_layout.addStretch(1) # Push checkboxes to the top
        content_layout.addWidget(scroll_area)
        parent_layout.addWidget(group_box)
        
        storage_dict[storage_key] = checkbox_list # Store the list of QCheckBox objects

    def _add_distplot_config(self, layout):
        # Replaces the old QListWidget with the new CollapsibleGroupBox selector
        self._add_distplot_column_selector(layout, self.basic_config_widgets, "hist_data_cols_list")
        # Existing widgets for distplot
        curve_combo = self._add_widget_pair("Curve Type:", QComboBox(), layout, self.basic_config_widgets, "curve_type")
        curve_combo.addItems(["kde", "normal", "None"])
        self._add_widget_pair("Show Histogram(s):", QCheckBox(), layout, self.basic_config_widgets, "show_hist").setChecked(True)
        self._add_widget_pair("Show Rug Plot(s):", QCheckBox(), layout, self.basic_config_widgets, "show_rug").setChecked(True)
        bin_edit = self._add_widget_pair("Bin Size/Count:", QLineEdit(), layout, self.basic_config_widgets, "bin_size")
        bin_edit.setPlaceholderText("e.g., 0.1 (width) or 100 (count) or empty")

    def _add_hover_data_config_section(self, layout, storage_dict_key_prefix=""):
        # Hover Data Configuration
        hover_group = CollapsibleGroupBox("Hover Data Configuration")
        hover_group.setChecked(False) # Default to collapsed
        hover_content_layout = hover_group.get_content_layout()

        # Add a scroll area within the hover group for potentially many columns
        hover_scroll_area = QScrollArea()
        hover_scroll_area.setWidgetResizable(True)
        hover_scroll_widget = QWidget()
        hover_scroll_layout = QVBoxLayout(hover_scroll_widget)
        hover_scroll_area.setWidget(hover_scroll_widget)
        hover_scroll_area.setMinimumHeight(100) # Ensure it has some initial height
        hover_scroll_area.setMaximumHeight(200) # Prevent it from taking too much space

        self.hover_data_configs.clear() # Clear previous configs

        if self.current_df is not None:
            for col_name in self.current_df.columns:
                if col_name == "__original_index__": continue

                row_layout = QHBoxLayout()
                checkbox = QCheckBox(col_name)
                lineedit = QLineEdit()
                lineedit.setPlaceholderText(f"Display as: {col_name}")
                lineedit.setEnabled(False) # Enable only if checkbox is checked
                checkbox.toggled.connect(lineedit.setEnabled)

                row_layout.addWidget(checkbox)
                row_layout.addWidget(lineedit)
                hover_scroll_layout.addLayout(row_layout)
                self.hover_data_configs[col_name] = {'checkbox': checkbox, 'lineedit': lineedit}
        else:
            hover_scroll_layout.addWidget(QLabel("No data loaded to select hover columns."))
        
        hover_scroll_layout.addStretch() # Push items to the top
        hover_content_layout.addWidget(hover_scroll_area)
        layout.addWidget(hover_group)


    def _add_advanced_config_widgets(self, layout, plot_type_name): # plot_type_name for context
        self.advanced_config_widgets.clear() # Clear before adding

        self._add_widget_pair("Plot Title:", QLineEdit(), layout, self.advanced_config_widgets, "plot_title_edit")
        self._add_widget_pair("Subtitle (Markdown):", QTextEdit(), layout, self.advanced_config_widgets, "subtitle_edit").setFixedHeight(60)

        self._add_widget_pair("X-axis Label:", QLineEdit(), layout, self.advanced_config_widgets, "xaxis_label_edit")
        self._add_widget_pair("Y-axis Label:", QLineEdit(), layout, self.advanced_config_widgets, "yaxis_label_edit")
        self._add_widget_pair("Legend/Colorbar Title:", QLineEdit(), layout, self.advanced_config_widgets, "legend_title_edit")

        # Hover data configuration section
        self._add_hover_data_config_section(layout, "adv_hover_") # Prefix to avoid clashes if needed

        self._add_widget_pair("Log X-axis:", QCheckBox(), layout, self.advanced_config_widgets, "log_x_check")
        self._add_widget_pair("Log Y-axis:", QCheckBox(), layout, self.advanced_config_widgets, "log_y_check")

        self._add_widget_pair("X-axis Range:", QLineEdit(), layout, self.advanced_config_widgets, "xaxis_range_edit").setPlaceholderText("min,max")
        self._add_widget_pair("Y-axis Range:", QLineEdit(), layout, self.advanced_config_widgets, "yaxis_range_edit").setPlaceholderText("min,max")

        color_map_help = (
            "Define custom colors for categorical values.\n"
            "Format: {'CategoryValue1': 'color1', 'CategoryValue2': 'color2', ...}\n"
            "Example: {'Alpha': 'red', 'Beta': '#00FF00', 'Gamma': 'rgb(0,0,255)'}\n"
            "Accepted color formats: named (e.g., 'blue'), hex (e.g., '#FF5733'), rgb (e.g., 'rgb(255,87,51)')."
        )
        self._add_widget_pair("Color Discrete Map:", QLineEdit(), layout, self.advanced_config_widgets, "color_discrete_map_edit", help_text=color_map_help).setPlaceholderText("{'ValA':'red', ...}")

        csc_combo = self._add_widget_pair("Color Cont. Scale:", QComboBox(), layout, self.advanced_config_widgets, "color_continuous_scale_combo")
        csc_combo.addItems(["None"] + px.colors.named_colorscales())

        sym_combo = self._add_widget_pair("Global Marker Symbol:", QComboBox(), layout, self.advanced_config_widgets, "marker_symbol_combo")
        sym_combo.addItems(["None", "circle", "square", "diamond", "cross", "x", "triangle-up", "star"])

        tpl_combo = self._add_widget_pair("Plot Template:", QComboBox(), layout, self.advanced_config_widgets, "template_combo")
        tpl_combo.addItems(["None", "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"])

        if plot_type_name not in ["Pie Chart", "Distribution Plot (Distplot)"]: # Faceting not for these
            self._populate_column_combobox(self._add_widget_pair("Facet Row:", QComboBox(), layout, self.advanced_config_widgets, "facet_row_combo"), data_types=["categorical"])
            self._populate_column_combobox(self._add_widget_pair("Facet Col:", QComboBox(), layout, self.advanced_config_widgets, "facet_col_combo"), data_types=["categorical"])

    def _add_pie_advanced_config(self, layout): # For Pie Chart
        self.pie_advanced_widgets.clear()
        self._add_widget_pair("Plot Title:", QLineEdit(), layout, self.pie_advanced_widgets, "plot_title_edit")
        self._add_widget_pair("Subtitle (Markdown):", QTextEdit(), layout, self.pie_advanced_widgets, "subtitle_edit").setFixedHeight(60)
        self._add_widget_pair("Legend Title:", QLineEdit(), layout, self.pie_advanced_widgets, "legend_title_edit")
        
        # Pie charts have specific hover, so general hover config might be less relevant or need specific handling.
        # For now, let's keep it simple and not add the generic hover config section here.
        # If hover is needed for pie, it would usually be on the aggregated data (names, values, percent).

        self._add_widget_pair("Pull (CSV):", QLineEdit(), layout, self.pie_advanced_widgets, "pull_edit").setPlaceholderText("e.g., 0,0,0.2,0")
        self._add_widget_pair("Slice Colors (CSV):", QLineEdit(), layout, self.pie_advanced_widgets, "slice_colors_edit").setPlaceholderText("red,#00FF00,blue,...")
        self._add_widget_pair("Slice Line Color:", QLineEdit(), layout, self.pie_advanced_widgets, "slice_line_color_edit").setPlaceholderText("e.g., black or #333333")
        slice_line_width_spin = self._add_widget_pair("Slice Line Width:", QSpinBox(), layout, self.pie_advanced_widgets, "slice_line_width_spin")
        slice_line_width_spin.setRange(0,10); slice_line_width_spin.setValue(1)
        tpl_combo = self._add_widget_pair("Plot Template:", QComboBox(), layout, self.pie_advanced_widgets, "template_combo")
        tpl_combo.addItems(["None", "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"])

    def _add_distplot_advanced_config(self, layout): # Specific advanced for Distplot
        self.distplot_advanced_widgets.clear()
        self._add_widget_pair("Plot Title:", QLineEdit(), layout, self.distplot_advanced_widgets, "plot_title_edit")
        self._add_widget_pair("Subtitle (Markdown):", QTextEdit(), layout, self.distplot_advanced_widgets, "subtitle_edit").setFixedHeight(60)
        self._add_widget_pair("X-axis Label:", QLineEdit(), layout, self.distplot_advanced_widgets, "xaxis_label_edit")
        self._add_widget_pair("Y-axis Label:", QLineEdit(), layout, self.distplot_advanced_widgets, "yaxis_label_edit")
        self._add_widget_pair("Legend Title:", QLineEdit(), layout, self.distplot_advanced_widgets, "legend_title_edit")
        self._add_widget_pair("Colors (CSV):", QLineEdit(), layout, self.distplot_advanced_widgets, "distplot_colors").setPlaceholderText("#FF0000,#00FF00,...")
        tpl_combo = self._add_widget_pair("Plot Template:", QComboBox(), layout, self.distplot_advanced_widgets, "template_combo")
        tpl_combo.addItems(["None", "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"])


    def receive_dataframe(self, polars_df, filename_hint, log_file_path_from_source=None):
        if not isinstance(polars_df, pl.DataFrame):
            QMessageBox.warning(self, "Data Error", "Invalid data type received. Expected Polars DataFrame.")
            return

        self.current_df = polars_df
        self.current_filename_hint = filename_hint

        if log_file_path_from_source:
            self.current_log_file_path = log_file_path_from_source
            logger.set_log_file(self.current_log_file_path, "VisualWorkshop (Continuing Session)", associated_data_file=filename_hint)
        elif not self.current_log_file_path:
            base_name = os.path.splitext(os.path.basename(filename_hint if filename_hint else "vw_data"))[0]
            timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.current_log_file_path = os.path.join(os.getcwd(), f".__log__{base_name}_{timestamp_str}_vw.md")
            logger.set_log_file(self.current_log_file_path, "VisualWorkshop", associated_data_file=filename_hint)
        else:
            logger.set_log_file(self.current_log_file_path, "VisualWorkshop", associated_data_file=filename_hint)

        logger.log_action(
            "VisualWorkshop", "Data Received",
            f"Received data '{os.path.basename(filename_hint)}'",
            details={
                "Source Hint": filename_hint, "Rows": polars_df.height, "Columns": polars_df.width,
                "Log File In Use": self.current_log_file_path
            }
        )

        self.statusBar().showMessage(f"Data received: {filename_hint}. Dimensions: {self.current_df.shape}")
        print_vw_filename = os.path.basename(filename_hint)
        if len(print_vw_filename) > 30:
            print_vw_filename = print_vw_filename[:30] + "..."
        self.data_info_label.setText(f"<b>Source:</b> {print_vw_filename}\nDimensions: {self.current_df.height} rows, {self.current_df.width} cols")
        self._update_ui_for_data()
        self._update_plot_config_ui(self.plot_type_combo.currentText()) # This will rebuild UI and repopulate hover config

    def refresh_data_from_pickaxe(self):
        if self.source_app and hasattr(self.source_app, 'get_current_dataframe_for_vw') and \
           hasattr(self.source_app, 'get_current_filename_hint_for_vw'):
            df = self.source_app.get_current_dataframe_for_vw()
            hint = self.source_app.get_current_filename_hint_for_vw()
            log_path = None
            if hasattr(self.source_app, 'get_current_log_file_path'): # If pickaxe exposes this
                 log_path = self.source_app.get_current_log_file_path()

            if df is not None:
                self.receive_dataframe(df, hint, log_file_path_from_source=log_path)
                self.statusBar().showMessage("Data refreshed from Pickaxe.", 3000)
            else:
                QMessageBox.information(self, "No Data", f"{self.source_app.windowTitle() if hasattr(self.source_app, 'windowTitle') else 'Source App'} has no active data to refresh.")
        else:
            QMessageBox.warning(self, "Integration Error", "Cannot refresh. Source application instance not available or methods missing.")


    def open_file_directly(self):
        df_loaded = None
        file_name_loaded = None

        if not PICKAXE_LOADERS_AVAILABLE:
            file_name, _ = QFileDialog.getOpenFileName(self, "Open File", None, "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet)")
            if file_name:
                file_name_loaded = file_name
                try:
                    if file_name.lower().endswith('.csv'):
                        df_loaded = pl.read_csv(file_name)
                    elif file_name.lower().endswith(('.xlsx', '.xls')):
                        df_loaded = pl.read_excel(file_name, infer_schema_length=0)
                    elif file_name.lower().endswith('.parquet'):
                        df_loaded = pl.read_parquet(file_name)
                    else:
                        QMessageBox.warning(self, "Unsupported File", "Only CSV, Excel, and Parquet files are supported in this mode.")
                        return
                except Exception as e:
                    QMessageBox.critical(self, "File Load Error", f"Could not load file: {str(e)}")
                    return
        else:
             QMessageBox.information(self, "Open File", "Full Pickaxe file loading not implemented in this version. Using basic loader.")
             file_name, _ = QFileDialog.getOpenFileName(self, "Open File", None, "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet)")
             if file_name:
                file_name_loaded = file_name
                try:
                    if file_name.lower().endswith('.csv'):
                        df_loaded = pl.read_csv(file_name)
                    elif file_name.lower().endswith(('.xlsx', '.xls')):
                        df_loaded = pl.read_excel(file_name, infer_schema_length=0)
                    elif file_name.lower().endswith('.parquet'):
                        df_loaded = pl.read_parquet(file_name)
                except Exception as e:
                    QMessageBox.critical(self, "File Load Error", f"Could not load file: {str(e)}")
                    return

        if df_loaded is not None and file_name_loaded:
            try:
                log_file_dir = os.path.dirname(os.path.abspath(file_name_loaded))
                data_basename_no_ext = os.path.splitext(os.path.basename(file_name_loaded))[0]
                timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                log_filename_for_data = f".__log__{data_basename_no_ext}_{timestamp_str}.md"
                self.current_log_file_path = os.path.join(log_file_dir, log_filename_for_data)
                logger.set_log_file(self.current_log_file_path, "VisualWorkshop (Direct Open)", associated_data_file=file_name_loaded)
                logger.log_action("VisualWorkshop", "Data Session Start", f"Processing data file: {os.path.basename(file_name_loaded)}",
                                details={"Log file": self.current_log_file_path})
            except Exception as e:
                print(f"VW: Error setting up logger for direct open {file_name_loaded}: {e}")
                default_log_name = f".__log__vw_fallback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                self.current_log_file_path = os.path.join(os.getcwd(), default_log_name)
                logger.set_log_file(self.current_log_file_path, "VisualWorkshop (Fallback Log)", associated_data_file=file_name_loaded)
            
            self.receive_dataframe(df_loaded, file_name_loaded)
        elif file_name_loaded and df_loaded is None:
            pass


    def _update_ui_for_data(self):
        
        has_data = self.current_df is not None and not self.current_df.is_empty()
        self.generate_plot_button.setEnabled(has_data)
        if hasattr(self, 'hover_data_configs') and self.hover_data_configs: 
            plot_type = self.plot_type_combo.currentText()
            self._update_plot_config_ui(plot_type)


    def _calculate_representative_sample_size(self, df, target_column_name, margin_error_percent=5, confidence_level_percent=95):
        
        if target_column_name not in df.columns:
            self.statusBar().showMessage(f"Target column '{target_column_name}' for sampling not found. Using max 10k rows.", 3000)
            return min(10000, df.height)

        series = df.get_column(target_column_name).drop_nulls()
        if series.is_empty():
            self.statusBar().showMessage(f"Target column '{target_column_name}' is empty. Using max 10k rows for sampling.", 3000)
            return min(10000, df.height)

        n_total = df.height
        if n_total == 0: return 0

        confidence_level = confidence_level_percent / 100.0
        Z = 1.96 
        try:
            Z = norm.ppf(1 - (1 - confidence_level) / 2)
        except ImportError:
            self.statusBar().showMessage("Scipy not found, Z-score may not match selected confidence. Using 1.96 for 95% Confidence.", 2000)
            if confidence_level == 0.90: Z = 1.645
            elif confidence_level == 0.99: Z = 2.576
        except Exception: 
            self.statusBar().showMessage("Scipy.stats.norm not available, Z-score may not match. Using 1.96 for 95% Confidence.", 2000)


        sigma_or_p_estimate = 0.5
        dtype = series.dtype
        margin_error_prop = margin_error_percent / 100.0
        E = margin_error_prop

        if dtype.is_numeric():
            std_dev = series.std()
            if std_dev is not None and std_dev > 0:
                sigma_or_p_estimate = std_dev
                mean_val = series.mean()
                if mean_val is not None and abs(mean_val) > 1e-9:
                    E = margin_error_prop * abs(mean_val)
                else:
                    data_range = series.max() - series.min()
                    if data_range is not None and data_range > 1e-9:
                        E = margin_error_prop * data_range
                    else:
                         self.statusBar().showMessage(f"Cannot determine sensible margin of error for numeric '{target_column_name}'. Using high variability.", 3000)
                         E = sigma_or_p_estimate * 0.1
                if E == 0: E = 1e-6
            else:
                self.statusBar().showMessage(f"No variability in numeric column '{target_column_name}'. Cannot calculate representative sample meaningfully.", 3000)
                return n_total

        elif dtype == pl.Boolean or dtype == pl.Categorical or dtype == pl.Utf8:
            p_estimate = 0.5
            # sigma_or_p_estimate = np.sqrt(p_estimate * (1 - p_estimate)) # This is part of n0 for props
            E = margin_error_prop
        else: 
            p_estimate = 0.5
            # sigma_or_p_estimate = np.sqrt(p_estimate * (1 - p_estimate))
            E = margin_error_prop
        
        if E == 0: E = 1e-6

        try:
            if dtype.is_numeric():
                n0 = ((Z * sigma_or_p_estimate) / E)**2
            else: 
                n0 = (Z**2 * p_estimate * (1-p_estimate)) / (E**2)
            
            calculated_n = n0 / (1 + (n0 - 1) / n_total) if n_total > n0 else n0
            sample_s = max(1, min(int(np.ceil(calculated_n)), n_total))
            self.statusBar().showMessage(f"Calculated representative sample size: {sample_s} for '{target_column_name}' (CI: {confidence_level_percent}%, MoE: {margin_error_percent}%)", 5000)
            return sample_s
        except (OverflowError, ZeroDivisionError, ValueError) as e_calc:
            self.statusBar().showMessage(f"Error calculating sample size for '{target_column_name}': {e_calc}. Using random 10k.", 3000)
            return min(10000, n_total)


    @Slot()
    def generate_plot(self):
        # ... (Sampling and argument gathering logic is unchanged) ...
        if self.current_df is None or self.current_df.is_empty():
            QMessageBox.information(self, "No Data", "Please load data before generating a plot.")
            return

        self.statusBar().showMessage("Generating plot...", 2000)
        QApplication.processEvents()

        plot_type = self.plot_type_combo.currentText()
        df_for_plot_op = self.current_df

        df_to_plot = df_for_plot_op
        target_col_for_sampling = None
        sampling_method_log = "Use all data"

        if self.sample_random_rb.isChecked():
            if df_for_plot_op.height > 10000:
                sampling_method_log = "Random sample (max 10,000 rows)"
                self.statusBar().showMessage(f"Randomly sampling data from {df_for_plot_op.height} to 10,000 rows...", 3000)
                df_to_plot = df_for_plot_op.sample(n=10000, shuffle=True, seed=42)
                QApplication.processEvents()
        elif self.sample_representative_rb.isChecked():
            margin_error_val = self.margin_error_input_rep.value()
            confidence_level_val = int(self.ci_combo.currentText())
            sampling_method_log = f"Representative sample (CI: {confidence_level_val}%, MoE: {margin_error_val}%)"
            basic_args_temp = {k: self._get_widget_value(w) for k, w in self.basic_config_widgets.items()}
            target_col_for_sampling = None
            if plot_type in ["Scatter Plot", "Line Plot", "Bar Chart", "Box Plot", "Violin Plot", "Strip Plot", "Density Contour"]:
                target_col_for_sampling = basic_args_temp.get('y_col')
            elif plot_type == "Histogram": target_col_for_sampling = basic_args_temp.get('x_col')
            elif plot_type == "Pie Chart":
                target_col_for_sampling = basic_args_temp.get('values_col') if basic_args_temp.get('pie_mode') == "Sum Values from Column" else basic_args_temp.get('names_col')
            elif plot_type == "Distribution Plot (Distplot)":
                selected_dist_cols = basic_args_temp.get('hist_data_cols_list', [])
                if selected_dist_cols: target_col_for_sampling = selected_dist_cols[0]

            if target_col_for_sampling and target_col_for_sampling != "None" and target_col_for_sampling in df_for_plot_op.columns:
                calculated_sample_size = self._calculate_representative_sample_size(
                    df_for_plot_op, target_col_for_sampling, margin_error_val, confidence_level_val
                )
                proceed_with_large_sample = True
                if calculated_sample_size > 100000: 
                    reply = QMessageBox.warning(self, "Large Sample Size",
                                                f"The calculated sample size is {calculated_sample_size:,} rows. "
                                                "Plotting this may be very slow or consume a lot of memory.\n\n"
                                                "Do you want to proceed?",
                                                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.No:
                        proceed_with_large_sample = False
                        self.statusBar().showMessage("Plotting with large sample cancelled.", 3000)
                        return

                if proceed_with_large_sample:
                    if calculated_sample_size < df_for_plot_op.height:
                        df_to_plot = df_for_plot_op.sample(n=calculated_sample_size, shuffle=True, seed=42)
                        QApplication.processEvents()
            else:
                self.statusBar().showMessage("No valid target column for representative sampling. Using random 10k max if applicable.", 3000)
                if df_for_plot_op.height > 10000:
                    df_to_plot = df_for_plot_op.sample(n=10000, shuffle=True, seed=42)
        
        if df_to_plot is not None and not df_to_plot.is_empty():
            self.plot_data_info_label.setText(f"<b>No. of Rows used for Plotting:</b></br> {df_to_plot.height} <br/>")
        else:
            self.plot_data_info_label.setText("<b>No. of Rows used for Plotting:</b></br> N/A")
            QMessageBox.warning(self, "Sampling Error", "Data became empty after sampling. Cannot generate plot.")
            return

        basic_args = {k: self._get_widget_value(w) for k, w in self.basic_config_widgets.items()}
        log_details = {"Plot Type": plot_type, "Sampling Mode": self.sampling_button_group.checkedButton().text()}
        if self.sample_representative_rb.isChecked(): log_details["Margin of Error"] = self.margin_error_input_rep.value()
        log_details.update({f"Basic: {k}": v for k,v in basic_args.items() if v != "None" and v != "" and v is not False and not (isinstance(v, list) and not v)})
        log_details["Sampling Details"] = f"{sampling_method_log}, Plotting with: {df_to_plot.height} rows"

        advanced_args = {}
        hover_data_setting = []
        for col_name, config in self.hover_data_configs.items():
            if config['checkbox'].isChecked():
                custom_name = config['lineedit'].text().strip()
                hover_data_setting.append({'original_name': col_name, 'display_name': custom_name if custom_name else col_name})
        
        adv_config_source_dict = {}
        current_advanced_group = None
        if plot_type == "Distribution Plot (Distplot)":
            adv_config_source_dict = self.distplot_advanced_widgets
            current_advanced_group = self.distplot_advanced_group
        elif plot_type == "Pie Chart":
            adv_config_source_dict = self.pie_advanced_widgets
            current_advanced_group = self.pie_advanced_group
        else: 
            adv_config_source_dict = self.advanced_config_widgets
            current_advanced_group = self.advanced_group
        
        if current_advanced_group and current_advanced_group.isChecked():
            advanced_args = {k: self._get_widget_value(w) for k, w in adv_config_source_dict.items()}
        
        if plot_type not in ["Distribution Plot (Distplot)", "Pie Chart"]:
            advanced_args['hover_data_config'] = hover_data_setting
        if advanced_args:
            log_details.update({f"Adv: {k}": v for k,v in advanced_args.items() if v != "None" and v != "" and v is not False and not (isinstance(v, list) and not v) and k != 'hover_data_config'})
            if 'hover_data_config' in advanced_args:
                 log_details["Adv: Hover Columns"] = ", ".join([item['display_name'] for item in advanced_args['hover_data_config']])

        fig = None
        try:
            facet_row_col_name = advanced_args.get('facet_row_combo', "None")
            facet_col_col_name = advanced_args.get('facet_col_combo', "None")
            use_facets = (facet_row_col_name != "None" or facet_col_col_name != "None") and \
                         plot_type not in ["Pie Chart", "Distribution Plot (Distplot)"]

            if use_facets:
                row_cats = df_to_plot.get_column(facet_row_col_name).unique().drop_nulls().sort().to_list() \
                    if facet_row_col_name != "None" and facet_row_col_name in df_to_plot.columns else [None]
                col_cats = df_to_plot.get_column(facet_col_col_name).unique().drop_nulls().sort().to_list() \
                    if facet_col_col_name != "None" and facet_col_col_name in df_to_plot.columns else [None]

                subplot_titles = []
                valid_row_cats = row_cats != [None] and len(row_cats) > 0
                valid_col_cats = col_cats != [None] and len(col_cats) > 0

                if valid_row_cats and valid_col_cats:
                    subplot_titles = [f"{str(r_cat)}-{str(c_cat)}" for r_cat in row_cats for c_cat in col_cats]
                elif valid_row_cats:
                     subplot_titles = [str(r_cat) for r_cat in row_cats]
                elif valid_col_cats:
                     subplot_titles = [str(c_cat) for c_cat in col_cats]
                
                num_subplot_rows = len(row_cats) if row_cats != [None] else 1
                num_subplot_cols = len(col_cats) if col_cats != [None] else 1


                fig = sp.make_subplots(
                    rows=num_subplot_rows, cols=num_subplot_cols,
                    shared_xaxes=True, shared_yaxes=True,
                    subplot_titles=subplot_titles if subplot_titles else None
                )

                for r_idx, r_cat in enumerate(row_cats):
                    for c_idx, c_cat in enumerate(col_cats):
                        facet_df = df_to_plot
                        if r_cat is not None and facet_row_col_name in facet_df.columns: facet_df = facet_df.filter(pl.col(facet_row_col_name) == r_cat)
                        if c_cat is not None and facet_col_col_name in facet_df.columns: facet_df = facet_df.filter(pl.col(facet_col_col_name) == c_cat)

                        if facet_df.is_empty(): continue

                        self._add_traces_for_plot(fig, plot_type, facet_df, basic_args, advanced_args, r_idx + 1, c_idx + 1)
            else:
                fig = go.Figure()
                self._add_traces_for_plot(fig, plot_type, df_to_plot, basic_args, advanced_args)

            self._apply_advanced_layout(fig, advanced_args, basic_args, plot_type, df_to_plot)
            
            # --- MODIFICATION: Use the scheme handler ---
            # 1. Store the figure so the handler can access it.
            self.current_plotly_fig = fig
            
            # 2. Load the plot using our custom URL scheme.
            # The handler will intercept this and serve the plot from memory.
            plot_url = QUrl(f"{self.PLOT_SCHEME.decode()}://current_plot")
            self.plot_view.load(plot_url)
            # --- End of modification ---
            
            self.statusBar().showMessage("Plot generated successfully.", 3000)
            self.save_plot_button.setEnabled(True)

            logger.log_action(
                "VisualWorkshop", "Plot Generated", f"Generated {plot_type}.",
                details=log_details, df_shape_before=df_to_plot.shape
            )

        except Exception as e:
            QMessageBox.critical(self, "Plot Generation Error", f"An error occurred: {str(e)}")
            self.statusBar().showMessage(f"Error generating plot: {str(e)}", 5000)
            print(f"Plot Generation Error: {str(e)}\n{traceback.format_exc()}")
            self.plot_view.setHtml(f"<div style='color:red;padding:20px;'>Error generating plot: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>")
            self.current_plotly_fig = None
            self.save_plot_button.setEnabled(False)
            logger.log_action(
                "VisualWorkshop", "Plot Generation Error",
                f"Error generating {plot_type}.",
                details={"Error": str(e), "Plot Args": log_details}
            )

    def save_current_plot(self):
        if self.current_plotly_fig is None:
            QMessageBox.warning(self, "No Plot", "No plot has been generated yet to save.")
            return

        default_filename = f"{self.plot_type_combo.currentText().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        file_name, selected_filter = QFileDialog.getSaveFileName(
            self, 
            "Save Plot As",
            default_filename,
            "HTML File (*.html);;PNG Image (*.png);;JPEG Image (*.jpg);;SVG Image (*.svg);;PDF Document (*.pdf)"
        )

        if not file_name:
            return

        try:
            if self.current_log_file_path:
                logger.set_log_file(self.current_log_file_path, "VisualWorkshop")

            file_format = ""
            if selected_filter == "HTML File (*.html)":
                if not file_name.lower().endswith(".html"): file_name += ".html"
                self.current_plotly_fig.write_html(file_name)
                file_format = "HTML"
            elif selected_filter.endswith("(*.png)"):
                if not KALEIDO_AVAILABLE: raise Exception("Kaleido package is required for static image export.")
                if not file_name.lower().endswith(".png"): file_name += ".png"
                self.current_plotly_fig.write_image(file_name, scale=2)
                file_format = "PNG"
            elif selected_filter.endswith("(*.jpg)"):
                if not KALEIDO_AVAILABLE: raise Exception("Kaleido package is required for static image export.")
                if not file_name.lower().endswith((".jpg", ".jpeg")): file_name += ".jpg"
                self.current_plotly_fig.write_image(file_name, scale=2)
                file_format = "JPEG"
            elif selected_filter.endswith("(*.svg)"):
                if not KALEIDO_AVAILABLE: raise Exception("Kaleido package is required for static image export.")
                if not file_name.lower().endswith(".svg"): file_name += ".svg"
                self.current_plotly_fig.write_image(file_name, scale=2)
                file_format = "SVG"
            elif selected_filter.endswith("(*.pdf)"):
                if not KALEIDO_AVAILABLE: raise Exception("Kaleido package is required for static image export.")
                if not file_name.lower().endswith(".pdf"): file_name += ".pdf"
                self.current_plotly_fig.write_image(file_name)
                file_format = "PDF"
            else:
                QMessageBox.warning(self, "Unknown Format", "Selected file format is not supported.")
                return

            self.statusBar().showMessage(f"Plot saved as {file_format}: {file_name}", 3000)
            logger.log_action("VisualWorkshop", "Plot Saved", f"Plot saved as {file_format}: {os.path.basename(file_name)}",
                              details={"Path": file_name, "Format": file_format})

        except Exception as e:
            if "Kaleido" in str(e):
                 QMessageBox.critical(self, "Dependency Missing", 
                                     f"{str(e)}\nPlease install it (e.g., 'pip install kaleido').")
            else:
                QMessageBox.critical(self, "Save Plot Error", f"Could not save plot: {str(e)}")
            logger.log_action("VisualWorkshop", "Plot Save Error", f"Error saving plot: {os.path.basename(file_name)}",
                              details={"Error": str(e)})
            print(f"Error saving plot: {e}\n{traceback.format_exc()}")


    def _add_traces_for_plot(self, fig, plot_type, df, basic_args, advanced_args, row=None, col=None):
        hover_data_config = advanced_args.get('hover_data_config', [])

        customdata_numpy_array = None
        original_col_names_for_customdata = []

        if plot_type not in ["Bar Chart", "Pie Chart", "Distribution Plot (Distplot)"]:
            if hover_data_config and not df.is_empty():
                select_exprs = []
                for item in hover_data_config:
                    original_name = item['original_name']
                    if original_name in df.columns:
                        original_col_names_for_customdata.append(original_name)
                        if df.schema[original_name] in [pl.Date, pl.Datetime]:
                            select_exprs.append(pl.col(original_name).dt.strftime("%Y-%m-%d %H:%M:%S").fill_null("N/A").alias(original_name))
                        else:
                            select_exprs.append(pl.col(original_name).cast(pl.Utf8).fill_null("N/A").alias(original_name))
                if select_exprs:
                    try:
                        customdata_numpy_array = df.select(select_exprs).to_numpy()
                    except pl.ColumnNotFoundError:
                        customdata_numpy_array = None
                        original_col_names_for_customdata = []
        
        plot_trace_adders = {
            "Scatter Plot": self._add_scatter_traces, "Line Plot": self._add_line_traces,
            "Bar Chart": self._add_bar_traces, "Pie Chart": self._add_pie_traces,
            "Histogram": self._add_histogram_traces, "Box Plot": self._add_box_traces,
            "Violin Plot": self._add_violin_traces, "Strip Plot": self._add_strip_traces,
            "Density Contour": self._add_density_contour_traces, "Distribution Plot (Distplot)": self._add_distplot_traces,
        }

        if plot_type in plot_trace_adders:
            if plot_type in ["Bar Chart", "Pie Chart"]:
                 plot_trace_adders[plot_type](fig, df, basic_args, advanced_args, None, hover_data_config, row, col)
            elif plot_type == "Distribution Plot (Distplot)":
                 plot_trace_adders[plot_type](fig, df, basic_args, advanced_args, None, [], row, col)
            else:
                 plot_trace_adders[plot_type](fig, df, basic_args, advanced_args,
                                           customdata_numpy_array, hover_data_config, row, col)

    def _get_hovertemplate(self, df_for_trace, x_col_name, y_col_name, z_col_name, names_col_name, values_col_name, hover_data_config_list, plot_type, advanced_args):
        ht_parts = []
        
        adv_args_source = {}
        if plot_type == "Distribution Plot (Distplot)":
            if self.distplot_advanced_group and self.distplot_advanced_group.isChecked():
                 adv_args_source = self.distplot_advanced_widgets
        elif self.advanced_group and self.advanced_group.isChecked():
            adv_args_source = self.advanced_config_widgets
        
        adv_x_label = advanced_args.get('xaxis_label_edit', "")
        adv_y_label = advanced_args.get('yaxis_label_edit', "")
        adv_z_label_widget = advanced_args.get('zaxis_label_edit', None)
        adv_z_label = adv_z_label_widget if adv_z_label_widget else None
        adv_legend_title = advanced_args.get('legend_title_edit', "")


        x_label = adv_x_label if adv_x_label else (x_col_name if x_col_name and x_col_name != "None" else "X")
        y_label = adv_y_label if adv_y_label else (y_col_name if y_col_name and y_col_name != "None" else "Y")
        z_label = adv_z_label if adv_z_label else (z_col_name if z_col_name and z_col_name != "None" else "Z")
        names_label = adv_legend_title if adv_legend_title else (names_col_name if names_col_name and names_col_name != "None" else "Name")
        values_label = "Value"

        def get_format_str(col_name_str, df_context):
            if col_name_str and col_name_str != "None" and col_name_str in df_context.columns:
                dtype = df_context.schema[col_name_str]
                if dtype == pl.Date: return "|%Y-%m-%d"
                if dtype == pl.Datetime: return "|%Y-%m-%d %H:%M:%S"
            return ""

        x_fmt = get_format_str(x_col_name, df_for_trace)
        y_fmt = get_format_str(y_col_name, df_for_trace)
        z_fmt = get_format_str(z_col_name, df_for_trace)

        if plot_type == "Pie Chart":
            if names_col_name and names_col_name != "None": ht_parts.append(f"<b>{names_label}</b>: %{{label}}<br>")
            if values_col_name and values_col_name != "None": ht_parts.append(f"<b>{values_label}</b>: %{{value}}<br>")
            ht_parts.append("<b>Percentage</b>: %{percent}<br>")
        elif plot_type == "Histogram" or plot_type == "Distribution Plot (Distplot)":
            x_axis_display_name_for_hover = x_label
            ht_parts.append(f"<b>{x_axis_display_name_for_hover} (bin)</b>: %{{x{x_fmt}}}<br>")
            ht_parts.append(f"<b>Count/Density</b>: %{{y}}<br>")
        elif plot_type == "Density Contour":
            if x_col_name and x_col_name != "None": ht_parts.append(f"<b>{x_label}</b>: %{{x{x_fmt}}}<br>")
            if y_col_name and y_col_name != "None": ht_parts.append(f"<b>{y_label}</b>: %{{y{y_fmt}}}<br>")
            if z_col_name and z_col_name != "None" and z_col_name in df_for_trace.columns: ht_parts.append(f"<b>{z_label} (Value)</b>: %{{z{z_fmt}}}<br>")
            else: ht_parts.append(f"<b>Density</b>: %{{z}}<br>")
        else:
            if x_col_name and x_col_name != "None": ht_parts.append(f"<b>{x_label}</b>: %{{x{x_fmt}}}<br>")
            if y_col_name and y_col_name != "None": ht_parts.append(f"<b>{y_label}</b>: %{{y{y_fmt}}}<br>")

        if hover_data_config_list:
            for i, item in enumerate(hover_data_config_list):
                display_name = item['display_name']
                ht_parts.append(f"<b>{display_name}</b>: %{{customdata[{i}]}}<br>")

        return "".join(ht_parts) + "<extra></extra>" if ht_parts else None

    def _handle_categorical_coloring_and_symboling(self, fig, df, trace_constructor, base_plotly_args,
                                                 color_by_col, symbol_by_col, plot_type,
                                                 hover_data_config, row, col, pattern_by_col=None, bar_pattern_shapes=None):

        color_palette = px.colors.qualitative.Plotly
        symbol_sequence = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'star', 'hexagram', 'pentagon']

        has_color_cat = color_by_col != "None" and color_by_col in df.columns and df.schema[color_by_col] in [pl.Utf8, pl.Categorical, pl.Boolean]
        has_symbol_cat = symbol_by_col != "None" and symbol_by_col in df.columns and df.schema[symbol_by_col] in [pl.Utf8, pl.Categorical, pl.Boolean]
        has_pattern_cat = pattern_by_col != "None" and pattern_by_col in df.columns and df.schema[pattern_by_col] in [pl.Utf8, pl.Categorical, pl.Boolean] and plot_type == "Bar Chart"


        def get_series_name(arg_val):
            return arg_val.name if isinstance(arg_val, pl.Series) else None

        original_x_series_name = get_series_name(base_plotly_args.get('x'))
        original_y_series_name = get_series_name(base_plotly_args.get('y'))
        original_z_series_name = get_series_name(base_plotly_args.get('z'))

        def build_slice_customdata(slice_df, hover_conf):
            if not hover_conf or slice_df.is_empty():
                return None
            select_exprs = []
            for item in hover_conf:
                orig_name = item['original_name']
                if orig_name in slice_df.columns:
                    if slice_df.schema[orig_name] in [pl.Date, pl.Datetime]:
                        select_exprs.append(pl.col(orig_name).dt.strftime("%Y-%m-%d %H:%M:%S").fill_null("N/A"))
                    else:
                        select_exprs.append(pl.col(orig_name).cast(pl.Utf8).fill_null("N/A"))
            return slice_df.select(select_exprs).to_numpy() if select_exprs else None


        if has_color_cat and has_pattern_cat and color_by_col != pattern_by_col and plot_type == "Bar Chart":
            unique_colors = df.get_column(color_by_col).unique().drop_nulls().sort().to_list()
            unique_patterns = df.get_column(pattern_by_col).unique().drop_nulls().sort().to_list()

            for i, color_val in enumerate(unique_colors):
                for j, pattern_val in enumerate(unique_patterns):
                    combined_slice_df = df.filter(
                        (pl.col(color_by_col) == color_val) & (pl.col(pattern_by_col) == pattern_val)
                    )
                    if combined_slice_df.is_empty():
                        continue

                    slice_customdata_np = build_slice_customdata(combined_slice_df, hover_data_config)
                    
                    trace_args_for_slice = {k: v for k, v in base_plotly_args.items()}
                    trace_args_for_slice['marker'] = base_plotly_args.get('marker', {}).copy()
                    trace_args_for_slice['showlegend'] = True
                    trace_args_for_slice['name'] = f"{color_val} ({pattern_val})"
                    trace_args_for_slice['legendgroup'] = str(color_val)

                    trace_args_for_slice['marker']['color'] = color_palette[i % len(color_palette)]
                    if bar_pattern_shapes:
                         trace_args_for_slice['marker']['pattern'] = {'shape': bar_pattern_shapes[j % len(bar_pattern_shapes)], 'solidity': 0.3}


                    if original_x_series_name and original_x_series_name in combined_slice_df.columns: trace_args_for_slice['x'] = combined_slice_df[original_x_series_name]
                    if original_y_series_name and original_y_series_name in combined_slice_df.columns: trace_args_for_slice['y'] = combined_slice_df[original_y_series_name]
                    
                    fig.add_trace(trace_constructor(**trace_args_for_slice, customdata=slice_customdata_np), row=row, col=col)
            return


        cat_col_to_iterate = None
        unique_values_list = []
        is_iterating_color, is_iterating_symbol, is_iterating_pattern = False, False, False

        if has_color_cat:
            cat_col_to_iterate = color_by_col
            unique_values_list = df.get_column(color_by_col).unique().drop_nulls().sort().to_list()
            is_iterating_color = True
            if has_pattern_cat and color_by_col == pattern_by_col and plot_type == "Bar Chart":
                is_iterating_pattern = True
        elif has_symbol_cat:
            cat_col_to_iterate = symbol_by_col
            unique_values_list = df.get_column(symbol_by_col).unique().drop_nulls().sort().to_list()
            is_iterating_symbol = True
        elif has_pattern_cat and plot_type == "Bar Chart":
            cat_col_to_iterate = pattern_by_col
            unique_values_list = df.get_column(pattern_by_col).unique().drop_nulls().sort().to_list()
            is_iterating_pattern = True


        if cat_col_to_iterate and unique_values_list:
            for idx, cat_value in enumerate(unique_values_list):
                df_slice = df.filter(pl.col(cat_col_to_iterate) == cat_value)
                if df_slice.is_empty(): continue

                slice_customdata_np = build_slice_customdata(df_slice, hover_data_config)

                trace_args_for_slice = {k: v for k, v in base_plotly_args.items()}
                trace_args_for_slice['marker'] = base_plotly_args.get('marker', {}).copy()
                trace_args_for_slice['showlegend'] = True
                trace_args_for_slice['name'] = str(cat_value)
                trace_args_for_slice['legendgroup'] = str(cat_value)

                if is_iterating_color:
                    trace_args_for_slice['marker']['color'] = color_palette[idx % len(color_palette)]
                if is_iterating_symbol:
                    trace_args_for_slice['marker']['symbol'] = symbol_sequence[idx % len(symbol_sequence)]
                if is_iterating_pattern and bar_pattern_shapes:
                    trace_args_for_slice['marker']['pattern'] = {'shape': bar_pattern_shapes[idx % len(bar_pattern_shapes)], 'solidity': 0.3}


                if original_x_series_name and original_x_series_name in df_slice.columns: trace_args_for_slice['x'] = df_slice[original_x_series_name]
                if original_y_series_name and original_y_series_name in df_slice.columns: trace_args_for_slice['y'] = df_slice[original_y_series_name]

                fig.add_trace(trace_constructor(**trace_args_for_slice, customdata=slice_customdata_np), row=row, col=col)
        else:
            full_df_customdata_np = build_slice_customdata(df, hover_data_config)
            
            final_args = base_plotly_args.copy()
            if 'name' not in final_args:
                if original_y_series_name and original_y_series_name != "None":
                    final_args['name'] = original_y_series_name
                elif original_x_series_name and original_x_series_name != "None":
                    final_args['name'] = original_x_series_name
            
            if final_args.get('name') is not None and final_args.get('name') != "None":
                 final_args['showlegend'] = True
            else:
                 if 'showlegend' not in final_args: final_args['showlegend'] = False


            fig.add_trace(trace_constructor(**final_args, customdata=full_df_customdata_np), row=row, col=col)


    def _add_scatter_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col = basic_args.get('x_col'), basic_args.get('y_col')
        color_col, size_col, symbol_col = basic_args.get('color_col'), basic_args.get('size_col'), basic_args.get('symbol_col')

        if x_col == "None" or y_col == "None" or x_col not in df.columns or y_col not in df.columns or df.is_empty(): return

        plotly_args = {"x": df.get_column(x_col), "y": df.get_column(y_col), "mode": "markers"}
        plotly_args["marker"] = {}

        if color_col != "None" and color_col in df.columns and not (df.schema[color_col] in [pl.Utf8, pl.Categorical, pl.Boolean]):
            plotly_args["marker"]["color"] = df.get_column(color_col)
            csc = advanced_args.get('color_continuous_scale_combo', "None")
            if csc != "None": plotly_args["marker"]["colorscale"] = csc
            plotly_args["marker"]["showscale"] = True
            user_legend_title = advanced_args.get('legend_title_edit')
            plotly_args["marker"]["colorbar"] = {"title": user_legend_title if user_legend_title else color_col}

        if size_col != "None" and size_col in df.columns:
            size_data_orig = df.get_column(size_col).fill_null(0).cast(pl.Float64)
            min_s_orig = size_data_orig.min()
            if min_s_orig is None: min_s_orig = 0.0
            shifted_size_data = size_data_orig + abs(min_s_orig)
            mean_shifted_size = shifted_size_data.mean()
            if mean_shifted_size is not None and abs(mean_shifted_size) > 1e-9:
                scaled_sizes = (10 * shifted_size_data / mean_shifted_size)
                plotly_args["marker"]["size"] = scaled_sizes.map_elements(lambda x: max(2, min(x if x is not None else 2, 50)), return_dtype=pl.Float64)
            else:
                plotly_args["marker"]["size"] = size_data_orig.map_elements(lambda x: max(2, min(x if x is not None else 0, 50)), return_dtype=pl.Float64).clip_min(2)
                if min_s_orig < 0:
                    self.statusBar().showMessage(f"Sizes for '{size_col}' set to default range due to uniform/problematic values.", 3000)


        if symbol_col != "None" and symbol_col in df.columns and not (df.schema[symbol_col] in [pl.Utf8, pl.Categorical, pl.Boolean]):
             plotly_args["marker"]["symbol"] = df.get_column(symbol_col)

        plotly_args["hovertemplate"] = self._get_hovertemplate(df, x_col, y_col, None, None, None, hover_data_config, "Scatter Plot", advanced_args)

        self._handle_categorical_coloring_and_symboling(
            fig, df, go.Scatter, plotly_args,
            color_col, symbol_col, "Scatter Plot",
            hover_data_config,
            row, col
        )

    def _add_line_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col = basic_args.get('x_col'), basic_args.get('y_col')
        color_col, symbol_col = basic_args.get('color_col'), basic_args.get('symbol_col')
        sort_x = basic_args.get('sort_x', True)

        if x_col == "None" or y_col == "None" or x_col not in df.columns or y_col not in df.columns or df.is_empty(): return

        plot_df = df
        if sort_x and x_col in plot_df.columns:
            plot_df = plot_df.sort(x_col)
        
        plotly_args = {"x": plot_df.get_column(x_col), "y": plot_df.get_column(y_col), "mode": "lines+markers"}
        plotly_args["marker"] = {}

        if color_col != "None" and color_col in plot_df.columns and not (plot_df.schema[color_col] in [pl.Utf8, pl.Categorical, pl.Boolean]):
            plotly_args["marker"]["color"] = plot_df.get_column(color_col)
            csc = advanced_args.get('color_continuous_scale_combo', "None")
            if csc != "None": plotly_args["marker"]["colorscale"] = csc
            plotly_args["marker"]["showscale"] = True
            user_legend_title = advanced_args.get('legend_title_edit')
            plotly_args["marker"]["colorbar"] = {"title": user_legend_title if user_legend_title else color_col}


        if symbol_col != "None" and symbol_col in plot_df.columns and not (plot_df.schema[symbol_col] in [pl.Utf8, pl.Categorical, pl.Boolean]):
            plotly_args["marker"]["symbol"] = plot_df.get_column(symbol_col)

        plotly_args["hovertemplate"] = self._get_hovertemplate(plot_df, x_col,y_col,None,None,None, hover_data_config, "Line Plot", advanced_args)

        self._handle_categorical_coloring_and_symboling(
            fig, plot_df, go.Scatter, plotly_args,
            color_col, symbol_col, "Line Plot",
            hover_data_config,
            row, col
        )

    def _add_bar_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col = basic_args.get('x_col'), basic_args.get('y_col')
        color_col, pattern_col = basic_args.get('color_col'), basic_args.get('pattern_col')
        agg_func, orientation = basic_args.get('agg_func'), basic_args.get('orientation')
        barmode = basic_args.get('barmode', 'group').lower()

        bar_pattern_shapes = ['/', '\\', 'x', '-', '|', '+', '.']

        if x_col == "None" or x_col not in df.columns or df.is_empty(): return

        plot_df_for_agg = df
        group_by_cols = [x_col]
        
        is_color_categorical_for_grouping = color_col != "None" and color_col in plot_df_for_agg.columns and \
                                           plot_df_for_agg.schema[color_col] in [pl.Utf8, pl.Categorical, pl.Boolean]
        if is_color_categorical_for_grouping and color_col != x_col:
            group_by_cols.append(color_col)

        is_pattern_categorical_for_grouping = pattern_col != "None" and pattern_col in plot_df_for_agg.columns and \
                                           plot_df_for_agg.schema[pattern_col] in [pl.Utf8, pl.Categorical, pl.Boolean]
        if is_pattern_categorical_for_grouping and pattern_col not in group_by_cols:
            group_by_cols.append(pattern_col)


        y_values_col_name_internal = "_vw_agg_y_val_"
        expressions_for_agg = []
        y_col_for_hovertemplate = y_col

        if y_col != "None" and y_col in plot_df_for_agg.columns:
            if agg_func != "count":
                plot_df_for_agg = plot_df_for_agg.with_columns(pl.col(y_col).cast(pl.Float64, strict=False))
            
            agg_map_polars = {
                "sum": pl.sum(y_col).alias(y_values_col_name_internal), "mean": pl.mean(y_col).alias(y_values_col_name_internal),
                "median": pl.median(y_col).alias(y_values_col_name_internal), "min": pl.min(y_col).alias(y_values_col_name_internal),
                "max": pl.max(y_col).alias(y_values_col_name_internal), "first": pl.first(y_col).alias(y_values_col_name_internal),
                "last": pl.last(y_col).alias(y_values_col_name_internal),
            }
            if agg_func == "count":
                expressions_for_agg.append(pl.len().alias(y_values_col_name_internal))
            elif agg_func in agg_map_polars:
                expressions_for_agg.append(agg_map_polars[agg_func])
            else:
                expressions_for_agg.append(pl.len().alias(y_values_col_name_internal))
            y_col_for_hovertemplate = y_values_col_name_internal
        else:
            y_values_col_name_internal = "_vw_count_y_val_"
            expressions_for_agg.append(pl.len().alias(y_values_col_name_internal))
            y_col_for_hovertemplate = y_values_col_name_internal

        processed_hover_cols_agg = []
        if hover_data_config:
            for item in hover_data_config:
                h_col_name = item['original_name']
                if h_col_name in plot_df_for_agg.columns and h_col_name not in group_by_cols and h_col_name != y_col:
                    expressions_for_agg.append(pl.first(h_col_name).alias(h_col_name))
                    processed_hover_cols_agg.append(item)
        
        aggregated_plot_df = plot_df_for_agg.group_by(group_by_cols, maintain_order=True).agg(
            expressions_for_agg
        ).sort(x_col)

        actual_y_col_name_in_agg_df = y_values_col_name_internal

        if is_color_categorical_for_grouping or is_pattern_categorical_for_grouping :
            fig.update_layout(barmode=barmode)
        else:
            fig.update_layout(barmode='group')

        bar_args = {}
        bar_args['x' if orientation == "Vertical" else 'y'] = aggregated_plot_df.get_column(x_col)
        bar_args['y' if orientation == "Vertical" else 'x'] = aggregated_plot_df.get_column(actual_y_col_name_in_agg_df)
        bar_args["orientation"] = 'v' if orientation == "Vertical" else 'h'
        bar_args["marker"] = {}

        if color_col != "None" and color_col in aggregated_plot_df.columns and not is_color_categorical_for_grouping:
            if not (aggregated_plot_df.schema[color_col] in [pl.Utf8, pl.Categorical, pl.Boolean]):
                bar_args["marker"]["color"] = aggregated_plot_df.get_column(color_col)
                csc = advanced_args.get('color_continuous_scale_combo', "None")
                if csc != "None": bar_args["marker"]["colorscale"] = csc
                bar_args["marker"]["showscale"] = True
                user_legend_title = advanced_args.get('legend_title_edit')
                bar_args["marker"]["colorbar"] = {"title": user_legend_title if user_legend_title else color_col}

        bar_args["hovertemplate"] = self._get_hovertemplate(
            aggregated_plot_df, x_col, actual_y_col_name_in_agg_df, None, None, None,
            processed_hover_cols_agg, "Bar Chart", advanced_args
        )
        
        color_col_for_handler = color_col if is_color_categorical_for_grouping else "None"
        pattern_col_for_handler = pattern_col if is_pattern_categorical_for_grouping else "None"


        self._handle_categorical_coloring_and_symboling(
            fig, aggregated_plot_df, go.Bar, bar_args,
            color_col_for_handler, "None", "Bar Chart",
            processed_hover_cols_agg, row, col,
            pattern_by_col=pattern_col_for_handler, bar_pattern_shapes=bar_pattern_shapes
        )


    def _add_pie_traces(self, fig, df, basic_args, advanced_args, _customdata_ignored, hover_data_config_ignored, row, col):
        names_col = basic_args.get('names_col')
        pie_mode = basic_args.get('pie_mode', "Count Occurrences (Rows)")
        values_col_for_sum_name = basic_args.get('values_col')
        hole = basic_args.get('hole_size', 0) / 100.0
        
        textinfo = basic_args.get('textinfo', 'percent+label')
        sort_slices = basic_args.get('sort_pie', True)
        direction = basic_args.get('direction', 'clockwise')

        pull_str = advanced_args.get('pull_edit', "")
        slice_colors_str = advanced_args.get('slice_colors_edit', "")
        slice_line_color = advanced_args.get('slice_line_color_edit', None)
        slice_line_width = advanced_args.get('slice_line_width_spin', 1)

        if names_col == "None" or names_col not in df.columns or df.is_empty():
            return

        plot_df = None
        actual_values_col_name_for_trace = None

        if pie_mode == "Count Occurrences (Rows)":
            plot_df = df.group_by(names_col).agg(pl.len().alias("count")).sort(names_col)
            actual_values_col_name_for_trace = "count"
        elif pie_mode == "Sum Values from Column":
            if values_col_for_sum_name == "None" or values_col_for_sum_name not in df.columns:
                return
            try:
                df_for_sum = df.with_columns(pl.col(values_col_for_sum_name).cast(pl.Float64, strict=False))
                plot_df = df_for_sum.group_by(names_col).agg(pl.sum(values_col_for_sum_name).alias("summed_values")).sort(names_col)
                actual_values_col_name_for_trace = "summed_values"
            except Exception as e:
                return
        
        if plot_df is None or plot_df.is_empty() or actual_values_col_name_for_trace not in plot_df.columns:
            return

        pull_values = None
        marker_options = {}

        fig.add_trace(go.Pie(
            labels=plot_df.get_column(names_col),
            values=plot_df.get_column(actual_values_col_name_for_trace),
            hole=hole, textinfo=textinfo if textinfo != "none" else None,
            sort=sort_slices, direction=direction, pull=pull_values,
            marker=marker_options if marker_options else None, name="",
            hovertemplate = self._get_hovertemplate(
                plot_df, None, None, None, names_col, actual_values_col_name_for_trace,
                [], "Pie Chart", advanced_args
            ),
        ), row=row, col=col)

    def _add_histogram_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col = basic_args.get('x_col'), basic_args.get('y_col')
        color_col = basic_args.get('color_col')
        nbinsx = basic_args.get('nbinsx', 0)
        histnorm = basic_args.get('histnorm', "None")
        cumulative_enabled = basic_args.get('cumulative_enabled', False)

        if x_col == "None" or x_col not in df.columns or df.is_empty(): return

        plotly_args = {"x": df.get_column(x_col)}
        if y_col != "None" and y_col in df.columns: plotly_args["y"] = df.get_column(y_col)

        if nbinsx > 0: plotly_args["nbinsx"] = nbinsx
        if histnorm != "None": plotly_args["histnorm"] = histnorm.lower().replace(" ", "")
        if cumulative_enabled: plotly_args["cumulative"] = {"enabled": True}
        
        plotly_args["marker"] = {}
        plotly_args["hovertemplate"] = self._get_hovertemplate(df, x_col, y_col, None,None,None, [], "Histogram", advanced_args)


        self._handle_categorical_coloring_and_symboling(
            fig, df, go.Histogram, plotly_args,
            color_col, "None", "Histogram", hover_data_config, row, col
        )

    def _add_box_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col = basic_args.get('x_col'), basic_args.get('y_col')
        color_col = basic_args.get('color_col')
        boxpoints, orientation, notched = basic_args.get('boxpoints'), basic_args.get('orientation'), basic_args.get('notched')

        if y_col == "None" or y_col not in df.columns or df.is_empty(): return

        plotly_args = {}
        main_val_col_series = df.get_column(y_col)
        group_col_name = x_col if x_col != "None" and x_col in df.columns else None
        group_col_series = df.get_column(group_col_name) if group_col_name else None

        ht_x_label, ht_y_label = None, None
        if orientation == "Vertical":
            plotly_args['y'] = main_val_col_series
            if group_col_series is not None: plotly_args['x'] = group_col_series
            ht_y_label, ht_x_label = y_col, group_col_name
        else:
            plotly_args['x'] = main_val_col_series
            if group_col_series is not None: plotly_args['y'] = group_col_series
            ht_x_label, ht_y_label = y_col, group_col_name
        
        plotly_args["boxpoints"] = boxpoints.lower().replace(" ", "") if boxpoints != "False" else False
        plotly_args["notched"] = notched
        
        plotly_args["hovertemplate"] = self._get_hovertemplate(df, ht_x_label, ht_y_label, None,None,None, hover_data_config, "Box Plot", advanced_args)

        self._handle_categorical_coloring_and_symboling(
            fig, df, go.Box, plotly_args, color_col, "None", "Box Plot", hover_data_config, row, col
        )

    def _add_violin_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col = basic_args.get('x_col'), basic_args.get('y_col')
        color_col = basic_args.get('color_col')
        box_visible, points, orientation = basic_args.get('box_visible'), basic_args.get('points'), basic_args.get('orientation')
        split_by_col = basic_args.get('split_by_col', "None")

        if y_col == "None" or y_col not in df.columns or df.is_empty(): return

        plotly_args = {}
        main_val_col_series = df.get_column(y_col)
        group_col_name = x_col if x_col != "None" and x_col in df.columns else None
        group_col_series = df.get_column(group_col_name) if group_col_name else None

        ht_x_label, ht_y_label = None, None
        if orientation == "Vertical":
            plotly_args['y'] = main_val_col_series
            if group_col_series is not None: plotly_args['x'] = group_col_series
            ht_y_label, ht_x_label = y_col, group_col_name
        else:
            plotly_args['x'] = main_val_col_series
            if group_col_series is not None: plotly_args['y'] = group_col_series
            ht_x_label, ht_y_label = y_col, group_col_name

        plotly_args["box"] = {"visible": box_visible}
        plotly_args["points"] = points.lower().replace(" ", "") if points != "False" else False
        plotly_args["hovertemplate"] = self._get_hovertemplate(df, ht_x_label, ht_y_label, None,None,None, hover_data_config, "Violin Plot", advanced_args)

        if split_by_col != "None" and split_by_col in df.columns:
            unique_split_values = df.get_column(split_by_col).drop_nulls().unique().to_list()
            if len(unique_split_values) == 2:
                self.statusBar().showMessage("Split violin with full custom hover not fully shown in this snippet.", 2000)


        self._handle_categorical_coloring_and_symboling(
            fig, df, go.Violin, plotly_args, color_col, "None", "Violin Plot", hover_data_config, row, col
        )

    def _add_strip_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col = basic_args.get('x_col'), basic_args.get('y_col')
        color_col, orientation = basic_args.get('color_col'), basic_args.get('orientation')

        if y_col == "None" or y_col not in df.columns or df.is_empty(): return

        plotly_args = {}
        main_val_col_series = df.get_column(y_col)
        group_col_name = x_col if x_col != "None" and x_col in df.columns else None
        group_col_series = df.get_column(group_col_name) if group_col_name else None

        ht_x_label, ht_y_label = None, None
        if orientation == "Vertical":
            plotly_args['y'] = main_val_col_series
            if group_col_series is not None: plotly_args['x'] = group_col_series
            ht_y_label, ht_x_label = y_col, group_col_name
        else:
            plotly_args['x'] = main_val_col_series
            if group_col_series is not None: plotly_args['y'] = group_col_series
            ht_x_label, ht_y_label = y_col, group_col_name

        plotly_args["mode"] = "markers"
        plotly_args["marker"] = {"size": 5}
        plotly_args["hovertemplate"] = self._get_hovertemplate(df, ht_x_label, ht_y_label, None,None,None, hover_data_config, "Strip Plot", advanced_args)
        
        self._handle_categorical_coloring_and_symboling(
            fig, df, go.Scatter, plotly_args, color_col, "None", "Strip Plot", hover_data_config, row, col
        )

    def _add_density_contour_traces(self, fig, df, basic_args, advanced_args, customdata_numpy_array_ignored, hover_data_config, row, col):
        x_col, y_col, z_col = basic_args.get('x_col'), basic_args.get('y_col'), basic_args.get('z_col')
        
        if x_col == "None" or y_col == "None" or x_col not in df.columns or y_col not in df.columns or df.is_empty(): return

        trace_type = go.Histogram2dContour
        plotly_args = {"x": df.get_column(x_col), "y": df.get_column(y_col)}
        if z_col != "None" and z_col in df.columns: plotly_args["z"] = df.get_column(z_col)
        
        csc = advanced_args.get('color_continuous_scale_combo', "None")
        if csc != "None": plotly_args["colorscale"] = csc
        plotly_args["showscale"] = True
        user_legend_title = advanced_args.get('legend_title_edit')
        plotly_args["colorbar"] = {"title": user_legend_title if user_legend_title else (z_col if z_col!="None" else "Density")}
        
        final_customdata_for_density = None
        if hover_data_config and not df.is_empty():
            select_exprs = []
            for item in hover_data_config:
                orig_name = item['original_name']
                if orig_name in df.columns:
                     if df.schema[orig_name] in [pl.Date, pl.Datetime]:
                         select_exprs.append(pl.col(orig_name).dt.strftime("%Y-%m-%d %H:%M:%S").fill_null("N/A"))
                     else:
                         select_exprs.append(pl.col(orig_name).cast(pl.Utf8).fill_null("N/A"))
            if select_exprs:
                final_customdata_for_density = df.select(select_exprs).to_numpy()


        plotly_args["hovertemplate"] = self._get_hovertemplate(df, x_col,y_col,z_col,None,None, hover_data_config, "Density Contour", advanced_args)
        fig.add_trace(trace_type(**plotly_args, customdata=final_customdata_for_density), row=row, col=col)


    def _add_distplot_traces(self, fig, df, basic_args, advanced_args_dist, customdata_ignored, hover_config_ignored, row, col):
        hist_data_cols_selected = basic_args.get('hist_data_cols_list', [])
        curve_type = basic_args.get('curve_type', "kde")
        show_hist = basic_args.get('show_hist', True)
        show_rug = basic_args.get('show_rug', True)
        bin_size_str = basic_args.get('bin_size', "")

        if not hist_data_cols_selected or df.is_empty():
            fig.add_annotation(text="No data columns selected for distplot.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return
        
        first_col_name = hist_data_cols_selected[0]
        if first_col_name not in df.columns:
            fig.add_annotation(text=f"Column '{first_col_name}' not found.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False); return
        first_col_type = df.schema[first_col_name]
        is_first_temporal = first_col_type in [pl.Date, pl.Datetime]
        hist_data_values, group_labels, rug_text_data, reference_date = [], [], [], None
        if is_first_temporal:
            min_dates = [df.get_column(cn).drop_nulls().min() for cn in hist_data_cols_selected if cn in df.columns and df.schema[cn] in [pl.Date, pl.Datetime] and not df.get_column(cn).drop_nulls().is_empty()]
            if not min_dates: fig.add_annotation(text="Selected temporal columns are empty/null.",showarrow=False); return
            ref_date_candidate = min(min_dates)
            reference_date = datetime.datetime.combine(ref_date_candidate, datetime.time.min) if isinstance(ref_date_candidate, datetime.date) and not isinstance(ref_date_candidate, datetime.datetime) else ref_date_candidate
            if not isinstance(reference_date, datetime.datetime): fig.add_annotation(text="Temporal column type error.",showarrow=False); return
        for col_name in hist_data_cols_selected:
            if col_name not in df.columns: continue
            current_col_type = df.schema[col_name]
            is_current_temporal = current_col_type in [pl.Date, pl.Datetime]
            if is_first_temporal != is_current_temporal:
                QMessageBox.warning(self, "Type Mismatch", "All columns for Distplot must be numeric or temporal.");fig.add_annotation(text="Mixed types for Distplot.",showarrow=False); return
            series = df.get_column(col_name).drop_nulls()
            if series.is_empty(): continue
            group_labels.append(col_name)
            if is_current_temporal:
                numeric_series = series.map_elements(lambda d: (d - reference_date).total_seconds() / (24*3600) if isinstance(d, datetime.datetime) else (datetime.datetime.combine(d, datetime.time.min) - reference_date).total_seconds() / (24*3600) if isinstance(d, datetime.date) else None, return_dtype=pl.Float64).drop_nulls()
                hist_data_values.append(numeric_series.to_list())
                rug_text_data.append(series.dt.strftime("%Y-%m-%d").to_list())
            else:
                hist_data_values.append(series.to_list())
                rug_text_data.append(series.cast(str).to_list())
        if not hist_data_values: fig.add_annotation(text="Selected columns empty/null after processing.",showarrow=False); return
        final_bin_size_param = []
        if bin_size_str.strip():
            try:
                bin_count = int(bin_size_str)
                if bin_count > 0:
                    final_bin_size_param = [((max(d_list) - min(d_list)) / bin_count) if len(d_list) > 1 and max(d_list) > min(d_list) else 0.01 for d_list in hist_data_values]
                else: final_bin_size_param = []
            except ValueError:
                try:
                    if bin_size_str.startswith('[') and bin_size_str.endswith(']'):
                        parsed_list = [float(x.strip()) for x in bin_size_str[1:-1].split(',')]
                        final_bin_size_param = parsed_list * len(hist_data_values) if len(parsed_list) == 1 and len(hist_data_values) > 1 else parsed_list
                    else: final_bin_size_param = [float(bin_size_str)] * len(hist_data_values)
                except ValueError: self.statusBar().showMessage("Invalid bin size. Auto.", 2000); final_bin_size_param = []
        else:
            final_bin_size_param = [((max(d_list) - min(d_list)) / 10.0) if len(d_list) > 1 and max(d_list) > min(d_list) else 0.01 for d_list in hist_data_values]
        
        if row is not None or col is not None: fig.add_annotation(text="Distplot not supported in faceting.", row=row,col=col,showarrow=False); return
        distplot_colors_str = advanced_args_dist.get('distplot_colors',"")
        colors_arg = [c.strip() for c in distplot_colors_str.split(',')] if distplot_colors_str else None
        
        try:
            dist_fig_ff = ff.create_distplot(hist_data_values, group_labels, show_hist=show_hist, show_rug=show_rug, bin_size=final_bin_size_param, curve_type=curve_type if curve_type != "None" else None, colors=colors_arg, rug_text=rug_text_data if show_rug else None)
            fig.data = []
            for trace_data in dist_fig_ff.data: fig.add_trace(trace_data)
            fig.layout = dist_fig_ff.layout
        except Exception as e:
            fig.add_annotation(text=f"Error creating distplot: {str(e)}",showarrow=False); print(f"Distplot error: {e}\n{traceback.format_exc()}")

    def _markdown_to_plotly_html(self, md_text):
        if not md_text:
            return ""

        html_text = md_text

        html_text = html_text.replace('\r\n', '<br>').replace('\n', '<br>')
        html_text = re.sub(r'^\s*###\s*(.*?)(<br>|$)', r'<span style="font-size: 1.0em; font-weight: bold;">\1</span>\2', html_text, flags=re.MULTILINE)
        html_text = re.sub(r'^\s*##\s*(.*?)(<br>|$)', r'<span style="font-size: 1.1em; font-weight: bold;">\1</span>\2', html_text, flags=re.MULTILINE)
        html_text = re.sub(r'^\s*#\s*(.*?)(<br>|$)', r'<span style="font-size: 1.2em; font-weight: bold;">\1</span>\2', html_text, flags=re.MULTILINE)
        html_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html_text)
        html_text = re.sub(r'__(.*?)__', r'<b>\1</b>', html_text)
        html_text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', html_text)
        html_text = re.sub(r'_(.*?)_', r'<i>\1</i>', html_text)

        html_text = html_text.strip()
        while html_text.startswith("<br>"):
            html_text = html_text[4:].strip()
        while html_text.endswith("<br>"):
            html_text = html_text[:-4].strip()
            
        return html_text


    def _apply_advanced_layout(self, fig, advanced_args, basic_args, plot_type, df_for_context):
        layout_updates = {}

        title_text = advanced_args.get('plot_title_edit')
        if not title_text:
            title_text = f"{plot_type}"
            if hasattr(self, 'current_filename_hint') and self.current_filename_hint and self.current_filename_hint != "No data":
                title_text += f" of {os.path.basename(self.current_filename_hint)}"
        layout_updates["title_text"] = title_text
        
        subtitle_markdown = advanced_args.get('subtitle_edit', "")
        if subtitle_markdown:
            subtitle_html = self._markdown_to_plotly_html(subtitle_markdown)

            if fig.layout.annotations is None:
                fig.layout.annotations = []
            
            fig.layout.annotations = [ann for ann in fig.layout.annotations if ann.get('name') != '_subtitle_annotation_']

            fig.add_annotation(
                text=subtitle_html, xref="paper", yref="paper",
                x=0.5, y=1.02, showarrow=False, align="center", 
                font=dict(color="dimgray"), xanchor="center", yanchor="bottom",
                name='_subtitle_annotation_' 
            )
            if "title_y" not in layout_updates:
                layout_updates["title_y"] = 0.95
                layout_updates["title_yanchor"] = "top"


        x_col_name_for_label = basic_args.get('x_col')
        y_col_name_for_label = basic_args.get('y_col')
        
        if plot_type == "Pie Chart":
            x_col_name_for_label = basic_args.get('names_col', "Labels") 
            y_col_name_for_label_widget = basic_args.get('values_col') 
            y_col_name_for_label = y_col_name_for_label_widget.currentText() if isinstance(y_col_name_for_label_widget, QComboBox) else "Values"
        elif plot_type == "Histogram" and (not fig.layout.yaxis or not fig.layout.yaxis.title or not fig.layout.yaxis.title.text):
             y_col_name_for_label = "Count / Density" 
        elif plot_type == "Distribution Plot (Distplot)":
            if not fig.layout.xaxis or not fig.layout.xaxis.title or not fig.layout.xaxis.title.text:
                 layout_updates["xaxis_title_text"] = advanced_args.get('xaxis_label_edit', "Value")
            if not fig.layout.yaxis or not fig.layout.yaxis.title or not fig.layout.yaxis.title.text:
                 layout_updates["yaxis_title_text"] = advanced_args.get('yaxis_label_edit', "Density")

        adv_x_label = advanced_args.get('xaxis_label_edit')
        adv_y_label = advanced_args.get('yaxis_label_edit')
        final_x_label = adv_x_label if adv_x_label else (x_col_name_for_label if x_col_name_for_label and x_col_name_for_label != "None" else None)
        final_y_label = adv_y_label if adv_y_label else (y_col_name_for_label if y_col_name_for_label and y_col_name_for_label != "None" else None)

        if final_x_label and plot_type != "Distribution Plot (Distplot)": layout_updates["xaxis_title_text"] = final_x_label
        if final_y_label and plot_type != "Distribution Plot (Distplot)": layout_updates["yaxis_title_text"] = final_y_label
        
        if plot_type == "Distribution Plot (Distplot)": 
            if adv_x_label: fig.layout.xaxis.title.text = adv_x_label
            elif fig.layout.xaxis and (not fig.layout.xaxis.title or not fig.layout.xaxis.title.text): fig.layout.xaxis.title.text = "Value"
            
            if adv_y_label: fig.layout.yaxis.title.text = adv_y_label
            elif fig.layout.yaxis and (not fig.layout.yaxis.title or not fig.layout.yaxis.title.text): fig.layout.yaxis.title.text = "Density"

            user_legend_title_dist = advanced_args.get('legend_title_edit')
            if user_legend_title_dist: fig.layout.legend.title.text = user_legend_title_dist
            elif fig.layout.legend and (not fig.layout.legend.title or not fig.layout.legend.title.text): fig.layout.legend.title.text = "Data Series"

        if advanced_args.get('log_x_check', False): layout_updates["xaxis_type"] = "log"
        if advanced_args.get('log_y_check', False): layout_updates["yaxis_type"] = "log"
        
        x_range_str = advanced_args.get('xaxis_range_edit')
        if x_range_str:
            try: layout_updates["xaxis_range"] = [float(v.strip()) for v in x_range_str.split(',')]
            except: 
                if hasattr(self, 'statusBar'): self.statusBar().showMessage("Invalid X-axis range format.", 2000)
        
        y_range_str = advanced_args.get('yaxis_range_edit')
        if y_range_str:
            try: layout_updates["yaxis_range"] = [float(v.strip()) for v in y_range_str.split(',')]
            except: 
                if hasattr(self, 'statusBar'): self.statusBar().showMessage("Invalid Y-axis range format.", 2000)

        template = advanced_args.get('template_combo', "None")
        if template != "None": layout_updates["template"] = template

        user_legend_title = advanced_args.get('legend_title_edit')
        color_by_col_name = basic_args.get('color_col', "None")
        symbol_by_col_name = basic_args.get('symbol_col', "None") 
        final_legend_title_text = user_legend_title 

        if not final_legend_title_text and df_for_context is not None and not df_for_context.is_empty(): 
            is_color_cat = color_by_col_name != "None" and color_by_col_name in df_for_context.columns and \
                           df_for_context.schema[color_by_col_name] in [pl.Utf8, pl.Categorical, pl.Boolean]
            is_symbol_cat = symbol_by_col_name != "None" and symbol_by_col_name in df_for_context.columns and \
                            df_for_context.schema[symbol_by_col_name] in [pl.Utf8, pl.Categorical, pl.Boolean]
            if is_color_cat:
                final_legend_title_text = color_by_col_name
            elif is_symbol_cat: 
                final_legend_title_text = symbol_by_col_name
        
        if final_legend_title_text and plot_type != "Distribution Plot (Distplot)":
            is_color_cont = False
            if df_for_context is not None and not df_for_context.is_empty() and \
               color_by_col_name != "None" and color_by_col_name in df_for_context.columns and \
               not (df_for_context.schema[color_by_col_name] in [pl.Utf8, pl.Categorical, pl.Boolean]):
                is_color_cont = True
                
            if is_color_cont: 
                if 'coloraxis' not in layout_updates: layout_updates['coloraxis'] = {}
                if 'colorbar' not in layout_updates['coloraxis']: layout_updates['coloraxis']['colorbar'] = {}
                layout_updates['coloraxis']['colorbar']['title_text'] = final_legend_title_text
            else: 
                layout_updates["legend_title_text"] = final_legend_title_text
        
        if plot_type != "Distribution Plot (Distplot)":
            fig.update_layout(**layout_updates)
        else: 
            if "title_text" in layout_updates: fig.layout.title.text = layout_updates["title_text"]
            if "title_y" in layout_updates: fig.layout.title.y = layout_updates["title_y"]
            if "title_yanchor" in layout_updates: fig.layout.title.yanchor = layout_updates["title_yanchor"]

            if "template" in layout_updates: fig.layout.template = layout_updates["template"]
            if "xaxis_type" in layout_updates and fig.layout.xaxis: fig.layout.xaxis.type = layout_updates["xaxis_type"]
            if "yaxis_type" in layout_updates and fig.layout.yaxis: fig.layout.yaxis.type = layout_updates["yaxis_type"]
            if "xaxis_range" in layout_updates and fig.layout.xaxis: fig.layout.xaxis.range = layout_updates["xaxis_range"]
            if "yaxis_range" in layout_updates and fig.layout.yaxis: fig.layout.yaxis.range = layout_updates["yaxis_range"]

        cdm_str = advanced_args.get('color_discrete_map_edit')
        if cdm_str:
            try:
                color_discrete_map = literal_eval(cdm_str)
                if isinstance(color_discrete_map, dict):
                    for trace in fig.data:
                        if hasattr(trace, 'name') and trace.name in color_discrete_map:
                            if not hasattr(trace, 'marker') or trace.marker is None: 
                                if isinstance(trace, go.Bar): trace.marker = go.bar.Marker()
                                elif isinstance(trace, go.Scatter): trace.marker = go.scatter.Marker()
                            if hasattr(trace, 'marker') and trace.marker is not None:
                                trace.marker.color = color_discrete_map[trace.name]
            except Exception as e:
                if hasattr(self, 'statusBar'): self.statusBar().showMessage(f"Invalid color discrete map format: {e}", 2000)

        marker_symbol = advanced_args.get('marker_symbol_combo', "None")
        if marker_symbol != "None":
            fig.update_traces(marker_symbol=marker_symbol)


    def _get_widget_value(self, widget):
        if isinstance(widget, QComboBox): return widget.currentText()
        if isinstance(widget, QLineEdit): return widget.text()
        if isinstance(widget, QTextEdit): return widget.toPlainText()
        if isinstance(widget, QCheckBox): return widget.isChecked()
        if isinstance(widget, QSpinBox): return widget.value()
        
        if isinstance(widget, list) and all(isinstance(item, QCheckBox) for item in widget):
            return [cb.text() for cb in widget if cb.isChecked()]
            
        if isinstance(widget, QListWidget): 
            return [item.text() for item in widget.selectedItems()]
            
        return None

    def closeEvent(self, event):
        # FIX: The temporary file is no longer used, so no cleanup is needed here.
        if hasattr(self.plot_view, 'stop'): self.plot_view.stop()
        self.plot_view.setAttribute(Qt.WA_DeleteOnClose)
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    class DummyPickaxe:
        def get_current_dataframe_for_vw(self):
            np.random.seed(42)
            n_rows = 120
            data = {
                'categoryA': np.random.choice(['Alpha', 'Beta', 'Gamma', 'Delta', None], n_rows, p=[0.2,0.3,0.25,0.15,0.1]),
                'categoryB': np.random.choice(['X', 'Y', 'Z', None], n_rows, p=[0.4,0.3,0.2,0.1]),
                'value1': np.random.rand(n_rows) * 100 - 20,
                'value2': np.random.rand(n_rows) * 50 + 10,
                'size_col': np.random.rand(n_rows) * 20 + 5,
                'time_data': pl.datetime_range(datetime.datetime(2023,1,1), datetime.datetime(2023,1,1) + datetime.timedelta(days=n_rows-1), "1d", eager=True).slice(0,n_rows),
                'date_data': pl.date_range(datetime.date(2022,1,1), datetime.date(2022,1,1) + datetime.timedelta(days=n_rows-1), "1d", eager=True).slice(0,n_rows),
                'bool_data': np.random.choice([True, False, None], n_rows, p=[0.45,0.4,0.15]),
            }
            df = pl.DataFrame(data)
            df = df.with_row_count("__original_index__")
            return df

        def get_current_filename_hint_for_vw(self):
            return "DummyData.csv"
        
    dummy_pickaxe_instance = DummyPickaxe()
    window = VisualWorkshopApp(pickaxe_instance=dummy_pickaxe_instance)
    initial_df = dummy_pickaxe_instance.get_current_dataframe_for_vw()
    initial_hint = dummy_pickaxe_instance.get_current_filename_hint_for_vw()
    
    if initial_df is not None:
         window.receive_dataframe(initial_df, initial_hint)

    window.show()
    sys.exit(app.exec())
