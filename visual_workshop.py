# visual_workshop.py
import sys
import os
import polars as pl
import numpy as np
import datetime
import traceback
import re
from ast import literal_eval

# --- Matplotlib and Seaborn Imports ---
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Circle # Correct import for Circle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QComboBox, QScrollArea, QGroupBox, QSplitter,
    QFileDialog, QListWidget, QAbstractItemView, QCheckBox, QSpinBox, QGridLayout,
    QMessageBox, QDialog, QDialogButtonBox, QSizePolicy, QRadioButton, QButtonGroup,
    QTextEdit, QToolButton
)
from PySide6.QtCore import Qt, Slot, QSize, QUrl
from PySide6.QtGui import QAction, QIcon

# --- Local Imports ---
from scipy.stats import norm
from main import resource_path # Assuming main.py and resource_path are available
from operation_logger import logger # Assuming operation_logger.py is available


PICKAXE_LOADERS_AVAILABLE = False
try:
    # from pickaxe.common.file_loaders import FileLoaderRegistry # Example
    PICKAXE_LOADERS_AVAILABLE = False
except ImportError:
    PICKAXE_LOADERS_AVAILABLE = False

# --- Apply a neat base design for all plots ---
matplotlib.use('QtAgg')
plt.style.use('seaborn-v0_8-whitegrid')
# Use constrained_layout as the default for better automatic spacing
plt.rcParams['figure.constrained_layout.use'] = True


class CollapsibleGroupBox(QGroupBox):
    def __init__(self, title="", parent=None, checked=False):
        super().__init__(title, parent)
        self.setCheckable(True)
        
        self._main_layout = QVBoxLayout(self)
        self._main_layout.setContentsMargins(5, 10, 5, 5)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0,0,0,0)

        self._main_layout.addWidget(self.content_widget)
        
        self.toggled.connect(self._toggle_content_internal)
        self.setChecked(checked)

    def _toggle_content_internal(self, checked):
        self.content_widget.setVisible(checked)
        if checked:
            self.content_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            self.content_widget.setMaximumHeight(16777215)
        else:
            self.content_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self.content_widget.setMaximumHeight(0)
        
        self.content_widget.adjustSize() 
        self.adjustSize()

        parent = self.parentWidget()
        while parent:
            layout = parent.layout()
            if layout: # FIX: Check if layout exists before activating
                layout.activate()
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
        
        if not self.current_log_file_path:
            default_log_name = f".__log__vw_session_default_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            self.current_log_file_path = os.path.join(os.getcwd(), default_log_name)
            logger.set_log_file(self.current_log_file_path, "VisualWorkshop (Default Session)", associated_data_file="N/A")

        self.setWindowTitle("Visual Workshop - Polars Plotting Studio (Matplotlib Edition)")
        self.setGeometry(150, 150, 1400, 900)

        pickaxe_icon_path = resource_path("pickaxe.ico")
        if os.path.exists(pickaxe_icon_path):
            self.setWindowIcon(QIcon(pickaxe_icon_path))

        # --- Matplotlib Figure ---
        self.fig = Figure(figsize=(10, 8))
        self.current_plot_ax = None

        self._create_actions()
        self._create_menubar()
        self._init_ui()

        self.statusBar().showMessage("Visual Workshop ready. Load data to begin.")
        self._update_ui_for_data()

    def _init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
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
        self.data_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        data_info_hbox.addWidget(self.data_info_label, 1)

        self.plot_data_info_label = QLabel("<b>No. of Rows used for Plotting:</b> N/A")
        self.plot_data_info_label.setWordWrap(True)
        self.plot_data_info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
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
            "Density Contour", "Distribution Plot (Distplot)", "Time Series Plot"
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

        # self._update_plot_config_ui(self.plot_type_combo.currentText())

    def _create_plot_display_pane(self):
        self.plot_display_widget = QWidget()
        plot_display_layout = QVBoxLayout(self.plot_display_widget)

        self.canvas = FigureCanvas(self.fig)
        
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "Load data and configure a plot to display.", 
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_axis_off()
        self.canvas.draw()
        
        self.toolbar = NavigationToolbar(self.canvas, self)

        plot_display_layout.addWidget(self.toolbar)
        plot_display_layout.addWidget(self.canvas)

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

    def _add_basic_config_widgets(self, plot_type, layout):
        widget_adders = {
            "Scatter Plot": self._add_scatter_config, "Line Plot": self._add_line_config,
            "Time Series Plot": self._add_timeseries_config,
            "Bar Chart": self._add_bar_config, 
            "Pie Chart": self._add_pie_config,
            "Histogram": self._add_histogram_config, 
            "Box Plot": self._add_box_config,
            "Violin Plot": self._add_violin_config, 
            "Strip Plot": self._add_strip_config,
            "Density Contour": self._add_density_contour_config, 
            "Distribution Plot (Distplot)": self._add_distplot_config,
        }
        if plot_type in widget_adders: widget_adders[plot_type](layout)

    def _add_widget_pair(self, label_text, widget, layout, storage_dict, storage_key, help_text=None):
        row_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(120)
        row_layout.addWidget(label)
        row_layout.addWidget(widget, 1)

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
        ordered_columns_to_add = []
        if self.current_df is not None:
            available_cols = self._get_columns_by_type(data_types, self.current_df)
            original_df_columns = self.current_df.columns
            ordered_columns_to_add = [col for col in original_df_columns if col in available_cols]

        if include_none: combobox.addItem("None")
        combobox.addItems(ordered_columns_to_add)

        selected_idx = -1
        if default_selection_hint and default_selection_hint in ordered_columns_to_add:
            selected_idx = combobox.findText(default_selection_hint)
        elif not include_none:
            sensible_defaults = [col for col in ordered_columns_to_add if col != "__original_index__"]
            if sensible_defaults: selected_idx = combobox.findText(sensible_defaults[0])
            elif ordered_columns_to_add: selected_idx = combobox.findText(ordered_columns_to_add[0])
        elif include_none:
            selected_idx = combobox.findText("None")
            if selected_idx == -1 and combobox.count() > 0: selected_idx = 0

        if selected_idx != -1: combobox.setCurrentIndex(selected_idx)
        elif combobox.count() > 0: combobox.setCurrentIndex(0)
        elif not include_none: combobox.addItem("N/A"); combobox.setCurrentIndex(0)

    def _get_columns_by_type(self, data_types_filter, df_source):
        if df_source is None: return []

        numeric_types_pl = [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
        temporal_types_pl = [pl.Date, pl.Datetime, pl.Duration, pl.Time]
        categorical_like_types = [pl.Utf8, pl.Categorical, pl.Boolean]
        
        if data_types_filter is None or 'all' in data_types_filter:
            return [col for col in df_source.columns if col != "__original_index__"]

        selected_columns = []
        for col_name in df_source.columns:
            if col_name == "__original_index__": continue
            col_type = df_source.schema[col_name]
            match = False
            if 'numeric' in data_types_filter and col_type in numeric_types_pl: match = True
            if 'temporal' in data_types_filter and col_type in temporal_types_pl: match = True
            if 'categorical' in data_types_filter and col_type in categorical_like_types: match = True
            if 'numeric_temporal' in data_types_filter and (col_type in numeric_types_pl or col_type in temporal_types_pl): match = True
            if 'sortable' in data_types_filter: match = True

            if match: selected_columns.append(col_name)
        return selected_columns

    def _add_scatter_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        self._populate_column_combobox(self._add_widget_pair("Size by:", QComboBox(), layout, self.basic_config_widgets, "size_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Style by (Symbol):", QComboBox(), layout, self.basic_config_widgets, "symbol_col"), data_types=["categorical"])

    def _add_line_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["sortable"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])
        self._populate_column_combobox(self._add_widget_pair("Style by (Dash):", QComboBox(), layout, self.basic_config_widgets, "symbol_col"), data_types=["categorical"])
        cb = self._add_widget_pair("Sort X-axis:", QCheckBox(), layout, self.basic_config_widgets, "sort_x")
        cb.setChecked(True)

    def _add_bar_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["categorical", "all"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["all"])

        self.bar_mode_label = QLabel("Bar Mode:")
        self.bar_mode_label.setFixedWidth(120)
        self.bar_mode_combo = QComboBox()
        self.bar_mode_combo.addItems(["Dodge", "Stack"])
        self.basic_config_widgets["barmode"] = self.bar_mode_combo
        self.bar_mode_row_layout = QHBoxLayout()
        self.bar_mode_row_layout.addWidget(self.bar_mode_label)
        self.bar_mode_row_layout.addWidget(self.bar_mode_combo)
        layout.addLayout(self.bar_mode_row_layout)
        self.basic_config_widgets["color_col"].currentTextChanged.connect(self._on_bar_color_by_changed)
        self._on_bar_color_by_changed(self.basic_config_widgets["color_col"].currentText())

        agg_combo = self._add_widget_pair("Aggregation:", QComboBox(), layout, self.basic_config_widgets, "agg_func")
        agg_combo.addItems(["sum", "mean", "median", "min", "max", "count", "first", "last"])
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])

    def _on_bar_color_by_changed(self, color_col_name):
        is_categorical_color = False
        if self.current_df is not None and not self.current_df.is_empty() and color_col_name != "None" and color_col_name in self.current_df.columns:
            dtype = self.current_df.schema[color_col_name]
            if dtype in [pl.Utf8, pl.Categorical, pl.Boolean]: is_categorical_color = True
        self.bar_mode_label.setVisible(is_categorical_color)
        self.bar_mode_combo.setVisible(is_categorical_color)

    def _add_pie_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("Names/Labels:", QComboBox(), layout, self.basic_config_widgets, "names_col"), include_none=False, data_types=["categorical"])
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
        self._add_widget_pair("Donut Chart:", QCheckBox(), layout, self.basic_config_widgets, "is_donut")

    def _on_pie_mode_changed(self, mode_text):
        show_values_column = (mode_text == "Sum Values from Column")
        if hasattr(self, 'pie_values_col_label') and hasattr(self, 'pie_values_col_widget'):
            self.pie_values_col_label.setVisible(show_values_column)
            self.pie_values_col_widget.setVisible(show_values_column)

    def _add_histogram_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis/Weights:", QComboBox(), layout, self.basic_config_widgets, "y_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["categorical"])
        bins_spin = self._add_widget_pair("Number of Bins:", QSpinBox(), layout, self.basic_config_widgets, "nbinsx")
        bins_spin.setRange(0, 1000); bins_spin.setValue(10)
        norm_combo = self._add_widget_pair("Normalization:", QComboBox(), layout, self.basic_config_widgets, "histnorm")
        norm_combo.addItems(["Count", "Frequency", "Probability", "Density"])
        self._add_widget_pair("Cumulative:", QCheckBox(), layout, self.basic_config_widgets, "cumulative_enabled")
        self._add_widget_pair("Add KDE Curve:", QCheckBox(), layout, self.basic_config_widgets, "add_kde")

    def _add_box_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), data_types=["categorical", "temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["categorical"])
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])
        self._add_widget_pair("Notched:", QCheckBox(), layout, self.basic_config_widgets, "notched")

    def _add_violin_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), data_types=["categorical", "temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["categorical"])
        self._populate_column_combobox(self._add_widget_pair("Split By:", QComboBox(), layout, self.basic_config_widgets, "split_by_col"), data_types=["categorical", "boolean"])
        self._add_widget_pair("Show Box:", QCheckBox(), layout, self.basic_config_widgets, "box_visible")
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])

    def _add_strip_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), data_types=["categorical", "temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["categorical"])
        orient_combo = self._add_widget_pair("Orientation:", QComboBox(), layout, self.basic_config_widgets, "orientation")
        orient_combo.addItems(["Vertical", "Horizontal"])
        self._add_widget_pair("Add Jitter:", QCheckBox(), layout, self.basic_config_widgets, "add_jitter").setChecked(True)

    def _add_density_contour_config(self, layout):
        self._populate_column_combobox(self._add_widget_pair("X-axis:", QComboBox(), layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Y-axis:", QComboBox(), layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric_temporal"])
        self._populate_column_combobox(self._add_widget_pair("Color by (Hue):", QComboBox(), layout, self.basic_config_widgets, "color_col"), data_types=["categorical"])
        self._add_widget_pair("Fill Contour:", QCheckBox(), layout, self.basic_config_widgets, "fill_contour").setChecked(True)

    def _add_timeseries_config(self, layout):
        core_group = CollapsibleGroupBox("Core Data", checked=True)
        core_layout = core_group.get_content_layout()
        self._populate_column_combobox(self._add_widget_pair("Time Axis (X):", QComboBox(), core_layout, self.basic_config_widgets, "x_col"), include_none=False, data_types=["temporal", "sortable"])
        self._populate_column_combobox(self._add_widget_pair("Actual Values:", QComboBox(), core_layout, self.basic_config_widgets, "y_col"), include_none=False, data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Secondary Values (Y2):", QComboBox(), core_layout, self.basic_config_widgets, "y2_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Budget (Bars):", QComboBox(), core_layout, self.basic_config_widgets, "budget_col"), data_types=["numeric"])
        layout.addWidget(core_group)

        forecast_group = CollapsibleGroupBox("Forecast Data")
        forecast_layout = forecast_group.get_content_layout()
        self._populate_column_combobox(self._add_widget_pair("Forecast:", QComboBox(), forecast_layout, self.basic_config_widgets, "forecast_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Forecast Lower:", QComboBox(), forecast_layout, self.basic_config_widgets, "forecast_lower_col"), data_types=["numeric"])
        self._populate_column_combobox(self._add_widget_pair("Forecast Upper:", QComboBox(), forecast_layout, self.basic_config_widgets, "forecast_upper_col"), data_types=["numeric"])
        layout.addWidget(forecast_group)
        
        analytics_group = CollapsibleGroupBox("Overlays & Analytics")
        analytics_layout = analytics_group.get_content_layout()
        ma_widget = QWidget()
        ma_layout = QHBoxLayout(ma_widget)
        ma_checkbox = self._add_widget_pair("Moving Average:", QCheckBox(), ma_layout, self.basic_config_widgets, "add_ma")
        ma_window_spin = self._add_widget_pair("Window:", QSpinBox(), ma_layout, self.basic_config_widgets, "ma_window")
        ma_window_spin.setRange(2, 1000); ma_window_spin.setValue(7)
        ma_window_spin.setEnabled(False)
        ma_checkbox.toggled.connect(ma_window_spin.setEnabled)
        analytics_layout.addWidget(ma_widget)
        self._add_widget_pair("Add Trendline:", QCheckBox(), analytics_layout, self.basic_config_widgets, "add_trendline")
        layout.addWidget(analytics_group)
        
        cb = self._add_widget_pair("Sort Time Axis:", QCheckBox(), layout, self.basic_config_widgets, "sort_x")
        cb.setChecked(True)

    def _add_distplot_column_selector(self, parent_layout, storage_dict, storage_key):
        group_box = CollapsibleGroupBox("Data Columns for Distribution", checked=True)
        content_layout = group_box.get_content_layout()
        scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget(); scroll_layout = QVBoxLayout(scroll_widget)
        scroll_area.setWidget(scroll_widget); scroll_area.setMinimumHeight(80); scroll_area.setMaximumHeight(200)
        checkbox_list = []
        if self.current_df is not None and not self.current_df.is_empty():
            dist_cols = self._get_columns_by_type(["numeric_temporal"], self.current_df)
            if not dist_cols:
                scroll_layout.addWidget(QLabel("No suitable (numeric/temporal) columns found."))
            else:
                for col_name in dist_cols:
                    if col_name == "__original_index__": continue
                    cb = QCheckBox(col_name)
                    scroll_layout.addWidget(cb)
                    checkbox_list.append(cb)
        else:
            scroll_layout.addWidget(QLabel("No data loaded."))
        scroll_layout.addStretch(1)
        content_layout.addWidget(scroll_area)
        parent_layout.addWidget(group_box)
        storage_dict[storage_key] = checkbox_list

    def _add_distplot_config(self, layout):
        self._add_distplot_column_selector(layout, self.basic_config_widgets, "hist_data_cols_list")
        self._add_widget_pair("Show Histogram:", QCheckBox(), layout, self.basic_config_widgets, "show_hist").setChecked(True)
        self._add_widget_pair("Show KDE Curve:", QCheckBox(), layout, self.basic_config_widgets, "show_kde").setChecked(True)
        self._add_widget_pair("Show Rug Plot:", QCheckBox(), layout, self.basic_config_widgets, "show_rug").setChecked(True)
        bin_edit = self._add_widget_pair("Bin Size/Count:", QLineEdit(), layout, self.basic_config_widgets, "bin_size")
        bin_edit.setPlaceholderText("e.g., 5 (width) or 50 (count) or empty")

    def _add_advanced_config_widgets(self, layout, plot_type_name):
        self.advanced_config_widgets.clear()
        self._add_widget_pair("Plot Title:", QLineEdit(), layout, self.advanced_config_widgets, "plot_title_edit")
        self._add_widget_pair("X-axis Label:", QLineEdit(), layout, self.advanced_config_widgets, "xaxis_label_edit")
        self._add_widget_pair("Y-axis Label:", QLineEdit(), layout, self.advanced_config_widgets, "yaxis_label_edit")
        self._add_widget_pair("Legend Title:", QLineEdit(), layout, self.advanced_config_widgets, "legend_title_edit")
        self._add_widget_pair("Log X-axis:", QCheckBox(), layout, self.advanced_config_widgets, "log_x_check")
        self._add_widget_pair("Log Y-axis:", QCheckBox(), layout, self.advanced_config_widgets, "log_y_check")
        self._add_widget_pair("X-axis Range:", QLineEdit(), layout, self.advanced_config_widgets, "xaxis_range_edit").setPlaceholderText("min,max")
        self._add_widget_pair("Y-axis Range:", QLineEdit(), layout, self.advanced_config_widgets, "yaxis_range_edit").setPlaceholderText("min,max")
        
        cmap_help = "Enter any valid Matplotlib colormap name (e.g., 'viridis', 'coolwarm', 'rocket')."
        self._add_widget_pair("Colormap:", QLineEdit(), layout, self.advanced_config_widgets, "colormap_edit", help_text=cmap_help).setPlaceholderText("e.g., viridis")

        if plot_type_name not in ["Pie Chart", "Distribution Plot (Distplot)"]:
            self._populate_column_combobox(self._add_widget_pair("Facet Row:", QComboBox(), layout, self.advanced_config_widgets, "facet_row_combo"), data_types=["categorical"])
            self._populate_column_combobox(self._add_widget_pair("Facet Col:", QComboBox(), layout, self.advanced_config_widgets, "facet_col_combo"), data_types=["categorical"])

    def _add_pie_advanced_config(self, layout):
        self.pie_advanced_widgets.clear()
        self._add_widget_pair("Plot Title:", QLineEdit(), layout, self.pie_advanced_widgets, "plot_title_edit")
        self._add_widget_pair("Legend Title:", QLineEdit(), layout, self.pie_advanced_widgets, "legend_title_edit")
        self._add_widget_pair("Explode Slice (CSV):", QLineEdit(), layout, self.pie_advanced_widgets, "explode_edit").setPlaceholderText("e.g., 0,0,0.1,0")
        cmap_help = "Enter any valid Matplotlib colormap name (e.g., 'viridis', 'coolwarm', 'rocket')."
        self._add_widget_pair("Colormap:", QLineEdit(), layout, self.pie_advanced_widgets, "colormap_edit", help_text=cmap_help).setPlaceholderText("e.g., viridis")

    def _add_distplot_advanced_config(self, layout):
        self.distplot_advanced_widgets.clear()
        self._add_widget_pair("Plot Title:", QLineEdit(), layout, self.distplot_advanced_widgets, "plot_title_edit")
        self._add_widget_pair("X-axis Label:", QLineEdit(), layout, self.distplot_advanced_widgets, "xaxis_label_edit")
        self._add_widget_pair("Y-axis Label:", QLineEdit(), layout, self.distplot_advanced_widgets, "yaxis_label_edit")
        self._add_widget_pair("Legend Title:", QLineEdit(), layout, self.distplot_advanced_widgets, "legend_title_edit")
        cmap_help = "Enter any valid Matplotlib colormap name (e.g., 'viridis', 'coolwarm', 'rocket')."
        self._add_widget_pair("Colormap:", QLineEdit(), layout, self.distplot_advanced_widgets, "colormap_edit", help_text=cmap_help).setPlaceholderText("e.g., viridis")

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
        logger.log_action("VisualWorkshop", "Data Received", f"Received data '{os.path.basename(filename_hint)}'",
            details={"Source Hint": filename_hint, "Rows": polars_df.height, "Columns": polars_df.width, "Log File In Use": self.current_log_file_path})
        self.statusBar().showMessage(f"Data received: {filename_hint}. Dimensions: {self.current_df.shape}")
        print_vw_filename = os.path.basename(filename_hint);
        if len(print_vw_filename) > 30: print_vw_filename = print_vw_filename[:30] + "..."
        self.data_info_label.setText(f"<b>Source:</b> {print_vw_filename}\nDimensions: {self.current_df.height} rows, {self.current_df.width} cols")
        self._update_ui_for_data()
        # self._update_plot_config_ui(self.plot_type_combo.currentText())

    def refresh_data_from_pickaxe(self):
        if self.source_app and hasattr(self.source_app, 'get_current_dataframe_for_vw') and \
           hasattr(self.source_app, 'get_current_filename_hint_for_vw'):
            df = self.source_app.get_current_dataframe_for_vw()
            hint = self.source_app.get_current_filename_hint_for_vw()
            log_path = None
            if hasattr(self.source_app, 'get_current_log_file_path'): log_path = self.source_app.get_current_log_file_path()
            if df is not None:
                self.receive_dataframe(df, hint, log_file_path_from_source=log_path)
                self.statusBar().showMessage("Data refreshed from Pickaxe.", 3000)
            else:
                title = self.source_app.windowTitle() if hasattr(self.source_app, 'windowTitle') else 'Source App'
                QMessageBox.information(self, "No Data", f"{title} has no active data to refresh.")
        else:
            QMessageBox.warning(self, "Integration Error", "Cannot refresh. Source application instance not available or methods missing.")

    def open_file_directly(self):
        df_loaded, file_name_loaded = None, None
        # FIX: Changed third argument from None to "" for type safety
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;Parquet Files (*.parquet)")
        if file_name:
            file_name_loaded = file_name
            try:
                if file_name.lower().endswith('.csv'): df_loaded = pl.read_csv(file_name)
                elif file_name.lower().endswith(('.xlsx', '.xls')): df_loaded = pl.read_excel(file_name, infer_schema_length=0)
                elif file_name.lower().endswith('.parquet'): df_loaded = pl.read_parquet(file_name)
                else: QMessageBox.warning(self, "Unsupported File", "Only CSV, Excel, and Parquet files are supported."); return
            except Exception as e: QMessageBox.critical(self, "File Load Error", f"Could not load file: {str(e)}"); return
        if df_loaded is not None and file_name_loaded:
            try:
                log_file_dir = os.path.dirname(os.path.abspath(file_name_loaded))
                data_basename_no_ext = os.path.splitext(os.path.basename(file_name_loaded))[0]
                timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                log_filename_for_data = f".__log__{data_basename_no_ext}_{timestamp_str}.md"
                self.current_log_file_path = os.path.join(log_file_dir, log_filename_for_data)
                logger.set_log_file(self.current_log_file_path, "VisualWorkshop (Direct Open)", associated_data_file=file_name_loaded)
                logger.log_action("VisualWorkshop", "Data Session Start", f"Processing data file: {os.path.basename(file_name_loaded)}", details={"Log file": self.current_log_file_path})
            except Exception as e:
                print(f"VW: Error setting up logger for direct open {file_name_loaded}: {e}")
                default_log_name = f".__log__vw_fallback_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                self.current_log_file_path = os.path.join(os.getcwd(), default_log_name)
                logger.set_log_file(self.current_log_file_path, "VisualWorkshop (Fallback Log)", associated_data_file=file_name_loaded)
            self.receive_dataframe(df_loaded, file_name_loaded)

    def _update_ui_for_data(self):
        has_data = self.current_df is not None and not self.current_df.is_empty()
        self.generate_plot_button.setEnabled(has_data)
        if has_data:
            plot_type = self.plot_type_combo.currentText()
            self._update_plot_config_ui(plot_type)

    def _calculate_representative_sample_size(self, df, target_column_name, margin_error_percent=5, confidence_level_percent=95):
        if target_column_name not in df.columns: self.statusBar().showMessage(f"Target column '{target_column_name}' for sampling not found. Using max 10k rows.", 3000); return min(10000, df.height)
        series = df.get_column(target_column_name).drop_nulls()
        if series.is_empty(): self.statusBar().showMessage(f"Target column '{target_column_name}' is empty. Using max 10k rows for sampling.", 3000); return min(10000, df.height)
        n_total = df.height
        if n_total == 0: return 0
        confidence_level = confidence_level_percent / 100.0; Z = 1.96
        try: Z = norm.ppf(1 - (1 - confidence_level) / 2)
        except ImportError: self.statusBar().showMessage("Scipy not found, Z-score may not match selected confidence. Using 1.96 for 95% Confidence.", 2000);
        except Exception: self.statusBar().showMessage("Scipy.stats.norm not available, Z-score may not match. Using 1.96 for 95% Confidence.", 2000)
        sigma_or_p_estimate = 0.5; dtype = series.dtype; margin_error_prop = margin_error_percent / 100.0; E = margin_error_prop
        if dtype.is_numeric():
            std_dev = series.std()
            if std_dev is not None and std_dev > 0:
                sigma_or_p_estimate = std_dev; mean_val = series.mean()
                if mean_val is not None and abs(mean_val) > 1e-9: E = margin_error_prop * abs(mean_val)
                else:
                    data_range = series.max() - series.min()
                    if data_range is not None and data_range > 1e-9: E = margin_error_prop * data_range
                    else: self.statusBar().showMessage(f"Cannot determine sensible margin of error for numeric '{target_column_name}'. Using high variability.", 3000); E = sigma_or_p_estimate * 0.1
                if E == 0: E = 1e-6
            else: self.statusBar().showMessage(f"No variability in numeric column '{target_column_name}'. Cannot calculate representative sample meaningfully.", 3000); return n_total
        if E == 0: E = 1e-6
        try:
            if dtype.is_numeric(): n0 = ((Z * sigma_or_p_estimate) / E)**2
            else: n0 = (Z**2 * 0.5 * 0.5) / (E**2)
            calculated_n = n0 / (1 + (n0 - 1) / n_total) if n_total > n0 else n0
            sample_s = max(1, min(int(np.ceil(calculated_n)), n_total))
            self.statusBar().showMessage(f"Calculated representative sample size: {sample_s} for '{target_column_name}' (CI: {confidence_level_percent}%, MoE: {margin_error_percent}%)", 5000)
            return sample_s
        except (OverflowError, ZeroDivisionError, ValueError) as e_calc: self.statusBar().showMessage(f"Error calculating sample size for '{target_column_name}': {e_calc}. Using random 10k.", 3000); return min(10000, n_total)

    @Slot()
    def generate_plot(self):
        if self.current_df is None or self.current_df.is_empty():
            QMessageBox.information(self, "No Data", "Please load data before generating a plot.")
            return

        self.statusBar().showMessage("Generating plot...", 2000)
        QApplication.processEvents()

        plot_type = self.plot_type_combo.currentText()
        df_to_plot = self.current_df
        if self.sample_random_rb.isChecked():
            if self.current_df.height > 10000:
                self.statusBar().showMessage(f"Randomly sampling data from {self.current_df.height} to 10,000 rows...", 3000)
                df_to_plot = self.current_df.sample(n=10000, shuffle=True, seed=42)
        elif self.sample_representative_rb.isChecked():
            margin_error_val = self.margin_error_input_rep.value()
            confidence_level_val = int(self.ci_combo.currentText())
            basic_args_temp = {k: self._get_widget_value(w) for k, w in self.basic_config_widgets.items()}
            target_col_for_sampling = None
            if plot_type in ["Scatter Plot", "Line Plot", "Bar Chart", "Box Plot", "Violin Plot", "Strip Plot", "Time Series Plot"]: target_col_for_sampling = basic_args_temp.get('y_col')
            elif plot_type == "Histogram": target_col_for_sampling = basic_args_temp.get('x_col')
            elif plot_type == "Pie Chart": target_col_for_sampling = basic_args_temp.get('values_col') if basic_args_temp.get('pie_mode') == "Sum Values from Column" else basic_args_temp.get('names_col')
            elif plot_type == "Density Contour": target_col_for_sampling = basic_args_temp.get('y_col')
            elif plot_type == "Distribution Plot (Distplot)":
                selected_dist_cols = basic_args_temp.get('hist_data_cols_list', [])
                if selected_dist_cols and isinstance(selected_dist_cols, list) and selected_dist_cols: # Pylance guard
                    target_col_for_sampling = selected_dist_cols[0]
            
            if target_col_for_sampling and target_col_for_sampling != "None" and target_col_for_sampling in df_to_plot.columns:
                calculated_sample_size = self._calculate_representative_sample_size(df_to_plot, target_col_for_sampling, margin_error_val, confidence_level_val)
                if calculated_sample_size < df_to_plot.height:
                    df_to_plot = df_to_plot.sample(n=calculated_sample_size, shuffle=True, seed=42)
            else:
                self.statusBar().showMessage("No valid target column for representative sampling. Using random 10k max if applicable.", 3000)
                if df_to_plot.height > 10000: df_to_plot = df_to_plot.sample(n=10000, shuffle=True, seed=42)
        
        if df_to_plot.is_empty():
            QMessageBox.warning(self, "Sampling Error", "Data became empty after sampling. Cannot generate plot."); return
        self.plot_data_info_label.setText(f"<b>No. of Rows used for Plotting:</b></br> {df_to_plot.height} <br/>")

        basic_args = {k: self._get_widget_value(w) for k, w in self.basic_config_widgets.items()}
        advanced_args = {}
        adv_config_source_dict, current_advanced_group = {}, None
        if plot_type == "Distribution Plot (Distplot)": adv_config_source_dict, current_advanced_group = self.distplot_advanced_widgets, self.distplot_advanced_group
        elif plot_type == "Pie Chart": adv_config_source_dict, current_advanced_group = self.pie_advanced_widgets, self.pie_advanced_group
        else: adv_config_source_dict, current_advanced_group = self.advanced_config_widgets, self.advanced_group
        if current_advanced_group and current_advanced_group.isChecked(): advanced_args = {k: self._get_widget_value(w) for k, w in adv_config_source_dict.items()}
        
        try:
            self.fig.clear()
            pd_df = df_to_plot.to_pandas()

            facet_row = advanced_args.get('facet_row_combo', "None")
            facet_col = advanced_args.get('facet_col_combo', "None")
            use_facets = (facet_row != "None" or facet_col != "None") and plot_type not in ["Pie Chart", "Distribution Plot (Distplot)"]

            if use_facets:
                row_cats = pd_df[facet_row].dropna().unique() if facet_row != "None" else [None]
                col_cats = pd_df[facet_col].dropna().unique() if facet_col != "None" else [None]
                axs = self.fig.subplots(len(row_cats), len(col_cats), sharex=True, sharey=True, squeeze=False)

                for r_idx, r_cat in enumerate(row_cats):
                    for c_idx, c_cat in enumerate(col_cats):
                        ax = axs[r_idx, c_idx]
                        facet_df = pd_df
                        if r_cat is not None: facet_df = facet_df[facet_df[facet_row] == r_cat]
                        if c_cat is not None: facet_df = facet_df[facet_df[facet_col] == c_cat]
                        
                        if facet_df.empty: ax.set_axis_off(); continue

                        self._draw_plot_on_ax(ax, plot_type, facet_df, basic_args, advanced_args)
                        
                        if r_idx == 0 and c_cat is not None: ax.set_title(str(c_cat))
                        if c_idx == 0 and r_cat is not None: ax.set_ylabel(str(r_cat), rotation=0, size='large', ha='right')

                self.current_plot_ax = axs
            else:
                ax = self.fig.add_subplot(111)
                self._draw_plot_on_ax(ax, plot_type, pd_df, basic_args, advanced_args)
                self.current_plot_ax = ax

            self._apply_matplotlib_layout(self.fig, self.current_plot_ax, advanced_args, basic_args, plot_type)
            self.canvas.draw()
            self.statusBar().showMessage("Plot generated successfully.", 3000)
            self.save_plot_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Plot Generation Error", f"An error occurred: {str(e)}")
            self.statusBar().showMessage(f"Error generating plot: {str(e)}", 5000)
            print(f"Plot Generation Error: {str(e)}\n{traceback.format_exc()}")
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', color='red', wrap=True)
            self.canvas.draw()
            self.current_plot_ax = None
            self.save_plot_button.setEnabled(False)

    def _draw_plot_on_ax(self, ax, plot_type, df_pd, basic_args, advanced_args):
        plot_drawers = {
            "Scatter Plot": self._draw_scatter, "Line Plot": self._draw_line, "Bar Chart": self._draw_bar,
            "Pie Chart": self._draw_pie, "Histogram": self._draw_histogram, "Box Plot": self._draw_box,
            "Violin Plot": self._draw_violin, "Strip Plot": self._draw_strip, "Density Contour": self._draw_density,
            "Distribution Plot (Distplot)": self._draw_distplot, "Time Series Plot": self._draw_timeseries,
        }
        if plot_type in plot_drawers:
            plot_drawers[plot_type](ax, df_pd, basic_args, advanced_args)

    def save_current_plot(self):
        if self.current_plot_ax is None:
            QMessageBox.warning(self, "No Plot", "No plot has been generated yet to save.")
            return

        default_filename = f"{self.plot_type_combo.currentText().replace(' ', '_')}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Plot As", default_filename,
            "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Image (*.svg);;PDF Document (*.pdf)")

        if not file_name: return
        try:
            self.fig.savefig(file_name, dpi=300, bbox_inches='tight')
            self.statusBar().showMessage(f"Plot saved to {file_name}", 3000)
            logger.log_action("VisualWorkshop", "Plot Saved", f"Plot saved as {os.path.basename(file_name)}", details={"Path": file_name})
        except Exception as e:
            QMessageBox.critical(self, "Save Plot Error", f"Could not save plot: {str(e)}")
            logger.log_action("VisualWorkshop", "Plot Save Error", f"Error saving plot: {os.path.basename(file_name)}", details={"Error": str(e)})

    def _get_seaborn_kwargs(self, basic_args, advanced_args):
        kwargs = {}
        if basic_args.get('color_col') != "None": kwargs['hue'] = basic_args.get('color_col')
        if basic_args.get('symbol_col') != "None": kwargs['style'] = basic_args.get('symbol_col')
        if basic_args.get('size_col') != "None": kwargs['size'] = basic_args.get('size_col')
        if advanced_args.get('colormap_edit'): kwargs['palette'] = advanced_args.get('colormap_edit')
        return kwargs
    
    def _draw_scatter(self, ax, df, basic, adv):
        sns.scatterplot(data=df, x=basic.get('x_col'), y=basic.get('y_col'), ax=ax, **self._get_seaborn_kwargs(basic, adv))

    def _draw_line(self, ax, df, basic, adv):
        df_plot = df.sort_values(by=basic.get('x_col')) if basic.get('sort_x', True) else df
        sns.lineplot(data=df_plot, x=basic.get('x_col'), y=basic.get('y_col'), ax=ax, marker='o', **self._get_seaborn_kwargs(basic, adv))

    def _draw_bar(self, ax, df, basic, adv):
        orient = 'v' if basic.get('orientation') == 'Vertical' else 'h'
        x, y = (basic.get('x_col'), basic.get('y_col')) if orient == 'v' else (basic.get('y_col'), basic.get('x_col'))
        agg_map = {'mean': np.mean, 'sum': np.sum, 'median': np.median}
        estimator = agg_map.get(basic.get('agg_func'), 'mean') if basic.get('y_col') != 'None' else 'count'
        sns.barplot(data=df, x=x, y=y, ax=ax, orient=orient, hue=basic.get('color_col') if basic.get('color_col') != 'None' else None,
                    palette=adv.get('colormap_edit') or None, estimator=estimator)

    def _draw_pie(self, ax, df, basic, adv):
        names_col, mode = basic.get('names_col'), basic.get('pie_mode')
        if mode == "Count Occurrences (Rows)":
            data = df[names_col].value_counts()
        else:
            values_col = basic.get('values_col')
            data = df.groupby(names_col)[values_col].sum()
        
        labels, values = data.index, data.values
        explode = None
        if adv.get('explode_edit'):
            try: explode = [float(x.strip()) for x in adv.get('explode_edit').split(',')]
            except: self.statusBar().showMessage("Invalid explode format.", 2000)

        ax.pie(values, labels=labels, autopct='%1.1f%%', explode=explode, shadow=True, startangle=90)
        if basic.get('is_donut'):
            ax.add_artist(Circle((0,0),0.70,fc='white'))
        ax.axis('equal')
        ax.set_title(adv.get('plot_title_edit', 'Pie Chart'))

    def _draw_histogram(self, ax, df, basic, adv):
        sns.histplot(data=df, x=basic.get('x_col'), weights=basic.get('y_col') if basic.get('y_col') != 'None' else None,
                     hue=basic.get('color_col') if basic.get('color_col') != 'None' else None,
                     bins=basic.get('nbinsx') or 'auto', kde=basic.get('add_kde', False),
                     cumulative=basic.get('cumulative_enabled', False), stat=basic.get('histnorm', 'count').lower(),
                     ax=ax, palette=adv.get('colormap_edit') or None)

    def _draw_box(self, ax, df, basic, adv):
        orient = 'v' if basic.get('orientation') == 'Vertical' else 'h'
        x, y = (basic.get('x_col'), basic.get('y_col')) if orient == 'v' else (basic.get('y_col'), basic.get('x_col'))
        if x == 'None': x = None
        sns.boxplot(data=df, x=x, y=y, hue=basic.get('color_col') if basic.get('color_col') != 'None' else None,
                    orient=orient, notch=basic.get('notched', False), ax=ax, palette=adv.get('colormap_edit') or None)

    def _draw_violin(self, ax, df, basic, adv):
        orient = 'v' if basic.get('orientation') == 'Vertical' else 'h'
        x, y = (basic.get('x_col'), basic.get('y_col')) if orient == 'v' else (basic.get('y_col'), basic.get('x_col'))
        if x == 'None': x = None
        split = False
        hue = basic.get('color_col') if basic.get('color_col') != 'None' else None
        if basic.get('split_by_col') != 'None':
            hue = basic.get('split_by_col')
            if len(df[hue].unique()) == 2: split = True
        sns.violinplot(data=df, x=x, y=y, hue=hue, split=split, orient=orient, 
                       inner='box' if basic.get('box_visible') else 'quartile', ax=ax, palette=adv.get('colormap_edit') or None)

    def _draw_strip(self, ax, df, basic, adv):
        orient = 'v' if basic.get('orientation') == 'Vertical' else 'h'
        x, y = (basic.get('x_col'), basic.get('y_col')) if orient == 'v' else (basic.get('y_col'), basic.get('x_col'))
        if x == 'None': x = None
        sns.stripplot(data=df, x=x, y=y, hue=basic.get('color_col') if basic.get('color_col') != 'None' else None,
                      orient=orient, jitter=basic.get('add_jitter', True), ax=ax, palette=adv.get('colormap_edit') or None)

    def _draw_density(self, ax, df, basic, adv):
        sns.kdeplot(data=df, x=basic.get('x_col'), y=basic.get('y_col'),
                    hue=basic.get('color_col') if basic.get('color_col') != 'None' else None,
                    fill=basic.get('fill_contour', True), ax=ax, palette=adv.get('colormap_edit') or None)

    def _draw_distplot(self, ax, df, basic, adv):
        cols_to_plot = basic.get('hist_data_cols_list', [])
        if not cols_to_plot:
            ax.text(0.5, 0.5, "No columns selected.", ha='center', va='center'); return
        for col in cols_to_plot:
            sns.kdeplot(data=df, x=col, ax=ax, label=col, fill=basic.get('show_hist', True), 
                        cut=0, cumulative=False, 
                        # rug=basic.get('show_rug', False)
                        )
        if len(cols_to_plot) > 1: ax.legend()

    def _draw_timeseries(self, ax, df, basic, adv):
        x_col, y_col = basic.get('x_col'), basic.get('y_col')
        df_plot = df.sort_values(by=x_col) if basic.get('sort_x', True) else df
        if basic.get('budget_col') != "None": ax.bar(df_plot[x_col], df_plot[basic.get('budget_col')], label='Budgeted', color='sandybrown', alpha=0.6)
        if basic.get('forecast_lower_col') != "None" and basic.get('forecast_upper_col') != "None":
            ax.fill_between(df_plot[x_col], df_plot[basic.get('forecast_lower_col')], df_plot[basic.get('forecast_upper_col')], color='gray', alpha=0.3, label='Forecast CI')
        if basic.get('forecast_col') != "None": ax.plot(df_plot[x_col], df_plot[basic.get('forecast_col')], 'k--', label='Forecast')
        if basic.get('add_ma'):
            window = basic.get('ma_window', 7)
            ma_series = df_plot[y_col].rolling(window=window).mean()
            ax.plot(df_plot[x_col], ma_series, color='lightseagreen', label=f'{window}-period MA')
        if basic.get('add_trendline'):
            sns.regplot(data=df_plot, x=x_col, y=y_col, ax=ax, scatter=False, color='purple', line_kws={'linestyle':'--'}, label='Trendline')
        ax.plot(df_plot[x_col], df_plot[y_col], 'o-', color='royalblue', label='Actual')
        if basic.get('y2_col') != "None":
            ax2 = ax.twinx()
            ax2.plot(df_plot[x_col], df_plot[basic.get('y2_col')], '.-', color='firebrick', label=basic.get('y2_col'))
            ax2.set_ylabel(basic.get('y2_col'))
            ax2.legend(loc='upper right')
        ax.legend(loc='upper left')

    def _apply_matplotlib_layout(self, fig, ax, advanced_args, basic_args, plot_type):
        if ax is None: return
        axs = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
        title = advanced_args.get('plot_title_edit') or f"{plot_type}"
        fig.suptitle(title, fontsize=16)
        
        main_ax = axs[0]
        main_ax.set_xlabel(advanced_args.get('xaxis_label_edit') or basic_args.get('x_col'))
        main_ax.set_ylabel(advanced_args.get('yaxis_label_edit') or basic_args.get('y_col'))
        
        legend_title = advanced_args.get('legend_title_edit')
        if not legend_title:
            if basic_args.get('color_col', 'None') != 'None': legend_title = basic_args.get('color_col')
            elif basic_args.get('symbol_col', 'None') != 'None': legend_title = basic_args.get('symbol_col')

        if legend_title:
            try:
                handles, labels = main_ax.get_legend_handles_labels()
                if handles: main_ax.legend(title=legend_title)
            except (AttributeError, IndexError): pass

        for current_ax in axs:
            if advanced_args.get('log_x_check'): current_ax.set_xscale('log')
            if advanced_args.get('log_y_check'): current_ax.set_yscale('log')
            try:
                if advanced_args.get('xaxis_range_edit'):
                    min_val, max_val = map(float, advanced_args.get('xaxis_range_edit').split(','))
                    current_ax.set_xlim(min_val, max_val)
                if advanced_args.get('yaxis_range_edit'):
                    min_val, max_val = map(float, advanced_args.get('yaxis_range_edit').split(','))
                    current_ax.set_ylim(min_val, max_val)
            except (ValueError, IndexError):
                self.statusBar().showMessage("Invalid axis range format. Use 'min,max'.", 2000)

    def _get_widget_value(self, widget):
        if isinstance(widget, QComboBox): return widget.currentText()
        if isinstance(widget, QLineEdit): return widget.text()
        if isinstance(widget, QTextEdit): return widget.toPlainText()
        if isinstance(widget, QCheckBox): return widget.isChecked()
        if isinstance(widget, QSpinBox): return widget.value()
        if isinstance(widget, list) and all(isinstance(item, QCheckBox) for item in widget): return [cb.text() for cb in widget if cb.isChecked()]
        if isinstance(widget, QListWidget): return [item.text() for item in widget.selectedItems()]
        return None

    def closeEvent(self, event):
        plt.close(self.fig)
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    class DummyPickaxe:
        def get_current_dataframe_for_vw(self):
            import pandas as pd
            np.random.seed(42)
            n_rows = 120
            # FIX: Use np.array with dtype=object for lists containing None
            data = {
                'categoryA': np.random.choice(np.array(['Alpha', 'Beta', 'Gamma', 'Delta', None], dtype=object), n_rows, p=[0.2,0.3,0.25,0.15,0.1]),
                'categoryB': np.random.choice(np.array(['X', 'Y', 'Z', None], dtype=object), n_rows, p=[0.4,0.3,0.2,0.1]),
                'value1': np.random.randn(n_rows) * 100 - 20,
                'value2': np.random.rand(n_rows) * 50 + 10,
                'size_col': np.random.rand(n_rows) * 20 + 5,
                'time_data': pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_rows, freq='D')),
                'date_data': pd.to_datetime(pd.date_range(start='2022-01-01', periods=n_rows, freq='D')).date,
                'bool_data': np.random.choice(np.array([True, False, None], dtype=object), n_rows, p=[0.45,0.4,0.15]),
            }
            df = pl.from_pandas(pd.DataFrame(data))
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