# PickAxe: A Data File Viewer Application

## Overview

This application is a standalone tool designed for viewing and manipulating data files such as Excel and CSV files. It provides a user-friendly interface built with PySide6, allowing users to load, view, and filter data.

## Features

- **File Loading**: Supports loading of various file formats including `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, and `.csv`.
- **Data Visualisation**: Displays data in a tabular format with options to parse numbers and date into view.
- **Filtering**: Allows users to apply filters to the data for focused analysis by numeric, date and regex filtering.
- **User Interface**: Easy GUI with support for themes and custom icons.
- **Performance**: Optimized for handling large datasets with efficient loading and processing.

## Installation

### Prerequisites

- Python 3.6 or higher
- PySide6
- pandas
- numpy
- pyarrow
- chardet
- openpyxl
- pyxlsb
- xlrd

### Steps

1. Clone the repository to your local machine.
2. Install the required dependencies using pip:
   ```bash
   pip install pandas numpy PySide6 pyarrow chardet openpyxl pyxlsb xlrd
   ```
3. Run the application using the starter script:
   ```bash
   python starter.py
   ```

## Usage

1. Launch the application.
2. Use the file dialog to open a data file.
3. View and manipulate the data using the provided tools and filters.
