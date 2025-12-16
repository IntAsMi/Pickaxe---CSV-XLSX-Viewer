# Pickaxe

Pickaxe is a powerful and user-friendly data viewer and manipulation tool, now replacing Pandas library with the Polars library. It allows users to open, view, filter, analyze, and save large CSV and Excel files quickly, efficiently, all in a single installer executable file for this application only is required.

## Features

-   **Polars-Powered Backend**: Utilizes the high-performance Polars DataFrame library for exceptional speed and memory efficiency.
-   **Multi-format Support**: Open and read CSV, XLSX, XLS, XLSM, and XLSB files.
-   **Large File Handling**:
    *   Efficiently processes and displays large datasets.
    *   Automatic conversion of very large Excel files to temporary CSVs for optimized loading with Polars.
    *   Robust CSV dialect and encoding detection.
-   **Advanced Data Filtering**:
    *   **Structured Filters:** Intuitive UI to build complex filter queries with multiple conditions (AND/OR logic).
        *   Type-specific filtering options for String, Date, and Numeric data.
        *   String filters support **case sensitivity, regex, and negation**.
        *   Special column comparisons like "Is Duplicate Value" or "Is Unique Value".
    *   **Direct Polars Querying:** Input Polars expressions directly for advanced filtering, with **auto-completion** for syntax and column names.
    *   **Reset All Filters:** Easily revert to the original dataset.
-   **In-Depth Column Analysis**:
    *   Select any column to view detailed statistics: fill rate (visual bar), counts (empty, non-empty, unique), unique value distribution, numeric summaries (mean, median, std, etc.), and date/datetime ranges.
-   **Powerful Data Manipulation (via Context Menus)**:
    *   **Sort Data:** Sort the entire dataset by any column.
    *   **Set Row as Header:** Promote any data row to become the new column headers.
    *   **Filter Duplicate Rows:** Identify and filter rows that are complete duplicates across all columns (options to keep first, last, or remove all).
    *   **Column Type Conversion:** Convert column(s) to Numeric or Datetime types, with batch conversion support for multiple columns.
-   **Efficient Data Navigation**:
    *   Easily navigate through datasets with many columns using "Previous/Next Columns" buttons (defaults to 100 columns per page).
-   **Data Export**: Save filtered or modified data to CSV or Excel (XLSX) formats.
    *   Intelligent suggested filenames based on applied filters.
-   **User-friendly Interface**:
    *   Clean and responsive GUI.
    *   Progress indicators for long-running operations (file loading, filtering, saving), keeping the UI responsive.
    *   Toggleable filter panel.
    *   Clear display of file information, original dimensions, and current view details.

## Getting Started

1.  Download the `Pickaxe.exe` file from the Releases page of its repository.
2.  Double-click the downloaded file to run Pickaxe.

Only installation of this application is required!

## Usage

### Opening a File

1.  Click the "Open File" button.
2.  Select your CSV or Excel file.
3.  If it's an Excel file with multiple sheets, you'll be prompted to select a sheet. Pickaxe will then load and display the data.

### Viewing Data

-   The data is displayed in a table view.
-   Use the "Previous X Columns" and "Next X Columns" buttons (where X is typically 100) to navigate horizontally if your dataset has many columns.
-   The information panel displays details about the loaded file and the current view (number of rows/columns shown).

### Filtering Data

Pickaxe offers two ways to filter your data:

1.  **Structured Filters**:
    *   Click "Show/Hide Structured Filters" to display the filter panel.
    *   Click "Add Filter Row" to add a new filter condition.
    *   For each filter row:
        *   Select a **column**.
        *   Choose the **field type** (String, Date, Numeric) - Pickaxe often infers this.
        *   Select a **comparison operator** (e.g., Contains, Equals, >, <, Between, Is Duplicate).
        *   Enter the **filter value(s)**. For strings, you can specify case sensitivity or use Regex, and negate the filter.
        *   If you have multiple filter rows, define the logic (AND/OR) between them.
    *   Click "Apply Structured Filters". The table will update, and the Polars query input below might show the equivalent Polars expression.

2.  **Polars Query Input**:
    *   Directly type a Polars filter expression into the input field (e.g., `pl.col('Age') > 30 & pl.col('City') == 'New York'`).
    *   Use Tab or Ctrl+Space for auto-completion suggestions.
    *   Click "Apply Query".
    *   Click the "?" button for a link to Polars expression documentation.

To clear all filters and sorting, click "Reset All Filters/Sorting".

### Analyzing Data

-   Click on any column header in the table view.
-   The "Statistics" panel below the table will update to show detailed statistics for the selected column, including fill rate, unique counts, and type-specific summaries.

### Manipulating Data (Context Menus)

-   **Right-click on a column header** to:
    *   Sort the entire dataset by that column.
    *   Filter duplicate rows (based on all data).
    *   Convert the column's data type (or multiple selected columns).
-   **Right-click on a data row in the table** to:
    *   Set that row as the new header for your dataset.

### Saving Data

1.  After filtering or manipulating your data, click the "Save File" button.
2.  Choose a location and file name. Pickaxe suggests a name that includes details about applied filters.
3.  Select the file format (CSV or XLSX).

## Troubleshooting

-   If you encounter any issues while running Pickaxe, ensure you have the necessary permissions to run `.exe` files on your system.
-   Some antivirus software may flag new `.exe` files. You may need to add Pickaxe to your antivirus exceptions if this occurs.
-   For any persistent issues, please contact me or send me an email at .

## Contributing

While Pickaxe is often distributed as a compiled executable, the source code is available in its repository. Contributions to improve Pickaxe are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Pickaxe is built using these fantastic open-source libraries:

-   **PySide6**: For the graphical user interface.
-   **Polars**: For high-performance DataFrame operations.
-   **openpyxl, xlrd, pyxlsb**: For reading various Excel file formats.
-   **chardet**: For character encoding detection.
-   **xlsx2csv**: For converting large Excel files.