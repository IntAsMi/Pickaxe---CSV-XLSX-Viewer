# pip install pandas numpy PySide6 pyarrow chardet openpyxl pyxlsb xlrd python-calamine
import sys, os, re
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QLineEdit, QPushButton, QTableView, QHeaderView, QFileDialog, QRadioButton,
                               QComboBox, QMenu, QInputDialog, QMessageBox, QProgressBar,
                               QScrollArea, QFrame, QDateEdit, QSizePolicy, QCheckBox)
from PySide6.QtCore import Qt, QAbstractTableModel, QDate, QThread, Signal, Slot
from PySide6.QtGui import QAction, QIcon

import pyarrow.csv as pyspcsv
import pyarrow as pa
import time
import chardet
import csv
import mmap
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pyxlsb
import xlrd
import tempfile 
from datetime import datetime
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import codecs
import io
import multiprocessing
import contextlib
from pathlib import Path
import zipfile
import openpyxl

# signal the splash screen removal.
if "NUITKA_ONEFILE_PARENT" in os.environ:
   splash_filename = os.path.join(
      tempfile.gettempdir(),
      "onefile_%d_splash_feedback.tmp" % int(os.environ["NUITKA_ONEFILE_PARENT"]),
   )
 
   if os.path.exists(splash_filename):
      os.unlink(splash_filename)
 
# print("Splash Screen has been removed")

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f"\n >> {func.__name__} took {time.time() - start} seconds <<\n")
        return res
    return wrapper

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    # try:
    #     # PyInstaller creates a temp folder and stores path in _MEIPASS
    #     base_path = sys._MEIPASS
    # except Exception:
    #     base_path = os.path.abspath(".")
    base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def get_sheet_names(file_name):
    """Get sheet names from Excel files regardless of format."""
    if file_name.lower().endswith('.xls'):
        with xlrd.open_workbook(file_name, on_demand=True) as workbook:
            return workbook.sheet_names()
    
    elif file_name.lower().endswith('.xlsb'):
        with pyxlsb.open_workbook(file_name) as workbook:
            return workbook.sheets
    
    elif file_name.lower().endswith(('.xlsx', '.xlsm')):
        try:
            with zipfile.ZipFile(file_name) as archive:
                tree = ET.parse(archive.open('xl/workbook.xml'))
                root = tree.getroot()
                ns = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
                sheets = root.findall(".//main:sheet", ns)
                return [sheet.get('name') for sheet in sheets]
        except Exception:
            with openpyxl.load_workbook(file_name, read_only=True, data_only=True) as workbook:
                return workbook.sheetnames
    else:
        raise ValueError(f"Unsupported file format: {file_name}")


def run_FileLoaderWorker(args):
    return FileLoaderWorker(*args).run()
    
class FileLoaderWorker:
    def __init__(self, file_path, sheet_name=None, usecols = None, header_row = None, dtype = str, data_only=True, read_only=True, nrows = None, run_wd = None):
        super().__init__()
        self.file_path = file_path
        self._file_path_ = file_path
        self.sheet_name = sheet_name
        self.usecols = usecols
        self.header_row = header_row
        self.dtype = dtype
        self.data_only = data_only
        self.read_only = read_only
        self.nrows = nrows
        self.run_wd = run_wd

    def wait_for_file_access(self, file_path, max_attempts=10, delay=1):
        """Wait for file to be accessible and ready"""
        for attempt in range(max_attempts):
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    raise FileNotFoundError("Output file does not exist")
                    
                # Check if file is empty
                if os.path.getsize(file_path) == 0:
                    raise ValueError("Output file is empty")
                    
                # Try to open and read file
                with open(file_path, 'rb') as f:
                    f.read(4)
                return True
                
            except (PermissionError, FileNotFoundError, ValueError) as e:
                if attempt == max_attempts - 1:
                    raise Exception(f"File access failed after {max_attempts} attempts: {str(e)}")
                time.sleep(delay)
                
        return False

    def run(self, convert_all_to_csv = True):
        try:
            start_time = time.time()
            output_csv_path = None

            with contextlib.suppress(Exception):
                
                if not self.file_path.lower().endswith(('.xlsx', '.xlsm', '.xlsb', '.xls', 'csv')):
                    raise ValueError(f"Unable to read '{self.file_path.lower().split('.')[1]}' files.")

                if self.file_path.lower().endswith(('.xlsx', '.xlsm', '.xlsb', '.xls')) and (os.path.getsize(self.file_path)>= 1000 * (2**10 * 2**10)) and convert_all_to_csv:
                    # and (os.path.getsize(self.file_path)>= 2 * (2**10 * 2**10))
                    print(f"Converting Excel file '{os.path.basename(self.file_path)}' to CSV...", file = sys.__stdout__)

                    output_csv_path = Path(tempfile.gettempdir()).joinpath(
                        f"{Path(self.file_path).stem.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
                    
                    # Configure xlsx2csv with optimal buffer settings and sheet selection
                    from xlsx2csv import Xlsx2csv
                    converter = Xlsx2csv(
                        self.file_path,
                        outputencoding="utf-8",
                        delimiter=',',
                    )
                    
                    # Direct conversion to CSV using memory-efficient streaming
                    converter.convert(str(output_csv_path), sheetname=self.sheet_name)
                    
                    print(f"Excel file '{os.path.basename(self.file_path)}' converted to CSV: {output_csv_path}", file = sys.__stdout__)

                    self.file_path = output_csv_path
                    
                    if output_csv_path.exists() and output_csv_path.stat().st_size > 0 and self.wait_for_file_access(output_csv_path):
                        df, file_info = self.read_csv(output_csv_path)
                        file_info['filename'] = self._file_path_ if isinstance(self._file_path_,str) else self._file_path_._str
                        
                else:
                    if self.file_path.lower().endswith('.csv'):
                        df, file_info = self.read_csv(self.file_path)
                        file_info['filename'] = self.file_path
                    else:
                        df, file_info = self.read_excel(self.file_path, self.sheet_name, self.data_only, self.read_only, self.usecols, None, self.dtype, self.nrows)
                    
                file_info['load_time'] = time.time() - start_time
                # return

        except Exception as e:
            raise e

        finally:
            # Clean up output CSV if it exists
            if output_csv_path and os.path.exists(output_csv_path):
                try:
                    os.unlink(output_csv_path)
                except Exception as e:
                    print(f"Warning: Could not delete output CSV file: {e}")

            if 'file_info' in locals() and self._file_path_.lower().endswith(('.xlsx', '.xlsm', '.xlsb', '.xls')):
                file_info['sheet_name'] = self.sheet_name
                file_info['total_sheets'] = ''  
                file_info['size'] = os.path.getsize(self._file_path_) / (1024 * 1024)

        return df, file_info
        
    def read_excel(
        self, 
        filename, 
        sheet_name=0, 
        data_only=True, 
        read_only=True, 
        usecols=None, 
        header_row=None, 
        dtype=str, 
        nrows=None
    ):
        """
        Read Excel file with flexible parsing options using calamine.
        
        Args:
            filename (str): Path to the Excel file
            sheet_name (int/str, optional): Sheet to read. Defaults to first sheet.
            data_only (bool, optional): Read cell values, not formulas. Defaults to True.
            read_only (bool, optional): Memory-efficient reading. Defaults to True.
            usecols (list/None, optional): Columns to read. Defaults to None.
            header_row (int/None, optional): Row to use as column names. Defaults to None.
            dtype (type, optional): Data type for columns. Defaults to str.
            nrows (int/None, optional): Number of rows to read. Defaults to None.
        
        Returns:
            pandas.DataFrame: Parsed Excel data
        """
        
        # Read workbook with specified parameters
        df = pd.read_excel(
            filename, 
            engine='calamine',
            sheet_name=sheet_name,
            header=header_row,
            usecols=usecols,
            nrows=nrows,
            dtype=dtype,
            keep_default_na=False
        )

        file_info = {
            'filename': filename,
            'sheet_name': sheet_name,
            'total_sheets': len(get_sheet_names(filename)),
            'rows': df.shape[0],
            'columns': df.shape[1],
            'size': os.path.getsize(filename) / (1024 * 1024)
        }
        

        
        return df, file_info
    
    def read_pyarrow_contr_w_delim_fix(self, file_path, encoding, delimiter, quotechar, keep_footer=True):
        
        with open(file_path, 'r', encoding=encoding) as f:
            sample_rows = []
            for i, line in enumerate(f):
                if i < 100:
                    sample_rows.append(line)
                else:
                    break

        columns_names = sample_rows[0].strip().split(delimiter)

        def ana_f_struct(sample_rows): # Analyze file structure

            delimiter_counts = [line.strip().count(delimiter) for line in sample_rows]
            expected_no_delims = max(set(delimiter_counts))

            # Check for inconsistent delimitation
            inconsistent = (
                any(line.startswith(delimiter) for line in sample_rows) and not all(line.startswith(delimiter) for line in sample_rows)
            ) or (
                any(line.endswith(delimiter) for line in sample_rows) and not all(line.endswith(delimiter) for line in sample_rows)
            ) or (
                len(set(delimiter_counts)) > 1
            )

            # Calculate median and mean only if there are non-zero counts
            non_zero_counts = [count for count in delimiter_counts if count > 0]
            if non_zero_counts:
                median_count = np.median(non_zero_counts)
                mean_count = np.mean(non_zero_counts)
                if median_count > 0 and mean_count > 0:
                    inconsistent = inconsistent or (abs(median_count - mean_count) / mean_count > 0.1)
            
            return inconsistent, expected_no_delims

        inconsistent, expected_no_delims = ana_f_struct(sample_rows)
        

        def read_with_arrow(file_obj):
            read_options = pyspcsv.ReadOptions(
                use_threads=True,
                encoding=encoding,
                autogenerate_column_names=False,
                # column_names=columns_names,
            )

            def skip_footer(row):
                if row.actual_columns < row.expected_columns:
                    # Calculate how many commas need to be added
                    commas_to_add = row.expected_columns - row.actual_columns
                    modified_row = row.text + ',' * commas_to_add
                    print(f"\nPYARROW READING CSV - MODIFYING ROW: \nOriginal: '{row.text}'\nModified: '{modified_row}'\n")
                    row.text = modified_row
                    
                    return 'modify'
                elif (row.actual_columns < 4) and (all(c.isdigit() or c in '.;' for c in row.text)):
                    print(f"\nPYARROW READING CSV - SKIPPING ROW: '{row.text}'\n")
                    return 'skip'
                else:
                    return 'error'

            parse_options = pyspcsv.ParseOptions(
                delimiter=delimiter,
                quote_char=quotechar,
                double_quote=True,
                escape_char=False,
                newlines_in_values=True,
                ignore_empty_lines=False,
                invalid_row_handler= skip_footer
            )

            convert_options = pyspcsv.ConvertOptions(
                check_utf8=False,
                # strings_can_be_null=False,
                # include_columns=None,
                include_missing_columns=True,
                auto_dict_encode=False,
                timestamp_parsers=[],
                # column_types={"f"+str(col): pa.string() for col in range(0, expected_no_delims + 1)}  # Set all columns to string type
                column_types={col: pa.string() for col,_ in zip(columns_names, list(range(expected_no_delims + 1)))}
            )

            table = pyspcsv.read_csv(
                file_obj,
                read_options=read_options,
                parse_options=parse_options,
                convert_options=convert_options
            )

            return table
        
        try:
            table = read_with_arrow(file_path)

        except Exception:

            if len(columns_names) <= expected_no_delims:
                columns_names.extend([''] * (expected_no_delims - len(columns_names) + 1))
            
            # if expected_no_delims < len(columns_names) - 1:
            #     expected_no_delims = len(columns_names) - 1
            
            
            def read_with_mmap():
                with open(file_path, 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        return mm.read().decode(encoding).splitlines()

            # Method 2: Using buffered reading            
            
            def read_with_buffer(chunk_size=1024*1024):  # 1MB buffer
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read().splitlines()

            # Method 3: Parallel reading for very large files            
            
            def read_chunk(file, start  , size):
                file.seek(start)
                chunk = file.read(size)
                return chunk.decode(encoding).splitlines()
            
            def parallel_read(num_chunks=os.cpu_count()-1):
                file_size = os.path.getsize(file_path)
                chunk_size = file_size // num_chunks
                
                with open(file_path, 'rb') as f:
                    with ThreadPoolExecutor(max_workers=num_chunks) as executor:
                        futures = []
                        for i in range(num_chunks):
                            start = i * chunk_size
                            size = chunk_size if i < num_chunks - 1 else file_size - start
                            futures.append(executor.submit(read_chunk, f, start, size))
                        
                        results = []
                        for future in futures:
                            results.extend(future.result())
                
                return results
                    
            if inconsistent:

                file_size = os.path.getsize(file_path)
                if file_size < 10 * 1024 * 1024:  # Less than 10MB
                    file_content = read_with_buffer()
                elif file_size < 100 * 1024 * 1024:  # Less than 500MB
                    file_content = read_with_mmap()
                else:  # Large files
                    file_content = parallel_read()

                file_content = self.adjust_inconsistent_delimitation(file_content, delimiter, expected_no_delims)

                #>>> set([line.strip().count(delimiter) for line in adjusted_content])
                #>>> {8}
                #>>> set([line.strip().count(delimiter) for line in file_content])
                #>>> {8, 6, 7}

                file_obj = io.BytesIO( '\n' .join(file_content).encode(encoding))

                del file_content

            else:
                file_obj = file_path

            table = read_with_arrow(file_obj)

        # Convert to pandas DataFrame
        def to_pandas(table):
            return  table.to_pandas(self_destruct=True)
        
        df = to_pandas(table)
        
        # Remove rows where all values are empty/null
        mask = ~df.isna().all(axis=1) & ~(df == '').all(axis=1)
        df = df[mask]

        return df
    
    def adjust_inconsistent_delimitation(self, file_content, delimiter, expected_no_delims):
        def process_chunk(chunk):
            return [self.adjust_line(line, delimiter, expected_no_delims) for line in chunk]

        # Split the file content into chunks for parallel processing
        chunk_size = max(1, len(file_content) // (4 * 10))  # Adjust based on your needs
        chunks = [file_content[i:i + chunk_size] for i in range(0, len(file_content) + 1, chunk_size)]

        # {k:i for k,i in zip(chunks[0][0].split(','),chunks[0][1].split(','))}
        # set([l.strip().count(delimiter) for l in adjusted_content])

        adjusted_content = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for processed_chunk in executor.map(process_chunk, chunks):
                adjusted_content.extend(processed_chunk)

        return adjusted_content

    def adjust_line(self, line, delimiter, expected_no_delims):
        line = line.strip()

        current_no_delims = line.count(delimiter)
        if current_no_delims < expected_no_delims:
            line += delimiter * (expected_no_delims - current_no_delims)
        elif current_no_delims > expected_no_delims:
            line = delimiter.join(line.split(delimiter)[:expected_no_delims])
            # pass

        return line + '\n' 
    
    def detect_encoding_and_dialect(self, sampling_size_bytes=2**20):
        encoding = None
        with open(self.file_path, 'rb') as file:
            raw = file.read(128)
            for enc, boms in [('utf-8-sig', (codecs.BOM_UTF8,)),
                                ('utf-16', (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)),
                                ('utf-32', (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE))]:
                if any(raw.startswith(bom) for bom in boms):
                    encoding = enc
                    break

        if encoding is None:
            st = time.time()
            with open(self.file_path, 'rb') as f:
                result = chardet.detect(f.read(sampling_size_bytes))
            encoding = 'ISO-8859-1' if result['encoding'] == 'ascii' else result['encoding']
            # print(f"Detected encoding: {encoding} in {time.time() - st:.2f} seconds")

        with open(self.file_path, 'r', encoding=encoding) as f:
            sample = f.read(sampling_size_bytes)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t', '|'])
                if dialect.delimiter not in sample[:100]:
                    char_count = {char: sample.count(char) for char in [',', ';', '\t', '|']}
                    dialect.delimiter = max(char_count.items(), key=lambda x: x[1])[0]

            except Exception:
                dialect = csv.excel
                dialect.delimiter = ','
                dialect.quotechar = '"'
                dialect.escapechar = None
                dialect.doublequote = True
                dialect.skipinitialspace = True
                dialect.quoting = csv.QUOTE_MINIMAL
        
        if not hasattr(dialect, 'delimiter'):
            dialect = csv.excel
            dialect.delimiter = ','
            dialect.quotechar = '"'
            dialect.escapechar = None
            dialect.doublequote = True
            dialect.skipinitialspace = True
            dialect.quoting = csv.QUOTE_MINIMAL

        return encoding, dialect
    
    def read_csv_with_column_mismatch(self, file_path, encoding, dialect):
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        max_columns = max(len(line.split(dialect.delimiter)) for line in lines)
        padded_lines = [line.strip().split(dialect.delimiter) + [''] * (max_columns - len(line.split(dialect.delimiter))) for line in lines]

        df = pd.DataFrame(padded_lines)

        if df.iloc[0].notna().all():
            df.columns = df.iloc[0]
            df = df.iloc[1:]

        return df

    def read_csv(self, file_path):
        
        def read_pandas():
            df = pd.read_csv(file_path, dtype=str, dialect=dialect, encoding = encoding, header = None, low_memory = False)
            return df

        encoding, dialect = self.detect_encoding_and_dialect()

        QApplication.processEvents()

        try:
            encoding_list = ['ISO-8859-1', 'utf-8', 'utf-16', 'utf-32', 'utf-8-sig', 'utf-16-le', 'utf-16-be', 'utf-32-le', 'utf-32-be']
            if len(set(encoding_list).intersection([encoding]))>0:
                encoding_list.remove(set(encoding_list).intersection([encoding]).pop())
            encoding_list.sort(reverse = False)
            df = None
            for enc in [encoding] + encoding_list:
                try:
                    df = self.read_pyarrow_contr_w_delim_fix(file_path, enc, dialect.delimiter, dialect.quotechar, keep_footer = True)
                    # df = read_pandas()
                    encoding = enc
                    break
                except UnicodeError as e:
                    
                    print("\n" + enc + "\n" + e.args[0] + "\n")
                    continue

                except pa.lib.ArrowInvalid as e:
                    print(f"\nError with encoding {enc}:\n{str(e)}\n")
                    if "Expected" in str(e) and "columns, got" in str(e):
                        # If it's a column mismatch error, try to handle it
                        try:
                            df = self.read_csv_with_column_mismatch(file_path, enc, dialect)
                            encoding = enc
                            break
                        except Exception as inner_e:
                            print(f"Failed to handle column mismatch: {str(inner_e)}")
                            print("\n" + enc + "\n" + e.args[0] + "\n")
                            raise
                    continue

                except:
                    continue
                
            if df is None:
                df = read_pandas()

        except Exception as e:
            df = read_pandas()

        file_info = {
            'delimiter': dialect.delimiter,
            'encoding': encoding,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'size': os.path.getsize(file_path) / (1024 * 1024),
            'dialect' : {
        'delimiter': dialect.delimiter,
        'doublequote': dialect.doublequote,
        'escapechar': dialect.escapechar,
        'lineterminator': dialect.lineterminator if hasattr(dialect, 'lineterminator') else '\r\n',
        'quotechar': dialect.quotechar,
        'quoting': dialect.quoting,
        'skipinitialspace': dialect.skipinitialspace,
        'strict': dialect.strict if hasattr(dialect, 'strict') else False
    }
        }

        return df, file_info


class FilterWorker(QThread):
    progress_updated = Signal(int, str)
    filter_completed = Signal(pd.DataFrame)

    def __init__(self, df, filters_and_logic):
        super().__init__()
        self.df = df
        self.filters_and_logic = filters_and_logic

    def run(self):
        total_filters = len([f for f in self.filters_and_logic if isinstance(f, tuple)])
        mask_list = []
        logic_list = []
        count = 0

        for item in self.filters_and_logic:
            if isinstance(item, tuple):
                # Unpack the new values
                column, comparison, filter_string, filter_string2, case_sensitive, field_type, use_regex, negate_filter = item
                count += 1
                self.progress_updated.emit(int((count / total_filters) * 100), f"Applying filter {count}/{total_filters}...")
                
                current_mask = pd.Series(True, index=self.df.index) # Default to True
                if field_type == 'Date':
                    current_mask = self.apply_date_filter(column, comparison, filter_string, filter_string2, negate_filter)
                elif field_type == 'Numeric':
                    current_mask = self.apply_numeric_filter(column, comparison, filter_string, filter_string2, negate_filter)
                else:  # String
                    current_mask = self.apply_string_filter(column, comparison, filter_string, case_sensitive, use_regex, negate_filter)
                
                mask_list.append(current_mask)
            else: # This is a logic string ('and' or 'or')
                logic_list.append(item)

        if not mask_list: # No filters applied
            filtered_df = self.df.copy() # Return a copy of the original DataFrame
        else:
            # Combine masks
            final_mask = mask_list[0]
            for i in range(1, len(mask_list)):
                logic = logic_list[i-1] # Get the logic operator between current and previous mask
                if logic.lower() == "or":
                    final_mask = final_mask | mask_list[i]
                else: # Default to 'and'
                    final_mask = final_mask & mask_list[i]
            
            filtered_df = self.df[final_mask]

        self.progress_updated.emit(100, "Filtering completed.")
        self.filter_completed.emit(filtered_df)

    def apply_date_filter(self, column, comparison, filter_string, filter_string2, negate_filter):
        mask = pd.Series(False, index=self.df.index) # Default to False
        try:
            date_column = pd.to_datetime(self.df[column], errors='coerce')
            if comparison == 'Empty':
                mask = date_column.isna()
            elif pd.isna(pd.to_datetime(filter_string, errors='coerce')): # If primary filter date is invalid for other ops
                return mask if not negate_filter else ~mask
            else:
                filter_date = pd.to_datetime(filter_string, errors='coerce')
                if comparison in ('Equals', 'Contains'): # 'Contains' for date can mean equals date part
                    mask = date_column.dt.normalize() == filter_date.normalize()
                elif comparison == '>':
                    mask = date_column > filter_date
                elif comparison == '<':
                    mask = date_column < filter_date
                elif comparison == '>=':
                    mask = date_column >= filter_date
                elif comparison == '<=':
                    mask = date_column <= filter_date
                elif comparison == 'Between':
                    filter_date2 = pd.to_datetime(filter_string2, errors='coerce')
                    if pd.notna(filter_date2):
                        mask = (date_column >= filter_date) & (date_column <= filter_date2)
                    # If filter_date2 is NaT, 'Between' effectively becomes '>=' filter_date or similar
                    # Depending on desired behavior. For now, if filter_date2 is NaT, it might not match anything as expected for 'Between'.
                    # Consider making 'Between' require both dates to be valid.
        except Exception as e:
            print(f"Date filter error: {e}")
            # mask remains all False
        
        return ~mask if negate_filter else mask

    def apply_numeric_filter(self, column, comparison, filter_string, filter_string2, negate_filter):
        mask = pd.Series(False, index=self.df.index)
        try:
            numeric_column = pd.to_numeric(self.df[column], errors='coerce')
            if comparison == 'Empty':
                mask = numeric_column.isna()
            elif pd.isna(pd.to_numeric(filter_string, errors='coerce')): # If primary filter value is invalid for other ops
                return mask if not negate_filter else ~mask
            else:
                val1 = pd.to_numeric(filter_string, errors='coerce')
                if comparison == '>':
                    mask = numeric_column > val1
                elif comparison == '<':
                    mask = numeric_column < val1
                elif comparison == '>=':
                    mask = numeric_column >= val1
                elif comparison == '<=':
                    mask = numeric_column <= val1
                elif comparison == 'Between':
                    val2 = pd.to_numeric(filter_string2, errors='coerce')
                    if pd.notna(val2):
                        mask = (numeric_column >= val1) & (numeric_column <= val2)
                elif comparison == 'Equals': # Handles 'Contains' as 'Equals' for numeric
                    mask = numeric_column == val1
            
            # Handle NaNs in the column if val1 is not NaN: numeric_column > NaN is always False.
            # If you want to exclude NaNs from matching, you can add: mask = mask & numeric_column.notna()
            # Or if val1 is NaN, and comparison is 'Equals', then: mask = numeric_column.isna()

        except Exception as e:
            print(f"Numeric filter error: {e}")
        
        return ~mask if negate_filter else mask

    def apply_string_filter(self, column, comparison, filter_string, case_sensitive, use_regex, negate_filter):
        mask = pd.Series(False, index=self.df.index)
        original_col = self.df[column] # For checking actual NaNs
        col_astype_str = original_col.astype(str)

        try:
            if comparison == 'Empty':
                # Checks for actual NaN in original data or common string representations of empty/NaN
                mask = original_col.isna() | (col_astype_str == '') | (col_astype_str.str.lower() == 'nan') | (col_astype_str.str.lower() == 'none') | (col_astype_str.str.lower() == 'null')

            elif use_regex:
                # Regex search, case sensitivity is part of the pattern or re.IGNORECASE flag if used with re.compile
                # pandas str.contains with regex=True respects case by default.
                # For case-insensitive regex, pattern should include (?i) or use re.IGNORECASE with re.compile
                # We assume user provides pattern correctly for case sensitivity.
                # For `Equals` with regex, it would be `^pattern$`
                if comparison == 'Equals':
                    # Ensure the regex matches the entire string for 'Equals'
                    effective_filter_string = f"^{filter_string}$"
                    mask = col_astype_str.str.contains(effective_filter_string, regex=True, na=False)
                elif comparison == 'Contains':
                    mask = col_astype_str.str.contains(filter_string, regex=True, na=False)
                # Other comparisons for regex might not be standard (e.g., >, <)
                # If needed, they would require custom logic or clear definition.
            else:
                # Standard string operations
                if comparison == 'Contains':
                    mask = col_astype_str.str.contains(filter_string, case=case_sensitive, na=False, regex=False)
                elif comparison == 'Equals':
                    if case_sensitive:
                        mask = (col_astype_str == filter_string)
                    else:
                        mask = (col_astype_str.str.lower() == filter_string.lower())
                # Handling for other comparisons like '>', '<' for strings (lexicographical)
                elif comparison == '>':
                    mask = col_astype_str > filter_string # Case sensitive by default
                elif comparison == '<':
                    mask = col_astype_str < filter_string # Case sensitive by default
                elif comparison == '>=':
                    mask = col_astype_str >= filter_string # Case sensitive by default
                elif comparison == '<=':
                    mask = col_astype_str <= filter_string # Case sensitive by default
        
        except re.error as e: # Catch regex specific errors
            print(f"Regex error: {e} in pattern '{filter_string}'")
            # mask remains all False
        except Exception as e:
            print(f"String filter error: {e}")
            # mask remains all False
        
        return ~mask if negate_filter else mask
    
class FileSaverWorker(QThread):
    progress_updated = Signal(int, str)
    save_completed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, df, file_name,progress_callback, completion_callback, error_callback, file_info_specs):
        super().__init__()
        self.df = df
        self.file_name = file_name

        self.progress_updated.connect(progress_callback)
        self.save_completed.connect(completion_callback)
        self.error_occurred.connect(error_callback)
        self.file_info_specs = file_info_specs

    def run(self):
        try:
            self.progress_updated.emit(25, f"Saving file '{self.file_name}'...")
            if self.file_name.lower().endswith('.csv'):
                dialect = None if not 'dialect' in self.file_info_specs else self.file_info_specs['dialect']
                encoding_spec = 'utf-8' if not 'encoding' in self.file_info_specs else self.file_info_specs['encoding']
                if dialect:
                    self.df.to_csv(self.file_name, index=False, sep = dialect['delimiter'], encoding = encoding_spec)
                else:
                    self.df.to_csv(self.file_name, index=False, encoding = encoding_spec)
            elif self.file_name.lower().endswith('.xlsx'):
                self.df.to_excel(self.file_name, index=False)
            self.progress_updated.emit(100, "File saved successfully.")
            self.save_completed.emit(self.file_name)
        except Exception as e:
            self.error_occurred.emit(str(e))

class FieldTypeSwitch(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        self.combo = QComboBox()
        self.combo.addItems(['String', 'Date', 'Numeric'])
        layout.addWidget(self.combo)

    def set_type(self, field_type):
        self.combo.setCurrentText(field_type)

    def get_type(self):
        return self.combo.currentText()

class FilterWidget(QFrame):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        fixed_width = 110 # Adjusted width slightly for new checkboxes
        
        self.column_combo = QComboBox()
        self.column_combo.addItems(columns)
        self.column_combo.setFixedWidth(fixed_width + 20) # Give column combo a bit more space
        
        self.field_type_switch = FieldTypeSwitch()
        self.field_type_switch.setFixedWidth(fixed_width)
        
        self.comparison_combo = QComboBox()
        # Removed 'Regex' from here as it's now a checkbox
        self.comparison_combo.addItems(['Contains', 'Equals', '>', '<', '>=', '<=', 'Between', 'Empty'])
        self.comparison_combo.setFixedWidth(fixed_width)
        
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Enter filter text...")
        self.filter_input2 = QLineEdit()
        self.filter_input2.setPlaceholderText("Upper bound for 'Between'")
        self.filter_input2.setVisible(False)
        
        self.case_sensitive_combo = QComboBox()
        self.case_sensitive_combo.addItems(['Case Insensitive', 'Case Sensitive'])
        self.case_sensitive_combo.setFixedWidth(fixed_width + 20)

        # New Checkboxes
        self.use_regex_checkbox = QCheckBox("Regex")
        self.negate_filter_checkbox = QCheckBox("Not") # "Not" is shorter for "Negate"

        self.remove_button = QPushButton("X")
        self.remove_button.setFixedSize(30, 30) # Made remove button smaller
        
        current_year = datetime.now().year
        default_date = QDate(current_year, 3, 31)
        self.date_input = QDateEdit()
        self.date_input.setCalendarPopup(True)
        self.date_input.setDisplayFormat("yyyy-MM-dd")
        self.date_input.setDate(default_date)
        self.date_input.setVisible(False)
        self.date_input2 = QDateEdit()
        self.date_input2.setCalendarPopup(True)
        self.date_input2.setDisplayFormat("yyyy-MM-dd")
        self.date_input2.setDate(default_date)
        self.date_input2.setVisible(False)
        
        self.layout.addWidget(self.column_combo)
        self.layout.addWidget(self.field_type_switch)
        self.layout.addWidget(self.comparison_combo)
        self.layout.addWidget(self.filter_input)
        self.layout.addWidget(self.filter_input2)
        self.layout.addWidget(self.date_input)
        self.layout.addWidget(self.date_input2)
        self.layout.addWidget(self.case_sensitive_combo)
        self.layout.addWidget(self.use_regex_checkbox) # Add checkbox
        self.layout.addWidget(self.negate_filter_checkbox) # Add checkbox
        self.layout.addWidget(self.remove_button)
        
        self.filter_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.date_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.date_input2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        self.comparison_combo.currentTextChanged.connect(self.on_ui_changed)
        self.column_combo.currentTextChanged.connect(self.on_column_changed)
        self.field_type_switch.combo.currentTextChanged.connect(self.on_ui_changed) # Connect to on_ui_changed
        self.use_regex_checkbox.stateChanged.connect(self.on_ui_changed) # Connect to on_ui_changed

        self.on_column_changed(self.column_combo.currentText()) # Initial setup
        self.on_ui_changed() # Call once to set initial UI state

    def on_comparison_changed(self, text):
        is_between = text == 'Between'
        field_type = self.field_type_switch.get_type()
        
        self.filter_input2.setVisible(is_between and field_type != 'Date')
        self.date_input2.setVisible(is_between and field_type == 'Date')
        self.case_sensitive_combo.setVisible(text in ['Contains', 'Equals'] and field_type == 'String')

    def on_ui_changed(self): # Renamed and consolidated UI update logic
        field_type = self.field_type_switch.get_type()
        comparison = self.comparison_combo.currentText()
        is_regex_checked = self.use_regex_checkbox.isChecked()

        is_string_type = field_type == 'String'
        is_date_type = field_type == 'Date'
        is_numeric_type = field_type == 'Numeric'
        is_empty_comparison = comparison == 'Empty' # New condition

        # Visibility of date/text inputs
        self.date_input.setVisible(is_date_type)
        self.filter_input.setVisible(not is_date_type)
        
        is_between = comparison == 'Between'
        self.date_input2.setVisible(is_date_type and is_between)
        self.filter_input2.setVisible(not is_date_type and is_between)

        # Regex checkbox visibility and placeholder text
        self.use_regex_checkbox.setVisible(is_string_type and comparison in ['Contains', 'Equals'])
        if is_string_type and comparison in ['Contains', 'Equals']:
            self.filter_input.setPlaceholderText("Enter text (or regex if checked)" if self.use_regex_checkbox.isVisible() else "Enter filter text...")
        elif is_numeric_type :
             self.filter_input.setPlaceholderText("Enter numeric value...")
        else:
            self.filter_input.setPlaceholderText("Enter filter text...")


        # Case sensitive combo visibility
        # Hide if regex is checked (regex handles its own case sensitivity)
        # or if not a relevant string comparison
        can_show_case_sensitive = is_string_type and comparison in ['Contains', 'Equals'] and not is_regex_checked
        self.case_sensitive_combo.setVisible(can_show_case_sensitive)

        # Adjust comparison options based on field type
        if is_date_type or is_numeric_type:
            allowed_comparisons = ['Equals', 'Empty', '>', '<', '>=', '<=', 'Between'] # Add 'Empty'
            if comparison not in allowed_comparisons:
                self.comparison_combo.setCurrentText('Equals') # Default for these types
            # Disable 'Contains' for non-string types (or handle appropriately)
            for i in range(self.comparison_combo.count()):
                item_text = self.comparison_combo.itemText(i)
                item_widget = self.comparison_combo.itemData(i)
                if item_widget: # Ensure item_widget is not None
                    item_widget.setEnabled(item_text in allowed_comparisons)
        else: # String type
            # Enable all comparisons for string type (pandas handles most of these on strings)
            for i in range(self.comparison_combo.count()):
                item_widget = self.comparison_combo.itemData(i)
                if item_widget:
                     item_widget.setEnabled(True)
        
        # If 'Empty' is selected, ensure other comparison-specific UIs are hidden
        if is_empty_comparison:
            self.filter_input.setVisible(False)
            self.filter_input2.setVisible(False)
            self.date_input.setVisible(False)
            self.date_input2.setVisible(False)
            self.case_sensitive_combo.setVisible(False)
            self.use_regex_checkbox.setVisible(False)


    def on_column_changed(self, column):
        # Auto-detect field type based on column name (simple heuristic)
        is_date = 'date' in column.lower() or 'dt' in column.lower()
        # Add more sophisticated type detection if needed, e.g., by sampling data
        
        field_type = 'Date' if is_date else 'String' # Default to String if not clearly date
        self.field_type_switch.set_type(field_type)
        self.on_ui_changed() # Update UI based on new column and auto-detected type

    def get_filter_values(self):
        column = self.column_combo.currentText()
        comparison = self.comparison_combo.currentText()
        field_type = self.field_type_switch.get_type()

        filter_string = ""
        filter_string2 = ""

        if comparison != 'Empty': # Only get values if not 'Empty'
            if field_type == 'Date':
                filter_string = self.date_input.date().toString(Qt.ISODate)
                filter_string2 = self.date_input2.date().toString(Qt.ISODate) if self.date_input2.isVisible() else ""
            else:
                filter_string = self.filter_input.text()
                filter_string2 = self.filter_input2.text() if self.filter_input2.isVisible() else ""
        
        case_sensitive = self.case_sensitive_combo.currentText() == 'Case Sensitive' if self.case_sensitive_combo.isVisible() else False
        use_regex = self.use_regex_checkbox.isChecked() if self.use_regex_checkbox.isVisible() else False
        negate_filter = self.negate_filter_checkbox.isChecked()

        # When regex is used, case_sensitive dropdown is ignored, regex pattern handles case.
        if use_regex:
            case_sensitive = False # Or rather, it's up to the regex pattern

        return column, comparison, filter_string, filter_string2, case_sensitive, field_type, use_regex, negate_filter

class LogicWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.addStretch()
        self.and_radio = QRadioButton("AND")
        self.or_radio = QRadioButton("OR")
        self.and_radio.setChecked(True)
        self.layout.addWidget(self.and_radio)
        self.layout.addWidget(self.or_radio)
        self.layout.addStretch()

class FilterPanel(QWidget):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.filters = []
        self.logic_widgets = []
        self.columns = columns

        self.scroll_area = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_area.setWidget(self.scroll_widget)
        self.scroll_area.setWidgetResizable(True)
        self.layout.addWidget(self.scroll_area)

        self.button_layout = QHBoxLayout()
        self.add_filter_button = QPushButton("Add Filter")
        self.add_filter_button.clicked.connect(self.add_filter)
        self.apply_button = QPushButton("Apply Filters")
        self.button_layout.addWidget(self.add_filter_button)
        self.button_layout.addWidget(self.apply_button)
        self.layout.addLayout(self.button_layout)

        self.add_filter()

    def add_filter(self):
        filter_widget = FilterWidget(self.columns)
        filter_widget.remove_button.clicked.connect(lambda: self.remove_filter(filter_widget))
        self.filters.append(filter_widget)
        self.scroll_layout.addWidget(filter_widget)

        if len(self.filters) > 1:
            logic_widget = LogicWidget()
            self.logic_widgets.append(logic_widget)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, logic_widget)

    def remove_filter(self, filter_widget):
        index = self.filters.index(filter_widget)
        self.filters.remove(filter_widget)
        filter_widget.deleteLater()

        if index < len(self.logic_widgets):
            logic_widget = self.logic_widgets.pop(index)
            logic_widget.deleteLater()
        elif index > 0 and self.logic_widgets:
            logic_widget = self.logic_widgets.pop(index - 1)
            logic_widget.deleteLater()

    def get_filters_and_logic(self):
        filters_and_logic = []
        for i, filter_widget in enumerate(self.filters):
            filters_and_logic.append(filter_widget.get_filter_values())
            if i < len(self.logic_widgets):
                filters_and_logic.append(
                    "and" if self.logic_widgets[i].and_radio.isChecked() else "or"
                )
        return filters_and_logic

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data
        self._row_count = self._data.shape[0]
        self._column_count = self._data.shape[1]
        self._column_headers = list(self._data.columns)
        self._index_headers = list(self._data.index)

    def rowCount(self, parent=None):
        return self._row_count

    def columnCount(self, parent=None):
        return self._column_count

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                value = self._data.iat[index.row(), index.column()]
                return str(value) if pd.notna(value) else ''
        return None
    
    @Slot(int, int, Qt.Orientation)
    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                try:
                    return self._column_headers[section]
                except IndexError:
                    return None
            if orientation == Qt.Vertical:
                try:
                    return str(self._index_headers[section])
                except IndexError:
                    return None
        return None

class AsyncFileLoaderThread(QThread):
    finished_signal = Signal(object, dict)  # object will be the DataFrame
    error_signal = Signal(str)
    progress_signal = Signal(int, str) # For more granular progress if needed

    def __init__(self, file_path, sheet_name=None, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.sheet_name = sheet_name
        # Store a reference to the original FileLoaderWorker logic
        # Assuming run_FileLoaderWorker is a top-level function or accessible
        # If FileLoaderWorker needs to be instantiated and run:
        self.loader_args = (file_path, sheet_name)


    def run(self):
        try:
            # self.progress_signal.emit(10, f"Starting to load {os.path.basename(self.file_path)}...")
            # Using the existing run_FileLoaderWorker function
            df, file_info = multiprocessing.Pool(1).map(run_FileLoaderWorker, [self.loader_args])[0]
            self.finished_signal.emit(df, file_info)
        except Exception as e:
            self.error_signal.emit(f"Error loading file: {str(e)}")
            
class Pickaxe(QMainWindow):

    update_model_signal = Signal(pd.DataFrame, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pickaxe")
        self.setGeometry(100, 100, 1400, 800)

        icon_path = resource_path("pickaxe.ico")
        icon = QIcon(icon_path)
        self.setWindowIcon(icon)

        # Add a member to store the column index/name for context menu actions
        self.context_menu_column_logical_index = -1
        self.context_menu_column_name = None

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.button_layout = QHBoxLayout()
        self.file_button = QPushButton("Open File")
        self.file_button.clicked.connect(self.load_file)
        self.save_button = QPushButton("Save File")
        self.save_button.clicked.connect(self.save_file)
        self.button_layout.addWidget(self.file_button)
        self.button_layout.addWidget(self.save_button)
        self.layout.addLayout(self.button_layout)

        self.info_layout = QHBoxLayout()
        self.file_info_label = QLabel()
        self.file_info_label.setWordWrap(True)
        self.column_range_label = QLabel()
        self.info_layout.addWidget(self.file_info_label)
        self.info_layout.addWidget(self.column_range_label)
        self.layout.addLayout(self.info_layout)

        self.filter_panel = None

        self.filter_nav_layout = QHBoxLayout()
        self.filter_toggle_button = QPushButton("Show/Hide Filters")
        self.filter_toggle_button.clicked.connect(self.toggle_filter_panel)
        self.filter_nav_layout.addWidget(self.filter_toggle_button, 1)

        self.columns_per_page = 50

        self.nav_button_layout = QHBoxLayout()
        self.prev_columns_button = QPushButton(f"Previous {self.columns_per_page} Columns")
        self.prev_columns_button.clicked.connect(self.show_previous_columns)
        self.next_columns_button = QPushButton(f"Next {self.columns_per_page} Columns")
        self.next_columns_button.clicked.connect(self.show_next_columns)
        self.nav_button_layout.addWidget(self.prev_columns_button)
        self.nav_button_layout.addWidget(self.next_columns_button)
        self.filter_nav_layout.addLayout(self.nav_button_layout, 1)
        
        self.layout.addLayout(self.filter_nav_layout)

        self.table_view = QTableView()
        self.table_view.setVerticalScrollMode(QTableView.ScrollPerPixel)
        self.table_view.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)
        self.layout.addWidget(self.table_view)
        
        # Add statistics label
        self.stats_label = QLabel("Select a column to see statistics.")
        self.stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter) # Align to bottom right of its cell
        self.stats_label.setMinimumHeight(40) # Give it some space
        self.stats_label.setWordWrap(True)
        font = self.stats_label.font()
        font.setPointSize(9)
        self.stats_label.setFont(font)
        self.layout.addWidget(self.stats_label) # Add to main layout

        # Connect column selection change to update statistics
        # Using currentColumnChanged from the selection model
            
        self.table_view.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.horizontalHeader().customContextMenuRequested.connect(self.show_header_context_menu)

        self.df = None
        self.model = None
        self._filepath = None
        self.applied_filters = []
        self.file_info = {}
        self.current_column_page = 0

        self.update_model_signal.connect(self.update_model)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.progress_label)
        
    def handle_column_selection_for_stats(self, current, previous):
        # 'current' is a QModelIndex for the current cell in the new column
        if not current.isValid():
            self.stats_label.setText("No column selected or data not loaded.")
            return
        
        # The current QModelIndex from currentColumnChanged refers to a cell,
        # its column() method gives the column index in the current view model.
        view_column_index = current.column()
        self.update_column_statistics_by_index(view_column_index)


    def update_column_statistics_by_index(self, view_column_index):
        if self.df is None or self.model is None : # Check if data is loaded
            self.stats_label.setText("No data loaded.")
            return

        # Determine which DataFrame to use (filtered or original)
        active_df = self.filtered_df if self.filtered_df is not None and not self.filtered_df.empty else self.df
        if active_df is None or active_df.empty:
             self.stats_label.setText("No data to analyze.")
             return

        # Map view_column_index (from the current display page) to the actual column index in active_df
        # This relies on self.current_column_page and self.columns_per_page
        # and assumes the model slice corresponds directly to these.
        actual_df_column_index = self.current_column_page * self.columns_per_page + view_column_index
        
        if not (0 <= actual_df_column_index < len(active_df.columns)):
            self.stats_label.setText("Invalid column index for statistics.")
            return

        column_name = active_df.columns[actual_df_column_index]
        series = active_df[column_name]

        if series.empty:
            self.stats_label.setText(f"Column '{column_name}' is empty.")
            return

        stats_text = f"Statistics for column: '{column_name}' (Type: {series.dtype})\n"
        not_na_count = series.count() # Counts non-NA values
        na_count = series.isna().sum()
        unique_count = series.nunique()

        stats_text += f"#Non-Empty: {not_na_count:,} | #Empty: {na_count:,} ({na_count /series.shape[0] * 100:.1f}%) | #Unique: {unique_count:,}"
        
        if unique_count <= 10: # If unique values are low, show them
            unique_values = series.unique()
            if len(''.join(map(str, unique_values))) > 100: # If too long, truncate
                unique_values = [str(x)[:10] + '...' for x in unique_values]
            unique_values = [f"'{x}'" + f" ({series.value_counts()[x]:,})" for x in unique_values]
            stats_text += f"\nUnique Values:     {' ; '.join(unique_values)} \n\n"
        else:
            stats_text += "\n\n"

        if pd.api.types.is_numeric_dtype(series.dtype) and not_na_count > 0:
            # Ensure there are numeric values to calculate these stats
            numeric_series_for_stats = series.dropna() # Calculate stats only on non-NA numeric values
            if not numeric_series_for_stats.empty:
                stats_text += f"Sum: {numeric_series_for_stats.sum():,.2f} | Mean: {numeric_series_for_stats.mean():,.2f} | Median: {numeric_series_for_stats.median():,.2f} | "
                stats_text += f"Min: {numeric_series_for_stats.min():,.2f} | Max: {numeric_series_for_stats.max():,.2f} | StD: {numeric_series_for_stats.std():,.2f} | Var: {numeric_series_for_stats.var():,.2f}"
            else:
                stats_text += " (No numeric data to calculate further stats)"
        elif pd.api.types.is_datetime64_any_dtype(series.dtype) and not_na_count > 0:
            datetime_series_for_stats = series.dropna()
            if not datetime_series_for_stats.empty:
                try:
                    stats_text += f"Earliest: {datetime_series_for_stats.min()} | Latest: {datetime_series_for_stats.max()}"
                except TypeError: # Handle cases like object dtype with mixed types that couldn't be fully coerced
                    stats_text += " (Min/Max date calculation error)"
            else:
                stats_text += " (No date/time data to calculate further stats)"


        self.stats_label.setText(stats_text)

    def show_header_context_menu(self, position):
        header = self.table_view.horizontalHeader()
        self.context_menu_column_logical_index = header.logicalIndexAt(position)
        
        # Try to get the column name from the model if available
        if self.model and self.context_menu_column_logical_index >= 0 and self.context_menu_column_logical_index < self.model.columnCount():
            self.context_menu_column_name = self.model.headerData(self.context_menu_column_logical_index, Qt.Horizontal, Qt.DisplayRole)
        else:
            # Fallback if model is not ready or index is out of bounds for current view
            # This might happen if the underlying df has more columns than currently displayed
            # We should use the full df's column list if possible
            if self.df is not None and self.context_menu_column_logical_index >= 0:
                 # map view index to overall df index if pagination is complex
                 # For now, assume logicalIndexAt directly maps if no complex slicing hides columns from model
                actual_df_column_index = self.current_column_page * self.columns_per_page + self.context_menu_column_logical_index
                if actual_df_column_index < len(self.df.columns):
                    self.context_menu_column_name = self.df.columns[actual_df_column_index]
                else:
                    self.context_menu_column_name = None # Not a valid column
            else:
                self.context_menu_column_name = None


        if self.context_menu_column_name is None: # Do not show menu if not on a valid column
            return

        menu = QMenu()
        set_as_header_action = QAction("Set Selected Row as Header", self) # Original action
        set_as_header_action.triggered.connect(self.set_row_as_header) # This uses selected row, not column
        
        convert_to_numeric_action = QAction(f"Convert '{self.context_menu_column_name}' to Numeric", self)
        convert_to_numeric_action.triggered.connect(self.convert_column_to_numeric)
        
        convert_to_datetime_action = QAction(f"Convert '{self.context_menu_column_name}' to Datetime", self)
        convert_to_datetime_action.triggered.connect(self.convert_column_to_datetime)

        # The "Set as Header" action in the original code was connected to table_view's context menu,
        # not the header's. If you want "Set as Header" on row right-click, keep that original connection.
        # If you intended it for header clicks, it might need rethinking (e.g. "Reset Header"?).
        # For now, I am adding convert actions to the header's context menu.
        # menu.addAction(set_as_header_action) # Decide if this should be here or on row context menu
        
        menu.addAction(convert_to_numeric_action)
        menu.addAction(convert_to_datetime_action)
        
        menu.exec(header.mapToGlobal(position))

    def _convert_column_type(self, conversion_function, type_name):
        if self.df is None or self.context_menu_column_name is None:
            QMessageBox.warning(self, "Conversion Error", "No data or column selected for conversion.")
            return

        column_name_to_convert = self.context_menu_column_name
        try:
            self.progress_bar.setVisible(True)
            self.update_progress(0, f"Converting column '{column_name_to_convert}' to {type_name}...")
            QApplication.processEvents()

            # Create a copy to attempt conversion on, to handle errors gracefully
            original_column = self.df[column_name_to_convert].copy()
            converted_column = conversion_function(original_column)
            
            # Check conversion success (e.g., if all became NaT/NaN for wrong types)
            # For numeric, if a large portion becomes NaN where it wasn't, it might be a bad conversion.
            # For datetime, NaT.
            # This check is basic. More sophisticated checks can be added.
            if converted_column.isna().sum() > original_column.isna().sum() + (0.1 * len(original_column)): # If more than 10% new NaNs
                if type_name == "datetime" and pd.api.types.is_datetime64_any_dtype(converted_column):
                     pass # Datetime conversion often results in NaT for unparseable, which is fine
                else:
                    reply = QMessageBox.question(self, "Conversion Warning",
                                             f"Converting '{column_name_to_convert}' to {type_name} resulted in a significant number of new empty values. Proceed?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                    if reply == QMessageBox.No:
                        self.update_progress(0, "Conversion cancelled.")
                        return
            
            self.df[column_name_to_convert] = converted_column
            if self.filtered_df is not None and column_name_to_convert in self.filtered_df.columns:
                # If filtered_df is a direct view or simple slice, this might work.
                # To be safe, if filters are active, it might be better to re-filter or update carefully.
                # For simplicity, also try to convert in filtered_df if it exists.
                 self.filtered_df[column_name_to_convert] = conversion_function(self.filtered_df[column_name_to_convert].copy())

            self.update_progress(50, "Updating view...")
            QApplication.processEvents()
            # Re-apply filters if they exist, as dtypes might affect them
            if self.applied_filters:
                self.apply_filters() # This will internally call update_model with new filtered_df
            else:
                self.update_model(self.df if self.filtered_df is None else self.filtered_df) 

            # Update statistics if a column was selected and stats are displayed
            if hasattr(self, 'stats_label') and self.table_view.selectionModel().hasSelection():
                 selected_columns_indices = self.table_view.selectionModel().selectedColumns()
                 if selected_columns_indices:
                    # Use the model index for the selected column
                    model_column_index = selected_columns_indices[0].column() 
                    # map view model_column_index to overall df index if necessary
                    # current_df_for_stats = self.filtered_df if self.filtered_df is not None else self.df
                    # actual_col_idx_in_df = self.current_column_page * self.columns_per_page + model_column_index

                    # This is simpler if update_column_statistics takes a name or a direct DataFrame column index
                    self.update_column_statistics_by_index(model_column_index)


            self.update_progress(100, f"Column '{column_name_to_convert}' converted to {type_name}.")
            # QMessageBox.information(self, "Conversion Successful", f"Column '{column_name_to_convert}' converted to {type_name}.")

        except Exception as e:
            self.update_progress(100, "Conversion failed.") # To hide progress bar
            QMessageBox.critical(self, "Conversion Error", f"Could not convert column '{column_name_to_convert}' to {type_name}:\n{e}")
        finally:
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    def convert_column_to_numeric(self):
        self._convert_column_type(lambda col: pd.to_numeric(col, errors='coerce'), "numeric")

    def convert_column_to_datetime(self):
        self._convert_column_type(lambda col: pd.to_datetime(col, errors='coerce'), "datetime")

    # Keep original show_context_menu for row-based actions like "Set as Header"
    def show_context_menu(self, position):
        menu = QMenu()
        # This action is for rows, so it should be connected to table_view's context menu
        set_as_header_action = QAction("Set Selected Row as Header", self)
        set_as_header_action.triggered.connect(self.set_row_as_header)
        menu.addAction(set_as_header_action)
        menu.exec(self.table_view.viewport().mapToGlobal(position))

    def load_file(self):
        self.initial_path = os.path.join(os.path.expanduser('~'), 'Documents')
        if not os.path.exists(self.initial_path):
            self.initial_path = os.path.join(os.path.expanduser('~'), 'Downloads')
        if not os.path.exists(self.initial_path):
            self.initial_path = os.path.expanduser('~')
            
        if hasattr(self, 'initial_path_selected') and self.initial_path_selected: # Check if not None or empty
            self.initial_path = self.initial_path_selected
            
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", 
                                                dir=self.initial_path, 
                                                filter="CSV Files (*.csv);;Excel Files (*.xlsx *.xls *.xlsb *.xlsm)")
        
        if file_name:
            self.initial_path_selected = os.path.dirname(file_name) # Store selected path
            self._filepath = file_name
            
            sheet_name = None
            if file_name.lower().endswith(('.xlsx', '.xlsm', '.xlsb', '.xls')):
                try:
                    self.sheet_names = get_sheet_names(file_name) # Use global get_sheet_names
                    if len(self.sheet_names) == 1:
                        sheet_name = self.sheet_names[0]
                    else:
                        sheet_name_selected, ok = QInputDialog.getItem(self, "Select Sheet", 
                                                                    "Choose a sheet:", self.sheet_names, 0, False)
                        if not ok:
                            return
                        sheet_name = sheet_name_selected
                except Exception as e:
                    self.on_error(f"Error reading sheet names from {os.path.basename(file_name)}: {e}")
                    return

            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.update_progress(0, f"Loading '{os.path.basename(file_name)}'...")
            QApplication.processEvents()

            # Use the new AsyncFileLoaderThread
            self.file_loader_thread = AsyncFileLoaderThread(file_name, sheet_name)
            self.file_loader_thread.finished_signal.connect(self.on_file_loaded)
            self.file_loader_thread.error_signal.connect(self.on_error)
            # If AsyncFileLoaderThread emits progress_signal, connect it too:
            # self.file_loader_thread.progress_signal.connect(self.update_progress) 
            self.file_loader_thread.start()
        else: # No file selected
            self.update_progress(0, "") # Clear progress if dialog is cancelled
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    @Slot(pd.DataFrame, dict)
    def on_file_loaded(self, df, file_info):
        
        # Make sure to handle the case where self.sheet_names might not be set if not an Excel file
        if 'sheet_name' in file_info and file_info['sheet_name'] and hasattr(self, 'sheet_names') and self.sheet_names:
            # This was for Excel, file_info should already contain total_sheets if applicable
            pass
        elif not file_info.get('sheet_name'): # For CSVs primarily
            self.sheet_names = [] # Clear or reset sheet_names for non-Excel files

        self.df = df
        self.filtered_df = df.copy() # It's good practice to work on a copy for filtering
        
        if self._filepath.lower().endswith(('.xlsx', '.xlsm', '.xlsb', '.xls')):
            if hasattr(self, 'sheet_names') and self.sheet_names: # Check if sheet_names exists
                 file_info['total_sheets'] = len(self.sheet_names)
            else: # Should not happen if sheet name logic is correct, but as a fallback
                 file_info['total_sheets'] = 1 if file_info.get('sheet_name') else 'N/A'
        
        self.file_info = file_info
        self.original_row_count = self.df.shape[0]  
        
        self.update_model(self.df, True) 
        self.update_file_info_label()

        if self.filter_panel:
            self.filter_panel.deleteLater()
            self.filter_panel = None # Ensure it's reset
        self.filter_panel = FilterPanel(self.df.columns)
        self.filter_panel.apply_button.clicked.connect(self.apply_filters)
        self.layout.insertWidget(3, self.filter_panel) # Adjust index if layout changes
        self.filter_panel.setVisible(False)

        self.update_progress(100, f"File '{os.path.basename(self._filepath)}' loaded successfully.")
        # The update_progress function already handles hiding the bar at 100% after a delay.
        # If self.file_loader_thread is stored as an instance variable, you might want to delete it or wait for it
        self.file_loader_thread.quit() # Properly terminate the thread
        self.file_loader_thread.wait() # Wait for it to finish cleanly
        del self.file_loader_thread # Optional: clean up

    def update_model(self, df, col_page_turn=False):
        if df is None:
            return
        self.current_column_page = 0
        start_col = self.current_column_page * self.columns_per_page
        end_col = min(start_col + self.columns_per_page, df.shape[1])

        # Limit rows shown for performance, but keep enough for context
        row_limit_print = min(1000, df.shape[0]) 
        subset = df.iloc[:row_limit_print, start_col:end_col]

        self.model = PandasModel(subset)
        self.table_view.setModel(self.model)
        
        if self.table_view.selectionModel(): # Ensure selection model exists
            self.table_view.selectionModel().currentColumnChanged.connect(self.handle_column_selection_for_stats)

        # Ensure Interactive mode is set (can be set once in __init__)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive) 
        self.table_view.verticalHeader().setDefaultSectionSize(20) # Keep reasonable row height

        self.update_column_range_label(df) 
        
    def show_previous_columns(self):
        if self.current_column_page > 0:
            self.current_column_page -= 1
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.update_progress(0, "Loading previous columns...")
            QApplication.processEvents()

            self.threaded_update_model()

            self.update_progress(100, "Previous columns loaded.")
            QApplication.processEvents()
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    def show_next_columns(self):
        if (self.current_column_page + 1) * self.columns_per_page < self.df.shape[1]:
            self.current_column_page += 1
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.update_progress(0, "Loading next columns...")
            QApplication.processEvents()  # Ensure UI updates

            self.threaded_update_model()

            self.update_progress(100, "Next columns loaded.")
            QApplication.processEvents()
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    def threaded_update_model(self):
        # This method will be run in a separate thread
        self.update_model_signal.emit(self.filtered_df, True)

    @Slot(int, str)
    def update_progress(self, value, message):
        self.progress_label.setText(message)
        self.progress_bar.setValue(value)
        if value == 100:
            QApplication.processEvents()  # Force UI update
            time.sleep(0.5)  # Short delay to show completion
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
    
    @Slot(str)
    def on_error(self, error_message):
        super().on_error(error_message) # Call existing on_error if it has other logic
        if hasattr(self, 'file_loader_thread') and self.file_loader_thread and self.file_loader_thread.isRunning():
            self.file_loader_thread.quit()
            self.file_loader_thread.wait()
        QMessageBox.critical(self, "Loading Error", error_message)

    def save_file(self):
        if self.df is not None: # Check if df exists
            original_name = os.path.splitext(os.path.basename(self._filepath if self._filepath else "data"))[0].split('.')[0]
            
            filter_info_parts = []
            for item in self.applied_filters: # self.applied_filters should store the (col, comp, val1, val2, case, ftype, regex, negate) tuples
                if isinstance(item, tuple):
                    column, comparison, filter_string, filter_string2, case_sensitive, field_type, use_regex, negate_filter = item

                    # Sanitize filter_string and filter_string2 for filename (basic)
                    fs1_clean = re.sub(r'[^a-zA-Z0-9_-]', '', str(filter_string))[:10] # Limit length
                    fs2_clean = re.sub(r'[^a-zA-Z0-9_-]', '', str(filter_string2))[:10] if filter_string2 else ""


                    comp_short = ""
                    if comparison == 'Equals': comp_short = 'eq'
                    elif comparison == 'Contains': comp_short = 'ct'
                    elif comparison == '>': comp_short = 'gt'
                    elif comparison == '<': comp_short = 'lt'
                    elif comparison == '>=': comp_short = 'ge'
                    elif comparison == '<=': comp_short = 'le'
                    elif comparison == 'Between': comp_short = 'btw'
                    
                    # Add indicators for negate and regex
                    neg_indicator = "N" if negate_filter else ""
                    rgx_indicator = "R" if use_regex else ""
                    
                    # Construct part
                    part = f"{str(column)[:6]}_{neg_indicator}{rgx_indicator}{str(field_type[:1]).lower()}_{comp_short}_{fs1_clean}"
                    if comparison == 'Between' and fs2_clean:
                        part += f"_{fs2_clean}"
                    filter_info_parts.append(part)

                elif isinstance(item, str): # Logic operator
                    filter_info_parts.append(item) 
            
            filter_string_for_filename = "_".join(filter_info_parts)
            # Further sanitize the whole filter_string_for_filename if needed
            filter_string_for_filename = re.sub(r'_+', '_', filter_string_for_filename) # Replace multiple underscores
            
            suggested_name = f"{original_name}_filtered_{filter_string_for_filename}.csv" if filter_string_for_filename else f"{original_name}_processed.csv"
            suggested_name = suggested_name[:200] + ".csv" # Limit total filename length more aggressively
            
            file_name, _ = QFileDialog.getSaveFileName(self, "Save File", suggested_name, "CSV Files (*.csv);;Excel Files (*.xlsx)")
            if file_name:
                self.progress_bar.setVisible(True)
                self.progress_label.setVisible(True)
                self.progress_bar.setValue(0)

                def progress_callback(value, message):
                    self.update_progress(value, message)
                    QApplication.processEvents() 

                def completion_callback(file_name_cb):
                    self.on_save_completed(file_name_cb)

                def error_callback(error_message):
                    self.on_save_error(error_message)

                # Determine DataFrame to save
                df_to_save = self.filtered_df if self.filtered_df is not None and not self.filtered_df.empty else self.df
                if df_to_save is None:
                    QMessageBox.warning(self, "Save Error", "No data to save.")
                    self.progress_bar.setVisible(False)
                    self.progress_label.setVisible(False)
                    return

                # Ensure we are saving the filtered data based on self.filtered_df's index if it's from self.df
                if self.filtered_df is not None and self.df is not None and self.filtered_df.index.is_monotonic_increasing and self.filtered_df.index.max() < len(self.df):
                     df_to_save = self.df.loc[self.filtered_df.index]
                elif self.filtered_df is not None:
                     df_to_save = self.filtered_df
                else:
                     df_to_save = self.df


                saver_worker = FileSaverWorker(df_to_save, file_name, progress_callback, completion_callback, error_callback, self.file_info)
                # saver_worker.start() # If you want it to run threaded, otherwise .run() for direct execution
                saver_worker.run() # .run() will block if not designed for QThread start()
                                   # Assuming FileSaverWorker is a QThread, use .start()
                                   # If it's just a class with a run method, then direct call is fine but will block UI
                                   # The provided FileSaverWorker IS a QThread, so .start() is appropriate.
                # Check if FileSaverWorker is a QThread. If so:
                # self.file_saver_thread = FileSaverWorker(...)
                # self.file_saver_thread.start()
                # For now, keeping .run() as per original structure, implies it might be blocking or the signals handle async.
                # Given the signals, it's likely meant to be a QThread.
                # Re-evaluating original code: FileSaverWorker(..).run() was used. This means it was likely blocking.
                # For a better UX, it should be self.saver_thread = FileSaverWorker(...); self.saver_thread.start()
                # Let's assume the current structure is intended.

    @Slot(str)
    def on_save_completed(self, file_name):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        QMessageBox.information(self, "Save Successful", f"File saved successfully as {file_name}")

    @Slot(str)
    def on_save_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        QMessageBox.critical(self, "Save Error", f"Error saving file: {error_message}")

    def apply_filters(self):
        if self.df is not None:
            filters_and_logic = self.filter_panel.get_filters_and_logic()
            self.applied_filters = filters_and_logic
            
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.filter_worker = FilterWorker(self.df, filters_and_logic)
            self.filter_worker.progress_updated.connect(self.update_progress)
            self.filter_worker.filter_completed.connect(self.on_filter_completed)
            self.filter_worker.start()

    def update_column_range_label(self, df):
        if df is not None:
            start_col = self.current_column_page * self.columns_per_page + 1
            end_col = min((self.current_column_page + 1) * self.columns_per_page, df.shape[1])
            total_cols = df.shape[1]
            current_rows = df.shape[0]
            
            # Check if we have information about the original number of rows
            if hasattr(self, 'original_row_count'):
                percentage = (current_rows / self.original_row_count) * 100
                percentage = int(percentage) if percentage.is_integer() else round(percentage,1) if percentage>1 else round(percentage,6) 
                row_info = f"   Rows: {current_rows:,} ({percentage}% of {self.original_row_count:,})"
            else:
                row_info = f"   Rows: {current_rows:,}"
            
            col_info = f"   Columns {start_col}-{end_col} of {total_cols}"
            self.column_range_label.setText(f"{row_info}\n{col_info}")

    @Slot(pd.DataFrame)
    def on_filter_completed(self, filtered_df):
        self.filtered_df = filtered_df
        self.update_model(filtered_df)
        self.update_column_range_label(filtered_df)  
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)

    def toggle_filter_panel(self):
        if self.filter_panel:
            self.filter_panel.setVisible(not self.filter_panel.isVisible())

    def set_row_as_header(self):
        index = self.table_view.selectionModel().currentIndex().row()
        if index >= 0:
            self.progress_bar.setVisible(True)
            self.progress_label.setVisible(True)
            self.update_progress(0, "Setting new header...")
            QApplication.processEvents()  # Ensure UI updates
            # speed_constant = min(1, self.df.shape[0]/400000)

            # # Simulate some processing time
            # for i in range(1, 101):
            #     self.update_progress(i, f"Setting new header... {i}%")
            #     QApplication.processEvents()
            #     time.sleep(0.01*(speed_constant))  # Short delay to show progress

            new_header = self.df.iloc[index]
            self.df.columns = new_header.astype(str).tolist()
            self.df.drop(self.df.index[:index+1], inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self.filtered_df = self.df.copy()  
            self.update_model(self.df)
            
            # Update filter panel with new columns
            if self.filter_panel:
                self.filter_panel.deleteLater()
            self.filter_panel = FilterPanel(self.df.columns)
            self.filter_panel.apply_button.clicked.connect(self.apply_filters)
            self.layout.insertWidget(3, self.filter_panel)
            self.filter_panel.setVisible(False)

            self.update_progress(100, "New header set successfully.")
            QApplication.processEvents()
            time.sleep(0.25)  # Show completion message briefly
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    def update_file_info_label(self):
        file_name = os.path.basename(self._filepath)
        if len(file_name) > 50:
            file_name = file_name[:50] + '...' + file_name[-6:]
        file_info = f"  File specifications for '{file_name}':\n"
        if self._filepath.lower().endswith(('.xlsx', '.xls', '.xlsm', '.xlsb')):
            file_info += f" Sheet: '{self.file_info['sheet_name']}' (Total sheets: {self.file_info['total_sheets']}) | "
        else:
            file_info += f" Delimiter: '{self.file_info.get('delimiter', 'N/A')}' | Encoding: '{self.file_info.get('encoding', 'N/A')}' | "
        file_info += f"Rows: {self.file_info['rows']:,} | Columns: {self.file_info['columns']:,} | "
        file_info += f"Size: {self.file_info['size']:,.2f} MB | "
        file_info += f"Load time: {self.file_info.get('load_time', 0):.1f} seconds"
        self.file_info_label.setText(file_info)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.filter_panel and self.filter_panel.isVisible():
                self.apply_filters()
        super().keyPressEvent(event)

