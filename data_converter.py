import contextlib
# import pandas as pd, numpy as np, typing, sys, datetime, concurrent.futures,re, os, threading, collections
from dataclasses import dataclass
import typing
import polars as pl
import collections
import re

def datetime_parser(col_name: str):
    # 1. Basic cleaning
    base = (
        pl.col(col_name)
        .cast(pl.String)
        .replace("None", None)
        .str.strip_chars()
    )

    # 2. Excel Numeric Logic (handling days -> nanoseconds)
    as_numeric = base.cast(pl.Float64, strict=False)
    
    excel_date = (
        pl.when(
            as_numeric.is_not_null() & 
            (as_numeric > 25569) & 
            (as_numeric % 1 == 0)
        )
        .then(
            # (days - offset) * seconds_per_day * nanoseconds_per_second
            ((as_numeric - 25569) * 86_400 * 1_000_000_000).cast(pl.Datetime("ns"))
        )
        .otherwise(None)
    )

    # 3. String Cleaning (RUST COMPATIBLE REGEX)
    # Instead of look-ahead, we target the patterns directly.
    clean_str = (
        base
        # Remove milliseconds (a dot followed by 1-9 digits)
        .str.replace_all(r"\.\d+", "")
        # Remove timezone offsets like +02:00, -0500, or +0530 at the end of the string
        .str.replace_all(r"(?:\+|-)\d{2}:?\d{2}$", "")
        # Remove trailing "Z" (UTC) if present
        .str.replace_all(r"Z$", "")
    )

    # 4. The Format Cascade
    # We use a list of common formats to try before falling back to 'mixed'
    formats = [
        "%Y-%m-%d %H:%M:%S", 
        "%Y-%m-%d", 
        "%d-%m-%Y", 
        "%Y/%m/%d", 
        "%d/%m/%Y",
        "%b %d, %Y %I:%M:%S %p",  # Dec 30, 2020 12:00:00 AM
        "%b %d, %Y %I:%M %p",     # Dec 30, 2020 12:00 AM
        "%B %d, %Y %I:%M:%S %p",  # December 30, 2020 12:00:00 AM
        "%b %d, %Y %H:%M:%S",  # Dec 30, 2020 12:00:00
        "%b %d, %Y %H:%M",     # Dec 30, 2020 12:00
        "%B %d, %Y %H:%M:%S",  # December 30, 2020 12:00:00
        "%d-%b-%y",               # 30-Dec-20
        "%d %b %Y",               # 30 Dec 2020      
    ]
    
    cascade = excel_date

    for fmt in formats:
        parsed = clean_str.str.to_datetime(fmt, time_unit="ns", strict=False)
        cascade = cascade.fill_null(parsed)

    # 5. Final Fallback (Handling Short Years and Mixed Formats)
    # Short year logic: 01/01/24 -> 01/01/2024
    short_year_fix = (
        clean_str.str.replace(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2})$", r"$1-$2-20$3")
        .str.to_datetime("%d-%m-%Y", time_unit="ns", strict=False)
    )
    
    cascade = cascade.fill_null(short_year_fix)

    # Mixed fallback
    final_fallback = clean_str.str.to_datetime(format=None, time_unit="ns", strict=False)
    
    # Return as naive datetime
    return cascade.fill_null(final_fallback).dt.replace_time_zone(None)

############################################################################################################################


def number_parser(col_name: str):
    
    try:
        return pl.col(col_name).cast(pl.Float64, strict=True)
    except:
        
        raw_col = pl.col(col_name)

        base = (
            raw_col
            .cast(pl.String)
            .fill_null("0")
            .str.strip_chars()
            .replace("None", "0")
        )

        is_pct = base.str.contains("%")
        is_accounting = base.str.starts_with("(") & base.str.ends_with(")")

        clean_s = (
            base.str.replace_all(r'[€£$¥₹₽₪฿₩₫₴₸₼₾₡₢₣₤₥₦₧₨₪₯₮₱₲₳₵₶₷₸₹₺₻₼₽₾₿%]', "")
            .str.strip_chars("() ")
        )

        # Regex Logic for separators
        is_european = clean_s.str.contains(r"\..*,") 
        is_standard = clean_s.str.contains(r",.*\.")
        is_thousand_only = clean_s.str.contains(r"[,.]\d{3}$") & ~clean_s.str.contains(r"[,.]\d{1,2}$")

        slow_path = (
            pl.when(is_european)
            .then(clean_s.str.replace_all(r"\.", "").str.replace(",", "."))
            .when(is_standard)
            .then(clean_s.str.replace_all(",", ""))
            .when(is_thousand_only)
            .then(clean_s.str.replace_all(r"[,.]", ""))
            .otherwise(clean_s.str.replace(",", "."))
            .cast(pl.Float64, strict=False)
        )

        # 3. COMBINE: Use fast path if available, otherwise use slow path
        # Then apply the percentage/accounting math to the result
        final_numeric = (
            pl.when(pl.col(col_name).cast(pl.Float64, strict=False).is_not_null())
            .then(pl.col(col_name).cast(pl.Float64, strict=False))
            .otherwise(slow_path)
            .fill_null(0.0)
        )

        return (
            pl.when(is_accounting).then(final_numeric * -1.0).otherwise(final_numeric)
            .pipe(lambda x: pl.when(is_pct).then(x / 100.0).otherwise(x))
            .alias(col_name)
        )