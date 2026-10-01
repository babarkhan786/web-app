import os
import tempfile
from io import BytesIO

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fpdf import FPDF
from pypdf import PdfReader, PdfWriter


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="EDA Web Application",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):
    return str(text).encode(
        "latin-1",
        "replace"
    ).decode("latin-1")


# =====================================================
# CREATE MAIN EDA PDF
# =====================================================

def create_pdf_report(
    df,
    dataset_name,
    numeric_columns
):

    pdf = FPDF()

    pdf.set_auto_page_break(
        auto=True,
        margin=15
    )

    # =================================================
    # COVER PAGE
    # =================================================

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        20
    )

    pdf.cell(
        0,
        15,
        "Exploratory Data Analysis Report",
        ln=True,
        align="C"
    )

    pdf.ln(5)

    pdf.set_font(
        "Arial",
        "B",
        13
    )

    pdf.cell(
        0,
        10,
        clean_text(dataset_name),
        ln=True,
        align="C"
    )

    pdf.ln(15)

    pdf.set_font(
        "Arial",
        "",
        11
    )

    pdf.cell(
        0,
        8,
        f"Rows: {df.shape[0]:,}",
        ln=True,
        align="C"
    )

    pdf.cell(
        0,
        8,
        f"Columns: {df.shape[1]:,}",
        ln=True,
        align="C"
    )

    pdf.cell(
        0,
        8,
        f"Missing Values: {df.isnull().sum().sum():,}",
        ln=True,
        align="C"
    )

    pdf.cell(
        0,
        8,
        f"Duplicate Rows: {df.duplicated().sum():,}",
        ln=True,
        align="C"
    )

    # =================================================
    # DATASET OVERVIEW
    # =================================================

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        15
    )

    pdf.cell(
        0,
        10,
        "1. Dataset Overview",
        ln=True
    )

    overview = [
        ["Metric", "Value"],
        ["Rows", f"{df.shape[0]:,}"],
        ["Columns", f"{df.shape[1]:,}"],
        [
            "Missing Values",
            f"{df.isnull().sum().sum():,}"
        ],
        [
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        ],
        [
            "Numeric Variables",
            f"{len(numeric_columns):,}"
        ],
        [
            "Categorical Variables",
            f"{len(df.select_dtypes(exclude='number').columns):,}"
        ]
    ]

    pdf.set_font(
        "Arial",
        "",
        10
    )

    for row in overview:

        pdf.cell(
            75,
            8,
            clean_text(row[0]),
            border=1
        )

        pdf.cell(
            60,
            8,
            clean_text(row[1]),
            border=1,
            ln=True
        )

    # =================================================
    # DATASET PREVIEW
    # =================================================

    pdf.ln(10)

    pdf.set_font(
        "Arial",
        "B",
        15
    )

    pdf.cell(
        0,
        10,
        "2. Dataset Preview - First 10 Rows",
        ln=True
    )

    preview = df.head(10).copy()

    max_columns = 8

    preview = preview.iloc[
        :,
        :max_columns
    ]

    available_width = 170

    column_width = (
        available_width /
        max(len(preview.columns), 1)
    )

    pdf.set_font(
        "Arial",
        "B",
        6
    )

    for column in preview.columns:

        pdf.cell(
            column_width,
            7,
            clean_text(str(column)[:16]),
            border=1
        )

    pdf.ln()

    pdf.set_font(
        "Arial",
        "",
        5
    )

    for _, row in preview.iterrows():

        for value in row:

            pdf.cell(
                column_width,
                6,
                clean_text(str(value)[:18]),
                border=1
            )

        pdf.ln()

    # =================================================
    # DESCRIPTIVE STATISTICS
    # =================================================

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        15
    )

    pdf.cell(
        0,
        10,
        "3. Descriptive Statistics",
        ln=True
    )

    desc = df.describe().T

    if not desc.empty:

        headers = [
            "Variable",
            "Count",
            "Mean",
            "Std",
            "Min",
            "25%",
            "50%",
            "75%",
            "Max"
        ]

        widths = [
            35,
            17,
            20,
            20,
            20,
            20,
            20,
            20,
            20
        ]

        pdf.set_font(
            "Arial",
            "B",
            7
        )

        for header, width in zip(
            headers,
            widths
        ):

            pdf.cell(
                width,
                7,
                header,
                border=1
            )

        pdf.ln()

        pdf.set_font(
            "Arial",
            "",
            6
        )

        for index, row in desc.iterrows():

            values = [
                str(index)[:22],
                f"{row['count']:.2f}",
                f"{row['mean']:.2f}",
                f"{row['std']:.2f}",
                f"{row['min']:.2f}",
                f"{row['25%']:.2f}",
                f"{row['50%']:.2f}",
                f"{row['75%']:.2f}",
                f"{row['max']:.2f}"
            ]

            for value, width in zip(
                values,
                widths
            ):

                pdf.cell(
                    width,
                    6,
                    clean_text(value),
                    border=1
                )

            pdf.ln()

    else:

        pdf.set_font(
            "Arial",
            "",
            10
        )

        pdf.cell(
            0,
            8,
            "No numeric variables available.",
            ln=True
        )

    # =================================================
    # DATA TYPES
    # =================================================

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        15
    )

    pdf.cell(
        0,
        10,
        "4. Data Types and Variable Information",
        ln=True
    )

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    headers = [
        "Column",
        "Data Type",
        "Missing",
        "Unique"
    ]

    widths = [
        65,
        40,
        30,
        30
    ]

    pdf.set_font(
        "Arial",
        "B",
        7
    )

    for header, width in zip(
        headers,
        widths
    ):

        pdf.cell(
            width,
            7,
            header,
            border=1
        )

    pdf.ln()

    pdf.set_font(
        "Arial",
        "",
        7
    )

    for _, row in dtype_df.iterrows():

        values = [
            str(row["Column"])[:35],
            str(row["Data Type"]),
            str(row["Missing Values"]),
            str(row["Unique Values"])
        ]

        for value, width in zip(
            values,
            widths
        ):

            pdf.cell(
                width,
                6,
                clean_text(value),
                border=1
            )

        pdf.ln()

    # =================================================
    # ALL HISTOGRAMS
    # =================================================

    for column in numeric_columns:

        values = df[column].dropna()

        if len(values) == 0:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 4.5)
        )

        sns.histplot(
            values,
            kde=True,
            ax=ax
        )

        ax.set_title(
            f"Distribution of {column}"
        )

        ax.set_xlabel(
            column
        )

        ax.set_ylabel(
            "Frequency"
        )

        fig.tight_layout()

        image_buffer = BytesIO()

        fig.savefig(
            image_buffer,
            format="png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        image_buffer.seek(0)

        pdf.add_page()

        pdf.set_font(
            "Arial",
            "B",
            15
        )

        pdf.cell(
            0,
            10,
            f"5. Histogram - {column}",
            ln=True
        )

        pdf.image(
            image_buffer,
            x=15,
            y=30,
            w=180
        )

    # =================================================
    # ALL BOXPLOTS
    # =================================================

    for column in numeric_columns:

        values = df[column].dropna()

        if len(values) == 0:
            continue

        fig, ax = plt.subplots(
            figsize=(8, 3.5)
        )

        sns.boxplot(
            x=values,
            ax=ax
        )

        ax.set_title(
            f"Boxplot of {column}"
        )

        ax.set_xlabel(
            column
        )

        fig.tight_layout()

        image_buffer = BytesIO()

        fig.savefig(
            image_buffer,
            format="png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        image_buffer.seek(0)

        pdf.add_page()

        pdf.set_font(
            "Arial",
            "B",
            15
        )

        pdf.cell(
            0,
            10,
            f"6. Boxplot - {column}",
            ln=True
        )

        pdf.image(
            image_buffer,
            x=15,
            y=35,
            w=180
        )

    # =================================================
    # CORRELATION HEATMAP
    # =================================================

    if len(numeric_columns) > 1:

        corr = df[
            numeric_columns
        ].corr()

        fig_width = max(
            8,
            len(numeric_columns) * 0.55
        )

        fig_height = max(
            6,
            len(numeric_columns) * 0.5
        )

        fig, ax = plt.subplots(
            figsize=(
                fig_width,
                fig_height
            )
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
            square=True
        )

        ax.set_title(
            "Correlation Heatmap"
        )

        fig.tight_layout()

        image_buffer = BytesIO()

        fig.savefig(
            image_buffer,
            format="png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        image_buffer.seek(0)

        pdf.add_page()

        pdf.set_font(
            "Arial",
            "B",
            15
        )

        pdf.cell(
            0,
            10,
            "7. Correlation Heatmap",
            ln=True
        )

        pdf.image(
            image_buffer,
            x=10,
            y=30,
            w=190
        )

    # =================================================
    # FINAL EDA PAGE
    # =================================================

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        16
    )

    pdf.cell(
        0,
        12,
        "EDA Report Summary",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    pdf.set_font(
        "Arial",
        "",
        10
    )

    summary = (
        "This section contains the dataset overview, "
        "dataset preview, descriptive statistics, "
        "data types, distributions, boxplots, and "
        "correlation analysis."
    )

    pdf.multi_cell(
        0,
        7,
        clean_text(summary)
    )

    return bytes(
        pdf.output()
    )


# =====================================================
# CREATE COMPLETE YDATA PROFILING PDF
# =====================================================

def create_ydata_profile_pdf(
    df,
    dataset_name
):

    from ydata_profiling import ProfileReport
    from playwright.sync_api import sync_playwright

    temp_dir = tempfile.mkdtemp()

    html_file = os.path.join(
        temp_dir,
        "ydata_profile.html"
    )

    pdf_file = os.path.join(
        temp_dir,
        "ydata_profile.pdf"
    )

    # ---------------------------------------------
    # CREATE YDATA PROFILE
    # ---------------------------------------------

    profile = ProfileReport(
        df,
        title=f"YData Profiling Report - {dataset_name}",
        explorative=True
    )

    # ---------------------------------------------
    # SAVE FULL HTML REPORT
    # ---------------------------------------------

    profile.to_file(
        html_file
    )

    # ---------------------------------------------
    # CONVERT HTML TO PDF
    # USING CHROMIUM
    # ---------------------------------------------

    with sync_playwright() as p:

        browser = p.chromium.launch(
            headless=True
        )

        page = browser.new_page(
            viewport={
                "width": 1600,
                "height": 1000
            }
        )

        page.goto(
            f"file:///{html_file}",
            wait_until="networkidle"
        )

        page.pdf(
            path=pdf_file,
            format="A4",
            print_background=True,
            margin={
                "top": "10mm",
                "bottom": "10mm",
                "left": "8mm",
                "right": "8mm"
            }
        )

        browser.close()

    return pdf_file


# =====================================================
# MERGE TWO PDF FILES
# =====================================================

def merge_pdf_reports(
    main_pdf,
    profiling_pdf
):

    writer = PdfWriter()

    # ---------------------------------------------
    # MAIN EDA PDF
    # ---------------------------------------------

    main_reader = PdfReader(
        BytesIO(main_pdf)
    )

    for page in main_reader.pages:

        writer.add_page(
            page
        )

    # ---------------------------------------------
    # YDATA PROFILING PDF
    # ---------------------------------------------

    profile_reader = PdfReader(
        profiling_pdf
    )

    for page in profile_reader.pages:

        writer.add_page(
            page
        )

    output = BytesIO()

    writer.write(
        output
    )

    return output.getvalue()


# =====================================================
# TITLE
# =====================================================

st.title(
    "📊 Exploratory Data Analysis"
)

st.caption(
    "Upload a CSV file or use the example dataset for analysis."
)


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "Dataset"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

use_example = st.sidebar.checkbox(
    "Use Example Delaney Solubility Dataset"
)


# =====================================================
# LOAD DATASET
# =====================================================

if uploaded_file is not None:

    df = pd.read_csv(
        uploaded_file
    )

    dataset_name = uploaded_file.name

    st.sidebar.success(
        "CSV uploaded successfully"
    )

elif use_example:

    url = (
        "https://raw.githubusercontent.com/"
        "dataprofessor/data/master/"
        "delaney_solubility_with_descriptors.csv"
    )

    df = pd.read_csv(
        url
    )

    dataset_name = (
        "Delaney Solubility Dataset"
    )

    st.sidebar.success(
        "Example dataset loaded"
    )

else:

    df = None

    dataset_name = None

    st.sidebar.info(
        "Upload a CSV file or select the example dataset."
    )


# =====================================================
# ANALYSIS
# =====================================================

if df is not None:

    # =================================================
    # DATASET NAME
    # =================================================

    st.header(
        f"📊 {dataset_name}"
    )

    # =================================================
    # DATASET OVERVIEW
    # =================================================

    st.subheader(
        "Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

    col2.metric(
        "Columns",
        f"{df.shape[1]:,}"
    )

    col3.metric(
        "Missing Values",
        f"{df.isnull().sum().sum():,}"
    )

    col4.metric(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}"
    )

    # =================================================
    # DATASET PREVIEW
    # =================================================

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # =================================================
    # DESCRIPTIVE STATISTICS
    # =================================================

    st.subheader(
        "Descriptive Statistics"
    )

    st.dataframe(
        df.describe().T
    )

    # =================================================
    # DATA TYPES
    # =================================================

    st.subheader(
        "Data Types"
    )

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        dtype_df,
        use_container_width=True,
        height=250
    )

    # =================================================
    # NUMERIC VARIABLES
    # =================================================

    numeric_columns = (
        df.select_dtypes(
            include="number"
        ).columns.tolist()
    )

    # =================================================
    # DATA VISUALIZATION
    # =================================================

    st.subheader(
        "Data Visualization"
    )

    if numeric_columns:

        selected_column = st.selectbox(
            "Select a numeric variable",
            numeric_columns
        )

        # ---------------------------------------------
        # HISTOGRAM
        # ---------------------------------------------

        st.write(
            f"**Distribution of {selected_column}**"
        )

        fig, ax = plt.subplots(
            figsize=(7, 4)
        )

        sns.histplot(
            df[selected_column].dropna(),
            kde=True,
            ax=ax
        )

        ax.set_xlabel(
            selected_column
        )

        ax.set_ylabel(
            "Frequency"
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        plt.close(fig)

        # ---------------------------------------------
        # BOXPLOT
        # ---------------------------------------------

        st.write(
            f"**Boxplot of {selected_column}**"
        )

        fig, ax = plt.subplots(
            figsize=(7, 3)
        )

        sns.boxplot(
            x=df[selected_column],
            ax=ax
        )

        ax.set_xlabel(
            selected_column
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        plt.close(fig)

        # ---------------------------------------------
        # SUMMARY
        # ---------------------------------------------

        st.write(
            "**Summary**"
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Mean",
            f"{df[selected_column].mean():.2f}"
        )

        col2.metric(
            "Median",
            f"{df[selected_column].median():.2f}"
        )

        col3.metric(
            "Minimum",
            f"{df[selected_column].min():.2f}"
        )

        col4.metric(
            "Maximum",
            f"{df[selected_column].max():.2f}"
        )

    else:

        st.info(
            "No numeric columns available for visualization."
        )

    # =================================================
    # CORRELATION HEATMAP
    # =================================================

    if len(numeric_columns) > 1:

        st.subheader(
            "Correlation Heatmap"
        )

        corr = df[
            numeric_columns
        ].corr()

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax
        )

        fig.tight_layout()

        st.pyplot(
            fig,
            use_container_width=False
        )

        plt.close(fig)

    # =================================================
    # YDATA PROFILING
    # =================================================

    st.subheader(
        "📋 Automated Profiling"
    )

    st.caption(
        f"Profiling report for: {dataset_name}"
    )

    if st.button(
        "Generate Profiling Report",
        use_container_width=True
    ):

        with st.spinner(
            "Generating YData Profiling report..."
        ):

            try:

                from ydata_profiling import ProfileReport
                from streamlit_pandas_profiling import (
                    st_profile_report
                )

                profile = ProfileReport(
                    df,
                    title=f"EDA Profiling Report - {dataset_name}",
                    explorative=True
                )

                st_profile_report(
                    profile
                )

                st.session_state[
                    "profiling_generated"
                ] = True

                st.success(
                    "YData Profiling report generated successfully."
                )

            except Exception as e:

                st.session_state[
                    "profiling_generated"
                ] = False

                st.error(
                    f"Profiling report could not be generated: {e}"
                )

    # =================================================
    # PDF REPORT
    # =================================================

    st.subheader(
        "📄 Complete PDF Report"
    )

    st.write(
        "The PDF contains the EDA report, all graphs, "
        "and the complete YData Profiling report."
    )

    if st.button(
        "Generate Complete PDF Report",
        use_container_width=True
    ):

        with st.spinner(
            "Generating complete EDA + YData Profiling PDF..."
        ):

            try:

                # -------------------------------------
                # MAIN EDA PDF
                # -------------------------------------

                main_pdf = create_pdf_report(
                    df,
                    dataset_name,
                    numeric_columns
                )

                # -------------------------------------
                # COMPLETE YDATA PROFILE PDF
                # -------------------------------------

                profiling_pdf = create_ydata_profile_pdf(
                    df,
                    dataset_name
                )

                # -------------------------------------
                # MERGE BOTH PDFs
                # -------------------------------------

                final_pdf = merge_pdf_reports(
                    main_pdf,
                    profiling_pdf
                )

                st.download_button(
                    label="📥 Download Complete PDF Report",
                    data=final_pdf,
                    file_name=(
                        f"{dataset_name}_Complete_EDA_Report.pdf"
                    ),
                    mime="application/pdf",
                    use_container_width=True
                )

                st.success(
                    "Complete PDF generated successfully."
                )

            except Exception as e:

                st.error(
                    f"PDF report could not be generated: {e}"
                )

else:

    # =================================================
    # NO DATASET
    # =================================================

    st.info(
        "Please upload a CSV file or select the "
        "example dataset from the sidebar."
    )