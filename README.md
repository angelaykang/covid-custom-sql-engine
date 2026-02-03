# Custom SQL Operations Engine

A custom-built SQL engine implementation with a Streamlit web interface for analyzing COVID-19 data. This project demonstrates core database operations including CSV parsing, filtering, projection, grouping, aggregation, and joins - all implemented from scratch without using pandas or other data processing libraries.

## Features

- **Custom CSV Parser**: Character-by-character parsing with state machine for handling quoted fields, escaped quotes, and type coercion
- **SQL Operations**: 
  - Filtering (WHERE clause) - Boolean mask-based row selection
  - Projection (SELECT clause) - Column subset extraction
  - Group By + Aggregation - Hash-based grouping with multiple aggregation functions
  - Join - Hash join algorithm for efficient table joins
- **Interactive Dashboard**: Streamlit-based web interface with:
  - Real-time data visualization
  - Country selection and date range filtering
  - Performance metrics and execution logs
  - Statistical summaries

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── src/                # Source code modules
│   ├── __init__.py
│   ├── csv_parser.py  # Custom CSV parser implementation
│   └── dataframe.py   # DataFrame class with SQL operations
├── data/               # Data files
│   ├── country_meta.csv    # Country metadata dataset
│   └── owid-covid-data.csv # COVID-19 dataset (Our World in Data)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install streamlit psutil plotly
   ```

### Configuration

The application uses environment variables for configuration (all optional with sensible defaults):

- `COVID_DATA_FILE`: Name of the COVID-19 data CSV file (default: `owid-covid-data.csv`)
- `COUNTRY_META_FILE`: Name of the country metadata CSV file (default: `country_meta.csv`)

Example:
```bash
export COVID_DATA_FILE="my-covid-data.csv"
export COUNTRY_META_FILE="my-country-meta.csv"
streamlit run app.py
```

All file paths are relative to the project structure - no hard-coded absolute paths are used.

### Running the Application

1. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to the URL shown in the terminal (typically `http://localhost:8501`)

## Usage

### SQL Operations Demo Tab

Demonstrates the 5 core database operations:
1. **CSV Parsing** - Character-by-character parsing with state machine
2. **Filtering (WHERE)** - Boolean mask row selection - O(n)
3. **Projection (SELECT)** - Column subset extraction - O(n × k)
4. **Group By + Aggregation** - Hash-based grouping - O(n) + O(g × a)
5. **Join** - Hash join algorithm - O(n + m)

### Dashboard Tab

- Select countries from the sidebar
- Filter by date range
- View interactive visualizations:
  - Daily new cases/deaths over time
  - Cumulative totals
  - Country comparisons
  - Vaccination rates
- View detailed data tables with country metadata (via JOIN operation)

## Technical Details

### Time Complexities

- **Filtering**: O(n) where n = number of rows
- **Projection**: O(n × k) where k = number of selected columns
- **Group By**: O(n) hash-based grouping + O(g × a × m × log m) aggregation
- **Join**: O(n + m) hash join where n = left rows, m = right rows

### Data Format

The application expects CSV files with the following required columns:
- `date`
- `location`
- `new_cases_smoothed`
- `new_deaths_smoothed`
- `new_cases_smoothed_per_million`
- `new_deaths_smoothed_per_million`
- `people_fully_vaccinated_per_hundred`

## Dataset

The application uses COVID-19 data from [Our World in Data](https://ourworldindata.org/covid-vaccinations). The dataset files should be placed in the `data/` directory.

