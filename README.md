# ToMoLDA: Topic Modeling with LDA [2022]

## Overview
ToMoLDA (Topic Modeling with Latent Dirichlet Allocation) is a Python application with a GUI interface designed to detect and analyze topics from text-based news documents. The application can either scrape news content from websites in real-time or process local datasets uploaded by users. It uses the LDA (Latent Dirichlet Allocation) algorithm to identify hidden topics within text documents and determine the proportion of each topic's occurrence.

## Features
- **Web Scraping**: Real-time scraping of news articles from tempo.co
- **Local Dataset Processing**: Upload and analyze your own CSV datasets
- **Topic Detection**: Identify hidden topics in text documents using LDA
- **Topic Distribution Analysis**: Determine the proportion of topics across documents
- **Interactive Visualization**: Generate visual representations of topic models
- **Export Functionality**: Save scraped data as CSV and visualizations as HTML

## Requirements
- Python 3.x
- Required Python packages:
  - requests
  - beautifulsoup4
  - pandas
  - nltk
  - numpy
  - gensim
  - pyLDAvis
  - Sastrawi (Indonesian stemmer)
  - tkinter

## Installation

1. Clone the repository or download the source code
2. Install the required packages:

```bash
pip install requests beautifulsoup4 pandas nltk numpy gensim pyLDAvis Sastrawi
```

3. Make sure tkinter is installed (it comes pre-installed with Python)

## Usage

### Starting the Application
Run the main Python script:

```bash
python tomolda.py
```

### Tab 1: Get the Dataset
- **Scrape Web**: Scrapes news articles from tempo.co
- **Save File CSV**: Saves the scraped data as a CSV file
- **Open File CSV**: Upload a local CSV dataset for analysis

### Tab 2: ToMo by File
Process and analyze topics from uploaded CSV files:
- **Membangun Topik**: Build topic models from the uploaded dataset
- **Sebaran Topik**: Display topic distribution across documents
- **Simpan Visualisasi**: Save an interactive visualization as an HTML file

### Tab 3: ToMo by Scrape
Process and analyze topics from scraped web content:
- **Membangun Topik**: Build topic models from the scraped content
- **Sebaran Topik**: Display topic distribution across documents
- **Simpan Visualisasi**: Save an interactive visualization as an HTML file

### Tab 4: About
Information about the application and development team.

## Data Processing Pipeline
1. **Text Preprocessing**:
   - Lowercasing
   - Punctuation removal
   - Stemming (using Sastrawi for Indonesian text)
   - Stopword removal
   - Tokenization

2. **Model Building**:
   - Dictionary creation
   - Document-term matrix creation
   - LDA model training with 5 topics and 50 passes

3. **Results Analysis**:
   - Topic keywords display
   - Document-topic distribution
   - Perplexity and coherence score calculation
   - Interactive visualization

## Output Files
- **CSV Files**: Scraped data saved as "berita_MMDDYYYY.csv"
- **HTML Visualization**: Interactive topic model visualization saved as "lda.html"

## Notes
- This application is primarily designed for processing Indonesian text
- Web scraping is specifically configured for tempo.co news site
- The default configuration detects 5 topics from the corpus
- Warning: The topic modeling process might take some time depending on the dataset size

## Contributors
- Novita Anggraini - 21083010104
- Galang Surya Ramadhan - 21083010081
- Awal Lidya Musaffak - 21083010088
- Radya Ardi Ninang Pudyastuti - 21083010097
- Mohamad Ibnu Fajar Maulana - 21083010106
