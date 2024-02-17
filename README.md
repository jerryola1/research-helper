
# Research Assistant Project

## Overview
This project aims to automate the process of searching, downloading, and analyzing academic research papers. It uses the arXiv API to search for papers related to Large Language Models (LLM), downloads them, and processes their content to make it easily accessible and analyzable.

## Features
- **Automatic Paper Download**: Searches and downloads papers from arXiv based on specified criteria.
- **Content Analysis**: Extracts and processes the content of downloaded papers, preparing them for further analysis.
- **Embedding Generation**: Uses GPT-4-based embeddings to analyze the text data.
- **Interactive Chat Model**: Enables querying the system about the content of the papers through an interactive chat interface.

## Technologies Used
- Python 3
- arXiv API for paper search and download
- `langchain_community` and `langchain` libraries for processing and embeddings
- Qdrant for vector storage and retrieval

## Setup and Installation
Ensure you have Python 3 installed on your system. Then, install the required Python packages:

```bash
pip install arxiv langchain_community
```

Clone this repository and navigate into the project directory:

```bash
git clone https://github.com/jerryola1/research-helper.git
cd research-helper
```

## Usage
To run the project, execute the following command in the project directory:

```bash
python app.py
```

This will start the process of searching for papers, downloading them, and preparing their content for analysis.

## Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
