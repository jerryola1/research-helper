---
title: Researcher Helper
emoji: 🐢
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.19.1
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Research Assistant Project

## Overview
This project automates the process of searching, downloading, and analyzing academic research papers. It leverages the arXiv API to search for papers related to Large Language Models (LLM), downloads them, processes their content, and provides an interactive interface to query the processed data.

## Features
- **Automatic Paper Download**: Searches and downloads papers from arXiv based on specified criteria.
- **Content Analysis**: Extracts and processes the content of downloaded papers.
- **Embedding Generation**: Uses GPT-4-based embeddings for text analysis.
- **Interactive Query Interface**: Utilizes Gradio to provide a web interface for querying the system about the content of the papers.

## Technologies Used
- Python 3
- arXiv API
- `langchain_community` and `langchain` libraries
- Qdrant for vector storage and retrieval
- Gradio for creating interactive web interfaces

## New Updates
- **Enhanced Query Capability**: The number of papers to download can now be specified, allowing for more extensive research.
- **Gradio Interface**: A Gradio interface has been added to enable easy querying through a web interface. Users can enter a search query and a question, and the system will provide answers based on the downloaded papers.

## Setup and Installation
Ensure Python 3 is installed. Install the required packages:

```bash
pip install arxiv langchain_community gradio
```

## Usage
To run the project and launch the Gradio interface:

```bash
python app.py
```

This starts a local web server. Open the displayed URL in your web browser to interact with the application.

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License.
