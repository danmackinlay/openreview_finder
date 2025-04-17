# ICLR 2025 Paper Search

A tool for extracting and semantically searching ICLR 2025 conference papers using SPECTER2 embeddings.

**Developed by Dan MacKinlay | [CSIRO](https://www.csiro.au/) (Commonwealth Scientific and Industrial Research Organisation)**

## Features

- Extract papers with full metadata from the OpenReview API
- Create semantic embeddings using SPECTER2 (specifically designed for academic papers)
- Build a robust search index with ChromaDB
- Search papers by semantic similarity, not just keywords
- Filter by authors and keywords
- Web interface for interactive searching
- Command-line interface for batch processing

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/openreview-finder.git
   cd openreview-finder
   ```

2. Install the package using uv:
   ```bash
   # Make sure uv is installed (https://github.com/astral-sh/uv)
   uv pip install -e .
   ```

## Usage

### Extracting and Indexing Papers

To extract papers from OpenReview and build the search index:

```bash
openreview-finder index
```

This process is robust against network outages - if interrupted, you can run the same command again to resume from the last checkpoint.

Options:
```
--batch-size INTEGER  Batch size for indexing (default: 50)
--force               Force reindexing even if data already exists
--help                Show this message and exit.
```

### Searching Papers

To search for papers using the command line:

```bash
openreview-finder search "transformer architecture improvements"
```

Advanced search options:

```
Options:
  -n, --num-results INTEGER     Number of results to return (default: 10)
  -f, --format [text|json|csv]  Output format (default: text)
  -o, --output TEXT             Output file path
  -a, --author TEXT             Filter by author name (can use multiple times)
  -k, --keyword TEXT            Filter by keyword (can use multiple times)
  --help                        Show this message and exit.
```

Examples:

```bash
# Limit to top 5 results
openreview-finder search "attention mechanism" --num-results 5

# Filter by author
openreview-finder search "reinforcement learning" --author "Yoshua Bengio"

# Filter by keyword
openreview-finder search "graph neural networks" --keyword "attention"

# Output in JSON format
openreview-finder search "language models" --format json

# Save results to file
openreview-finder search "diffusion models" --output results.csv
```

### Web Interface

To launch the web interface for interactive searching:

```bash
openreview-finder web
```

This opens a Gradio web interface in your browser with:
- Search box for semantic queries
- Slider to control number of results
- Filters for authors and keywords
- Interactive search history
- Links to papers and discussions

## Technical Details

### SPECTER2 Embeddings

This tool uses the SPECTER2 model from the Allen Institute for AI, which is specifically designed for scientific papers. It creates embeddings that capture the semantic meaning of academic text better than general-purpose embedding models.

The first time you run the indexing command, it will download the SPECTER2 model (about 440MB).

### Data Storage

- Paper embeddings and search index are stored in `./chroma_db/`
- API cache is stored in `./api_cache/` to reduce API calls
- Logs are saved to `openreview_finder.log`

## Requirements

- Python 3.9+
- Dependencies include:
  - openreview-py
  - transformers/torch
  - chromadb
  - adapters
  - gradio
- GPU support is optional but recommended for faster embedding generation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Allen Institute for AI](https://allenai.org/) for the SPECTER2 model
- [OpenReview](https://openreview.net/) for providing the API
- [CSIRO](https://www.csiro.au/) for supporting this work

---

Developed by Dan MacKinlay | CSIRO (Commonwealth Scientific and Industrial Research Organisation)