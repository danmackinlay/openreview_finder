# ICLR Paper Search

A tool for extracting and semantically searching ICLR 2025 papers using SPECTER2 embeddings.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/openreview-finder.git
   cd openreview-finder
   ```

2. Install the package using uv:
   ```
   # Make sure uv is installed (https://github.com/astral-sh/uv)
   uv pip install -e .
   ```

   Alternatively, you can use pip:
   ```
   pip install -e .
   ```

## Features

- Extract papers with full metadata from OpenReview, including author emails
- Create semantic embeddings using SPECTER2 (specifically designed for academic papers)
- Build a robust search index with ChromaDB
- Search papers by semantic similarity, not just keywords
- Filter results by category, author, or keywords
- Export results in various formats (text, JSON, CSV)

## Usage

### Extracting and Indexing Papers

To extract papers from OpenReview and build the search index:

```
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

To search for papers:

```
openreview-finder search "transformer architecture improvements"
```

Advanced search options:

```
Options:
  -n, --num-results INTEGER       Number of results to return (default: 10)
  -c, --category [oral|spotlight|poster]
                                  Filter by paper category
  -f, --format [text|json|csv]    Output format (default: text)
  -o, --output TEXT               Output file path
  -a, --author TEXT               Filter by author name (can use multiple times)
  -k, --keyword TEXT              Filter by keyword (can use multiple times)
  --help                          Show this message and exit.
```

Examples:

```bash
# Limit to top 5 results
openreview-finder search "attention mechanism" --num-results 5

# Filter by category
openreview-finder search "graph neural networks" --category oral

# Filter by author
openreview-finder search "reinforcement learning" --author "Yoshua Bengio"

# Output in JSON format
openreview-finder search "language models" --format json

# Save results to file
openreview-finder search "diffusion models" --output results.json
```

## Technical Details

### SPECTER2 Embeddings

This tool uses the SPECTER2 model from the Allen Institute for AI, which is specifically designed for scientific papers. It creates embeddings that capture the semantic meaning of academic text better than general-purpose embedding models.

The first time you run the indexing command, it will download the SPECTER2 model (about 440MB).

### Data Storage

- Extracted paper metadata is stored in `./data/`
- Embeddings and search index are stored in `./chroma_db/`
- Checkpoints for resuming interrupted operations are stored in `./checkpoints/`

## Requirements

- Python 3.9+
- GPU support is optional but recommended for faster embedding generation

## Troubleshooting

If you encounter issues:

1. Check the log file at `openreview_finder.log` for detailed error messages
2. For network-related issues, simply run the indexing command again to resume
3. If the search index becomes corrupted, use `openreview-finder index --force` to rebuild it
