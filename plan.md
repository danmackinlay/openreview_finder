
## Project Structure with pyproject.toml

First, let's define the project dependencies using pyproject.toml:

```toml
# pyproject.toml
[project]
name = "openreview-finder"
version = "0.1.0"
description = "Semantic search for ICLR papers"
requires-python = ">=3.9"
dependencies = [
    "openreview-py>=1.0.0",
    "pandas>=1.3.0",
    "tqdm>=4.62.0",
    "click>=8.0.0",
    "chromadb>=0.4.0",
    "transformers>=4.20.0",
    "torch>=1.10.0",
    "sentence-transformers>=2.2.0",
    "tenacity>=8.0.0",
    "scikit-learn>=1.0.0",
    "tabulate>=0.8.0",
]

[project.scripts]
openreview-finder = "openreview_finder.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Core Components

Now, let's focus on the essential components for indexing and searching:

### 1. Extractor Module

```python
# openreview_finder/extractors.py
import openreview
import pandas as pd
import os
import json
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenReviewExtractor:
    def __init__(self, checkpoint_dir="./checkpoints", data_dir="./data"):
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(checkpoint_dir, "extraction_checkpoint.json")
        self.papers_file = os.path.join(data_dir, "iclr2025_papers.json")
        self.csv_file = os.path.join(data_dir, "iclr2025_papers.csv")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
    def _get_client(self):
        """Get OpenReview client with retry logic"""
        try:
            client = openreview.Client(baseurl='https://api.openreview.net')
            return client
        except Exception as e:
            logger.error(f"Error connecting to OpenReview: {e}")
            raise

    def _load_checkpoint(self):
        """Load progress checkpoint if it exists"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'oral_completed': False,
            'spotlight_completed': False,
            'poster_completed': False,
            'oral_papers': [],
            'spotlight_papers': [],
            'poster_papers': []
        }

    def _save_checkpoint(self, state):
        """Save progress checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(state, f)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
    def _get_all_notes_with_retry(self, client, invitation):
        """Get notes with retry logic"""
        return client.get_all_notes(invitation=invitation)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
    def _get_author_emails(self, client, paper_id):
        """Extract author emails for a paper"""
        try:
            authors_group = client.get_group(f'ICLR.cc/2025/Conference/Paper{paper_id}/Authors')
            return authors_group.members if hasattr(authors_group, 'members') else []
        except Exception as e:
            logger.warning(f"Could not retrieve emails for paper {paper_id}: {e}")
            return []

    def extract_papers(self):
        """Extract all ICLR 2025 papers with robust checkpointing"""
        client = self._get_client()
        checkpoint = self._load_checkpoint()

        # Check if we already have the final output
        if os.path.exists(self.papers_file) and os.path.exists(self.csv_file):
            logger.info("Paper data already exists. Loading from disk.")
            with open(self.papers_file, 'r') as f:
                papers_data = json.load(f)
            return pd.DataFrame(papers_data)

        # Categories to extract
        categories = [
            {
                'name': 'oral',
                'invitation': 'ICLR.cc/2025/Conference/-/Oral_Submission',
                'completed_key': 'oral_completed',
                'papers_key': 'oral_papers'
            },
            {
                'name': 'spotlight',
                'invitation': 'ICLR.cc/2025/Conference/-/Spotlight_Submission',
                'completed_key': 'spotlight_completed',
                'papers_key': 'spotlight_papers'
            },
            {
                'name': 'poster',
                'invitation': 'ICLR.cc/2025/Conference/-/Poster_Submission',
                'completed_key': 'poster_completed',
                'papers_key': 'poster_papers'
            }
        ]

        # Extract papers by category with checkpointing
        for category in categories:
            if not checkpoint[category['completed_key']]:
                logger.info(f"Fetching {category['name']} papers...")
                try:
                    papers = self._get_all_notes_with_retry(client, category['invitation'])
                    paper_dicts = []

                    for paper in tqdm(papers, desc=f"Processing {category['name']} papers"):
                        try:
                            # Get author emails
                            author_emails = self._get_author_emails(client, paper.number)

                            paper_dict = {
                                'id': paper.id,
                                'number': paper.number,
                                'title': paper.content['title'],
                                'abstract': paper.content.get('abstract', ''),
                                'authors': paper.content.get('authors', []),
                                'author_emails': author_emails,
                                'keywords': paper.content.get('keywords', []),
                                'pdf_url': f"https://openreview.net/pdf?id={paper.id}",
                                'forum_url': f"https://openreview.net/forum?id={paper.forum}",
                                'category': category['name']
                            }
                            paper_dicts.append(paper_dict)

                            # Save intermediate checkpoint
                            if len(paper_dicts) % 10 == 0:
                                checkpoint[category['papers_key']] = paper_dicts
                                self._save_checkpoint(checkpoint)
                        except Exception as e:
                            logger.error(f"Error processing paper {paper.id}: {e}")

                    checkpoint[category['papers_key']] = paper_dicts
                    checkpoint[category['completed_key']] = True
                    self._save_checkpoint(checkpoint)
                except Exception as e:
                    logger.error(f"Error fetching {category['name']} papers: {e}")
                    # Save the current progress
                    self._save_checkpoint(checkpoint)
                    raise

        # Combine all papers
        all_papers = (
            checkpoint['oral_papers'] +
            checkpoint['spotlight_papers'] +
            checkpoint['poster_papers']
        )

        # Save to disk
        with open(self.papers_file, 'w') as f:
            json.dump(all_papers, f)

        df = pd.DataFrame(all_papers)
        df.to_csv(self.csv_file, index=False)

        logger.info(f"Extracted {len(df)} papers from ICLR 2025")
        return df
```

### 2. Embeddings Module

```python
# openreview_finder/embeddings.py
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import chromadb
import logging
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SPECTER2EmbeddingFunction:
    """Custom embedding function for SPECTER2"""

    def __init__(self, model_name="allenai/specter2"):
        logger.info(f"Loading SPECTER2 model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

    def __call__(self, texts):
        """Generate embeddings for a list of texts"""
        embeddings = []
        batch_size = 8  # Adjust based on available GPU memory

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                # Normalize
                batch_embeddings = normalize(batch_embeddings, axis=1)
                embeddings.extend(batch_embeddings)

        return embeddings

class PaperEmbedder:
    def __init__(self, db_dir="./chroma_db", checkpoint_dir="./checkpoints"):
        self.db_dir = db_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(db_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.embedding_checkpoint = os.path.join(checkpoint_dir, "embedding_checkpoint.json")

    def create_specter2_embeddings(self, df, collection_name="iclr2025_papers", batch_size=50):
        """Create embeddings using SPECTER2 model"""
        logger.info("Setting up SPECTER2 embedding function...")
        embedding_function = SPECTER2EmbeddingFunction()

        # Create ChromaDB client and collection
        client = chromadb.PersistentClient(path=self.db_dir)

        # Check if collection exists and recreate if requested
        try:
            client.delete_collection(collection_name)
            logger.info(f"Deleted existing collection: {collection_name}")
        except:
            logger.info(f"No existing collection to delete: {collection_name}")

        # Create new collection with the embedding function
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"description": "ICLR 2025 papers with SPECTER2 embeddings"}
        )

        # Prepare texts and metadata
        texts = []
        metadatas = []
        ids = []

        for _, row in df.iterrows():
            # Combine title and abstract for embedding
            text = f"{row['title']} {row['abstract']}"

            # Create metadata dictionary
            metadata = {
                'id': row['id'],
                'number': row['number'] if 'number' in row else None,
                'title': row['title'],
                'authors': row['authors'],
                'author_emails': row['author_emails'] if 'author_emails' in row else [],
                'keywords': row['keywords'],
                'category': row['category'],
                'pdf_url': row['pdf_url'],
                'forum_url': row['forum_url']
            }

            texts.append(text)
            metadatas.append(metadata)
            ids.append(str(row['id']))

        # Add documents in batches with checkpointing
        total_batches = (len(texts) - 1) // batch_size + 1

        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            batch_metadatas = metadatas[i:batch_end]
            batch_ids = ids[i:batch_end]

            logger.info(f"Processing batch {i//batch_size + 1}/{total_batches}")

            try:
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
                logger.info(f"Added batch {i//batch_size + 1}/{total_batches} to ChromaDB")
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
                raise

        logger.info(f"Successfully created embeddings for {len(texts)} papers")
        return collection
```

### 3. Search Module

```python
# openreview_finder/search.py
import chromadb
import pandas as pd
import json
import logging
from tabulate import tabulate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PaperSearcher:
    def __init__(self, db_dir="./chroma_db", collection_name="iclr2025_papers"):
        self.db_dir = db_dir
        self.collection_name = collection_name

    def connect(self):
        """Connect to the ChromaDB collection"""
        client = chromadb.PersistentClient(path=self.db_dir)
        try:
            collection = client.get_collection(self.collection_name)
            return collection
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB: {e}")
            logger.error("Make sure you've created the index first with 'openreview-finder index'")
            return None

    def search(self, query, n_results=10, category=None, authors=None, keywords=None):
        """
        Search papers using semantic similarity

        Args:
            query (str): The search query
            n_results (int): Number of results to return
            category (str): Filter by category (oral, spotlight, poster)
            authors (list): Filter by author names
            keywords (list): Filter by keywords

        Returns:
            list: List of paper dictionaries with similarity scores
        """
        collection = self.connect()
        if not collection:
            return []

        # Build filter criteria
        where_filter = {}
        if category:
            where_filter["category"] = category

        # Execute query
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )

        # Format results
        papers = []
        for i in range(len(results['ids'][0])):
            metadata = results['metadatas'][0][i]

            # Additional filtering (for fields we can't filter with where clause)
            if authors and not any(author in metadata['authors'] for author in authors):
                continue

            if keywords and not any(keyword in metadata['keywords'] for keyword in keywords):
                continue

            paper = {
                'id': metadata['id'],
                'title': metadata['title'],
                'authors': metadata['authors'],
                'author_emails': metadata.get('author_emails', []),
                'category': metadata['category'],
                'keywords': metadata['keywords'],
                'pdf_url': metadata['pdf_url'],
                'forum_url': metadata['forum_url'],
                'similarity': results['distances'][0][i] if 'distances' in results else None
            }
            papers.append(paper)

        return papers

    def format_results(self, papers, output_format="text"):
        """Format search results in various formats"""
        if output_format == "json":
            return json.dumps(papers, indent=2)

        elif output_format == "csv":
            df = pd.DataFrame(papers)
            return df.to_csv(index=False)

        else:  # text format
            if not papers:
                return "No results found."

            table_data = []
            for i, paper in enumerate(papers):
                authors = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors += f" (+{len(paper['authors']) - 3} more)"

                emails = ", ".join(paper['author_emails'][:2]) if paper['author_emails'] else "N/A"
                if len(paper['author_emails']) > 2:
                    emails += f" (+{len(paper['author_emails']) - 2} more)"

                table_data.append([
                    i+1,
                    paper['title'],
                    authors,
                    emails,
                    paper['category'].upper(),
                    f"{paper['similarity']:.4f}" if paper['similarity'] is not None else "N/A",
                ])

            return tabulate(
                table_data,
                headers=["#", "Title", "Authors", "Emails", "Type", "Score"],
                tablefmt="fancy_grid"
            )
```

### 4. CLI Module

```python
# openreview_finder/cli.py
import click
import os
import pandas as pd
import logging
import time
from .extractors import OpenReviewExtractor
from .embeddings import PaperEmbedder
from .search import PaperSearcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('openreview_finder.log')
    ]
)
logger = logging.getLogger(__name__)

# Define common directory paths
DATA_DIR = "./data"
DB_DIR = "./chroma_db"
CHECKPOINT_DIR = "./checkpoints"

def ensure_directories(dirs):
    """Create directories if they don't exist"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

@click.group()
def cli():
    """ICLR Paper Search - Search engine for ICLR 2025 papers"""
    # Ensure all necessary directories exist
    ensure_directories([DATA_DIR, DB_DIR, CHECKPOINT_DIR])

@cli.command()
@click.option('--force', is_flag=True, help='Force reindexing even if data already exists')
@click.option('--batch-size', default=50, help='Batch size for indexing')
@click.option('--model', default='specter2', help='Embedding model to use (currently only specter2 is supported)')
def index(force, batch_size, model):
    """Extract papers from OpenReview and build the search index"""
    start_time = time.time()

    # Step 1: Extract papers from OpenReview
    logger.info("Starting paper extraction from OpenReview...")
    extractor = OpenReviewExtractor(
        checkpoint_dir=CHECKPOINT_DIR,
        data_dir=DATA_DIR
    )

    try:
        df = extractor.extract_papers()
        logger.info(f"Successfully extracted {len(df)} papers")
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        logger.info("You can run the command again to resume from the last checkpoint")
        return

    # Step 2: Generate embeddings and build index
    logger.info("Building search index with embeddings...")
    embedder = PaperEmbedder(
        db_dir=DB_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )

    try:
        if model.lower() == 'specter2':
            embedder.create_specter2_embeddings(df, batch_size=batch_size)
        else:
            logger.error(f"Unsupported model: {model}. Please use 'specter2'")
            return
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        logger.info("You can run the command again to resume from the last checkpoint")
        return

    elapsed_time = time.time() - start_time
    logger.info(f"Indexing completed in {elapsed_time:.2f} seconds")
    logger.info(f"Search index built successfully with {len(df)} papers")
    logger.info("You can now search papers using 'openreview-finder search \"your query\"'")


@cli.command()
@click.argument('query')
@click.option('--num-results', '-n', default=10, help='Number of results to return')
@click.option('--category', '-c', type=click.Choice(['oral', 'spotlight', 'poster']), help='Filter by paper category')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'csv']), default='text', help='Output format')
@click.option('--output', '-o', help='Output file path')
@click.option('--author', '-a', multiple=True, help='Filter by author name (can use multiple times)')
@click.option('--keyword', '-k', multiple=True, help='Filter by keyword (can use multiple times)')
def search(query, num_results, category, format, output, author, keyword):
    """Search for papers based on semantic similarity"""
    searcher = PaperSearcher(db_dir=DB_DIR)

    results = searcher.search(
        query=query,
        n_results=num_results,
        category=category,
        authors=list(author) if author else None,
        keywords=list(keyword) if keyword else None
    )

    formatted_results = searcher.format_results(results, output_format=format)

    if output:
        save_results(
            results if format == 'json' else formatted_results,
            output_file=output,
            output_format=format
        )
    else:
        click.echo(formatted_results)

    click.echo(f"\nFound {len(results)} papers matching your query.")

@cli.command()
@click.option('--num-papers', '-n', default=10, help='Number of papers to list')
@click.option('--category', '-c', type=click.Choice(['oral', 'spotlight', 'poster']), help='Filter by paper category')
def list(num_papers, category):
    """List available papers in the database"""
    papers_file = os.path.join(DATA_DIR, "iclr2025_papers.json")

    if not os.path.exists(papers_file):
        click.echo("No paper data found. Run 'openreview-finder index' first.")
        return

    # Load papers
    try:
        with open(papers_file, 'r') as f:
            import json
            papers = json.load(f)
    except Exception as e:
        click.echo(f"Error loading papers: {e}")
        return

    # Filter by category if specified
    if category:
        papers = [p for p in papers if p['category'] == category]

    # Limit number of papers
    papers = papers[:num_papers]

    # Display papers
    for i, paper in enumerate(papers):
        click.echo(f"\n{i+1}. {paper['title']}")
        click.echo(f"   Category: {paper['category'].upper()}")
        click.echo(f"   Authors: {', '.join(paper['authors'][:3])}" +
                  (f" (+{len(paper['authors'])-3} more)" if len(paper['authors']) > 3 else ""))
        if 'author_emails' in paper and paper['author_emails']:
            click.echo(f"   Emails: {', '.join(paper['author_emails'][:2])}" +
                      (f" (+{len(paper['author_emails'])-2} more)" if len(paper['author_emails']) > 2 else ""))
        click.echo(f"   URL: {paper['forum_url']}")

if __name__ == '__main__':
    cli()
```
