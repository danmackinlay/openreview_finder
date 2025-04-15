#!/usr/bin/env python
import os
import json
import click
import torch
import pandas as pd
import openreview
import chromadb
from tqdm.auto import tqdm
from tabulate import tabulate
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(), logging.FileHandler('openreview_finder.log')])
logger = logging.getLogger(__name__)

# Constants
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "iclr2025_papers"
CHECKPOINT_FILE = "./extraction_progress.json"

# Ensure directories exist
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

# Simplified retry function
def with_retry(func, max_attempts=3):
    """Simple retry decorator with fixed backoff"""
    def wrapper(*args, **kwargs):
        attempts = 0
        last_error = None

        while attempts < max_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempts += 1
                last_error = e
                wait_time = 2 ** attempts  # Simple exponential backoff
                logger.warning(f"Attempt {attempts} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logger.error(f"All {max_attempts} attempts failed. Last error: {last_error}")
        raise last_error

    return wrapper

class SPECTER2Embedder:
    """SPECTER2 embedding function for academic papers using SentenceTransformers"""

    def __init__(self, model_name="allenai/specter2"):
        logger.info(f"Loading SPECTER2 model: {model_name}")
        try:
            # Load model using SentenceTransformers
            self.model = SentenceTransformer(model_name)
            self.device = self.model.device
            logger.info(f"Successfully loaded SPECTER2 model using SentenceTransformers")
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading SPECTER2 model: {e}")
            # Fallback to a different model if needed
            logger.info("Falling back to a general-purpose model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.device = self.model.device
            logger.info(f"Using fallback model on device: {self.device}")
    
    def __call__(self, texts):
        """Generate embeddings for a list of texts"""
        try:
            # SentenceTransformer handles batching internally
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()  # Convert numpy array to list for ChromaDB
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback (not ideal but prevents crashes)
            dimension = 768  # Default embedding dimension
            return [np.zeros(dimension).tolist() for _ in texts]

class OpenReviewFinder:
    """Main class handling paper extraction, indexing and search"""

    def __init__(self):
        self.client = None
        self._load_checkpoint()

    def get_client(self):
        """Get OpenReview client with simple retry"""
        if self.client is None:
            try:
                self.client = with_retry(openreview.Client)(baseurl='https://api.openreview.net')
            except Exception as e:
                logger.error(f"Failed to connect to OpenReview: {e}")
                raise
        return self.client

    def _load_checkpoint(self):
        """Load extraction checkpoint if it exists"""
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                self.checkpoint = json.load(f)
        else:
            self.checkpoint = {
                'completed_categories': [],
                'extracted_papers': {}
            }

    def _save_checkpoint(self):
        """Save current extraction progress"""
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(self.checkpoint, f)
        logger.info(f"Saved checkpoint with {len(self.checkpoint['extracted_papers'])} papers")

    def extract_papers(self):
        """Extract papers from OpenReview with robust checkpointing"""
        client = self.get_client()

        # Categories to extract
        categories = [
            {'name': 'oral', 'invitation': 'ICLR.cc/2025/Conference/-/Oral_Submission'},
            {'name': 'spotlight', 'invitation': 'ICLR.cc/2025/Conference/-/Spotlight_Submission'},
            {'name': 'poster', 'invitation': 'ICLR.cc/2025/Conference/-/Poster_Submission'}
        ]

        # Process categories not already completed
        for category in categories:
            if category['name'] in self.checkpoint['completed_categories']:
                logger.info(f"Skipping {category['name']} papers (already completed)")
                continue

            logger.info(f"Fetching {category['name']} papers...")
            try:
                get_notes = with_retry(client.get_all_notes)
                papers = get_notes(invitation=category['invitation'])

                for paper in tqdm(papers, desc=f"Processing {category['name']} papers"):
                    # Skip if already processed
                    if paper.id in self.checkpoint['extracted_papers']:
                        continue

                    # Try to get author emails with retry
                    author_emails = []
                    try:
                        get_group = with_retry(client.get_group)
                        authors_group = get_group(f'ICLR.cc/2025/Conference/Paper{paper.number}/Authors')
                        author_emails = authors_group.members if hasattr(authors_group, 'members') else []
                    except Exception as e:
                        logger.warning(f"Could not get emails for paper {paper.number}: {e}")

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
                    self.checkpoint['extracted_papers'][paper.id] = paper_dict

                    # Save checkpoint periodically
                    if len(self.checkpoint['extracted_papers']) % 20 == 0:
                        self._save_checkpoint()

                # Mark this category as completed
                self.checkpoint['completed_categories'].append(category['name'])
                self._save_checkpoint()

            except Exception as e:
                logger.error(f"Error fetching {category['name']} papers: {e}")
                self._save_checkpoint()  # Save progress before continuing
                continue

        # Return all papers as a list
        papers = list(self.checkpoint['extracted_papers'].values())
        logger.info(f"Extracted {len(papers)} papers from ICLR 2025")
        return papers

    def build_index(self, batch_size=50, force=False):
        """Build search index with ChromaDB and SPECTER2 embeddings"""
        # First extract papers
        papers = self.extract_papers()

        # Setup ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # Check if collection exists and handle accordingly
        if force:
            try:
                client.delete_collection(COLLECTION_NAME)
                logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
            except Exception as e:
                logger.warning(f"Error deleting collection: {e}")
        else:
            try:
                collection = client.get_collection(COLLECTION_NAME)
                count = collection.count()
                logger.info(f"Found existing collection with {count} documents")
                if count >= len(papers):
                    logger.info("Collection is up-to-date, no need to rebuild")
                    return collection
                else:
                    logger.info("Collection is incomplete, rebuilding...")
                    client.delete_collection(COLLECTION_NAME)
            except Exception:
                # Collection doesn't exist yet
                pass

        # Create embedding function and collection
        embedding_function = SPECTER2Embedder()
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function,
            metadata={"description": "ICLR 2025 papers with SPECTER2 embeddings"}
        )

        # Prepare batches for adding to collection
        total_batches = (len(papers) + batch_size - 1) // batch_size

        for i in range(0, len(papers), batch_size):
            batch_end = min(i + batch_size, len(papers))
            batch_papers = papers[i:batch_end]

            # Prepare batch data
            documents = [f"{p['title']} {p['abstract']}" for p in batch_papers]
            metadatas = batch_papers  # Store entire paper object as metadata
            ids = [p['id'] for p in batch_papers]

            logger.info(f"Adding batch {i//batch_size + 1}/{total_batches} to ChromaDB")

            try:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            except Exception as e:
                logger.error(f"Error adding batch to ChromaDB: {e}")
                # Continue with next batch rather than failing completely
                continue

        logger.info(f"Successfully indexed {len(papers)} papers")
        return collection

    def search(self, query, n_results=10, category=None, author=None, keyword=None, output_format="text"):
        """Search papers using semantic similarity"""
        # Connect to ChromaDB
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        try:
            collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Error accessing ChromaDB: {e}")
            logger.error("Make sure you've built the index first with 'openreview-finder index'")
            return "Error: Search index not found. Run 'openreview-finder index' to build it."

        # Build where filter for ChromaDB
        where_filter = {}
        if category:
            where_filter["category"] = {"$eq": category}

        # Get extra results for post-filtering
        n_query = n_results * 3 if (author or keyword) else n_results

        # Execute query
        results = collection.query(
            query_texts=[query],
            n_results=n_query,
            where=where_filter
        )

        if not results["ids"][0]:
            return "No matching papers found."

        # Format results with post-filtering
        papers = []
        for i, paper_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]

            # Skip if doesn't match author filter
            if author and not any(a.lower() in " ".join(metadata["authors"]).lower() for a in author):
                continue

            # Skip if doesn't match keyword filter
            if keyword and not any(k.lower() in " ".join(metadata["keywords"]).lower() for k in keyword):
                continue

            paper = {
                'id': paper_id,
                'title': metadata['title'],
                'authors': metadata['authors'],
                'author_emails': metadata.get('author_emails', []),
                'category': metadata['category'],
                'keywords': metadata.get('keywords', []),
                'pdf_url': metadata['pdf_url'],
                'forum_url': metadata['forum_url'],
                'similarity': results["distances"][0][i] if "distances" in results else None
            }
            papers.append(paper)

            # Stop once we have enough results after filtering
            if len(papers) >= n_results:
                break

        # Format the results based on output format
        if output_format == "json":
            return json.dumps(papers, indent=2)

        elif output_format == "csv":
            return pd.DataFrame(papers).to_csv(index=False)

        else:  # text format
            if not papers:
                return "No results found."

            table_data = []
            for i, paper in enumerate(papers):
                authors = ", ".join(paper['authors'][:3])
                if len(paper['authors']) > 3:
                    authors += f" (+{len(paper['authors']) - 3} more)"

                table_data.append([
                    i+1,
                    paper['title'],
                    authors,
                    paper['category'].upper(),
                    f"{paper['similarity']:.4f}" if paper['similarity'] is not None else "N/A",
                ])

            return tabulate(
                table_data,
                headers=["#", "Title", "Authors", "Type", "Score"],
                tablefmt="fancy_grid"
            )

# CLI Commands
@click.group()
def cli():
    """ICLR Paper Search - Search for ICLR 2025 papers using semantic search"""
    pass

@cli.command()
@click.option('--force', is_flag=True, help='Force reindexing even if index already exists')
@click.option('--batch-size', default=50, help='Batch size for indexing')
def index(force, batch_size):
    """Extract papers from OpenReview and build the search index"""
    start_time = time.time()
    finder = OpenReviewFinder()
    finder.build_index(batch_size=batch_size, force=force)
    elapsed = time.time() - start_time
    click.echo(f"Indexing completed in {elapsed:.2f}s. You can now search papers.")

@cli.command()
@click.argument('query')
@click.option('--num-results', '-n', default=10, help='Number of results to return')
@click.option('--category', '-c', type=click.Choice(['oral', 'spotlight', 'poster']), help='Filter by paper category')
@click.option('--author', '-a', multiple=True, help='Filter by author name (can use multiple times)')
@click.option('--keyword', '-k', multiple=True, help='Filter by keyword (can use multiple times)')
@click.option('--format', '-f', type=click.Choice(['text', 'json', 'csv']), default='text', help='Output format')
@click.option('--output', '-o', help='Output file path')
def search(query, num_results, category, author, keyword, format, output):
    """Search for papers based on semantic similarity"""
    finder = OpenReviewFinder()
    results = finder.search(
        query=query,
        n_results=num_results,
        category=category,
        author=author,
        keyword=keyword,
        output_format=format
    )

    if output:
        with open(output, 'w') as f:
            f.write(results)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(results)

if __name__ == "__main__":
    cli()
