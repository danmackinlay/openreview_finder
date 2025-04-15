#!/usr/bin/env python
import os
import json
import time
import re
import logging
import click
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from tabulate import tabulate
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

# OpenReview, chromadb, transformers, adapters
from openreview import api
import chromadb
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

# ===================
# Configuration
# ===================
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "iclr2025_papers"
API_CACHE_FILE = "./api_cache"

# Ensure required directories exist
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# ===================
# Logging Configuration
# ===================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("openreview_finder.log")],
)
logger = logging.getLogger(__name__)
logging.getLogger("chromadb").setLevel(logging.DEBUG)


# ===================
# Retry Decorator
# ===================
def with_retry(func, max_attempts=5):
    """
    Decorator to retry a function with exponential backoff.
    Detects rate limits and related errors.
    """

    def wrapper(*args, **kwargs):
        attempts = 0
        last_error = None
        while attempts < max_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempts += 1
                last_error = e
                wait_time = 2**attempts

                # Check if this is a rate-limit error.
                rate_limited = (hasattr(e, "status_code") and e.status_code == 429) or (
                    "429" in str(e)
                )
                if rate_limited:
                    match = re.search(r"try again in (\d+) seconds", str(e).lower())
                    if match:
                        wait_time = int(match.group(1)) + 1
                    else:
                        wait_time = 30
                    logger.warning(
                        f"Rate limit hit. Waiting for {wait_time}s before retrying..."
                    )
                else:
                    logger.warning(
                        f"Attempt {attempts} failed: {e}. Retrying in {wait_time}s..."
                    )
                time.sleep(wait_time)
        logger.error(f"All {max_attempts} attempts failed. Last error: {last_error}")
        raise last_error

    return wrapper


def clean_field(field):
    """Ensure that a field is a primitive type.
    If it's a dict with a 'value' key, return that value.
    If it's a list, clean each item.
    """
    if isinstance(field, dict):
        return field.get("value", field)
    elif isinstance(field, list):
        return [clean_field(item) for item in field]
    return field


def join_list_values(metadata):
    """
    Convert any list values in the metadata dict into a comma-separated string.

    Args:
        metadata (dict): The metadata dictionary.

    Returns:
        dict: A new metadata dictionary where any value that was a list is now a string.
    """
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            # Join list elements into a comma-separated string.
            sanitized[key] = ", ".join(map(str, value))
        else:
            sanitized[key] = value
    return sanitized


# ===================
# Simple Disk Cache Using shelve
# ===================
try:
    import diskcache
except ImportError:
    diskcache = None


class DiskCache:
    """
    A lightweight disk cache that uses diskcache if available.
    Falls back to shelve otherwise.
    """

    def __init__(self, cache_file):
        self.cache_file = cache_file
        if diskcache is not None:
            self.cache = diskcache.Cache(cache_file)
        else:
            self.cache = None  # will use shelve fallback

    def get(self, key):
        if self.cache is not None:
            return self.cache.get(key, default=None)
        else:
            import shelve

            with shelve.open(self.cache_file) as db:
                return db.get(key, None)

    def set(self, key, value):
        if self.cache is not None:
            self.cache.set(key, value)
        else:
            import shelve

            with shelve.open(self.cache_file, writeback=True) as db:
                db[key] = value

    def clear(self):
        if self.cache is not None:
            self.cache.clear()
        else:
            import os

            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)


# ===================
# SPECTER2 Embedder
# ===================
class SPECTER2Embedder(EmbeddingFunction):
    """SPECTER2 embedder using the adapters library."""

    def __init__(self):
        logger.info("Initializing SPECTER2 embedder...")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        logger.info("Loading proximity adapter...")
        self.model.load_adapter(
            "allenai/specter2", source="hf", load_as="proximity", set_active=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        logger.info(f"SPECTER2 model running on {self.device}")

    def __call__(self, input: Documents) -> Embeddings:
        batch_size = 8
        all_embeddings = []
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use CLS token and normalize
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
                batch_emb = batch_emb / norms
                all_embeddings.extend(batch_emb.tolist())
        logger.info(
            f"Produced embeddings of shape: {len(all_embeddings)} x {len(all_embeddings[0]) if all_embeddings else 0}"
        )
        return all_embeddings


# ===================
# Cached OpenReview Client
# ===================
class CachedOpenReviewClient:
    """
    Wraps the OpenReview client to cache API responses.
    """

    def __init__(
        self, baseurl="https://api2.openreview.net", cache_file=API_CACHE_FILE
    ):
        self.client = with_retry(api.OpenReviewClient)(baseurl=baseurl)
        self.cache = DiskCache(cache_file)

    def get_notes(self, **kwargs):
        key = f"get_notes-{json.dumps(kwargs, sort_keys=True)}"
        cached_response = self.cache.get(key)
        if cached_response is not None:
            logger.info(f"Cache hit for key: {key}")
            return cached_response
        else:
            logger.info(f"Cache miss for key: {key}. Making API call...")
            result = with_retry(self.client.get_notes)(**kwargs)
            self.cache.set(key, result)
            return result


# ===================
# OpenReview Finder
# ===================
class OpenReviewFinder:
    """
    Handles extraction, indexing, and searching of ICLR papers.
    No persistent checkpoint; extraction results are built in memory.
    """

    def __init__(self):
        self.api_client = CachedOpenReviewClient()

    def fetch_all_decision_notes(self, limit=1000):
        """Fetch all decision notes in batches and return as a list."""
        all_decision_notes = []
        offset = 0
        while True:
            batch = self.api_client.get_notes(
                invitation="ICLR.cc/2025/Conference/-/Decision",
                offset=offset,
                limit=limit,
            )
            if not batch:
                break
            all_decision_notes.extend(batch)
            offset += len(batch)
        return all_decision_notes

    def extract_papers(self):
        logger.info("Fetching papers from ICLR 2025 conference...")
        papers_dict = {}
        offset = 0

        # Fetch all decision notes in batches
        logger.info("Fetching all decision notes in paginated batches...")
        decision_notes_bulk = self.fetch_all_decision_notes(limit=1000)
        # Build mapping from forum id (or paper id) to decision text
        decision_mapping = {}
        for note in decision_notes_bulk:
            decision_text = note.content.get("decision", "").lower()
            # Using forum (or another unique identifier) as key
            decision_mapping[note.forum] = decision_text

        # Fetch submission papers in batches
        while True:
            try:
                papers = self.api_client.get_notes(
                    invitation="ICLR.cc/2025/Conference/-/Submission",
                    details="original,directReplies,tags,revisions",
                    offset=offset,
                    limit=1000,
                )
                logger.info(f"Fetched {len(papers)} papers at offset {offset}.")
                if not papers:
                    break

                for i, paper in tqdm(
                    enumerate(papers), desc=f"Processing offset {offset}"
                ):
                    # if i == 0:  # Print the first note as a sample.
                    #     print_raw_note(paper)
                    if paper.id in papers_dict:
                        continue

                    # Default category is 'poster'
                    category = "poster"
                    decision_found = False

                    # First, try to get decision from directReplies if available
                    if getattr(paper, "details", None) and getattr(
                        paper.details, "directReplies", None
                    ):
                        for reply in paper.details.directReplies:
                            if (
                                hasattr(reply, "invitation")
                                and "Decision" in reply.invitation
                            ):
                                if "decision" in reply.content:
                                    decision_text = reply.content["decision"].lower()
                                    if "oral" in decision_text:
                                        category = "oral"
                                        decision_found = True
                                    elif "spotlight" in decision_text:
                                        category = "spotlight"
                                        decision_found = True
                                    if decision_found:
                                        break

                    # If not found in directReplies, check the bulk decision mapping.
                    if not decision_found:
                        decision_text = decision_mapping.get(paper.id, "")
                        if decision_text:
                            if "oral" in decision_text:
                                category = "oral"
                            elif "spotlight" in decision_text:
                                category = "spotlight"
                    paper_data = {
                        "id": paper.id,
                        "number": paper.number if hasattr(paper, "number") else "",
                        "title": clean_field(paper.content.get("title", "[No Title]")),
                        "abstract": clean_field(paper.content.get("abstract", "")),
                        "authors": [
                            a for a in clean_field(paper.content.get("authors", []))
                        ],
                        "keywords": [
                            k.lower()
                            for k in clean_field(paper.content.get("keywords", []))
                        ],
                        "pdf_url": f"https://openreview.net/pdf?id={paper.id}",
                        "forum_url": f"https://openreview.net/forum?id={getattr(paper, 'forum', paper.id)}",
                        "category": category,
                    }
                    # logger.info(f"Extracted paper data: {paper_data}")
                    papers_dict[paper.id] = join_list_values(paper_data)

                offset += len(papers)
            except Exception as e:
                logger.error(f"Error during extraction: {e}")
                break

        papers_list = list(papers_dict.values())
        logger.info(f"Total papers extracted: {len(papers_list)}")
        return papers_list

    def build_index(self, batch_size=50, force=False):
        """
        Build (or rebuild) the search index using chromadb and SPECTER2 embeddings.
        If force is True, clears the API cache.
        """
        if force:
            logger.info("Force enabled: Clearing API cache.")
            CachedOpenReviewClient().cache.clear()

        papers = self.extract_papers()
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            if force or collection.count() < len(papers):
                chroma_client.delete_collection(COLLECTION_NAME)
                logger.info("Deleted existing collection; rebuilding index.")
                collection = None
        except Exception:
            collection = None

        if not collection:
            embedding_function = SPECTER2Embedder()
            collection = chroma_client.create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedding_function,
                metadata={"description": "ICLR 2025 papers with SPECTER2 embeddings"},
            )

        total_batches = (len(papers) + batch_size - 1) // batch_size
        for i in range(0, len(papers), batch_size):
            batch_papers = papers[i : i + batch_size]
            documents = [f"{p['title']} {p['abstract']}" for p in batch_papers]
            ids = [p["id"] for p in batch_papers]
            try:
                collection.add(documents=documents, metadatas=batch_papers, ids=ids)
                logger.info(f"Indexed batch {i // batch_size + 1}/{total_batches}.")
            except Exception as e:
                logger.error(f"Error indexing batch {i // batch_size + 1}: {e}")
                continue

        logger.info(f"Indexing complete. Indexed {len(papers)} papers.")
        return collection

    def search(
        self,
        query,
        n_results=10,
        category=None,
        author=None,
        keyword=None,
        output_format="text",
    ):
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            # Create the embedding function and pass it when getting the collection
            embedding_function = SPECTER2Embedder()
            collection = chroma_client.get_collection(
                name=COLLECTION_NAME, embedding_function=embedding_function
            )
        except Exception as e:
            logger.error(f"Error accessing ChromaDB collection: {e}")
            return "Index not built. Run 'openreview_finder index' to build the index."
          # Build filtering criteria.
        where_filter = {}
        if category:
            where_filter["category"] = {"$eq": category}

        # If no filter is provided, don't pass the where clause at all.
        query_args = dict(
            query_texts=[query],
            n_results=(n_results * 3 if (author or keyword) else n_results),
        )
        if where_filter:
            query_args["where"] = where_filter

        results = collection.query(**query_args)

        if not results["ids"][0]:
            return "No matching papers found."

        matched_papers = []
        for idx, paper_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx]
            if author:
                auth_filter = [a.lower() for a in author]
                if not any(
                    auth in " ".join(metadata["authors"]) for auth in auth_filter
                ):
                    continue
            if keyword:
                kw_filter = [k.lower() for k in keyword]
                if not any(kw in " ".join(metadata["keywords"]) for kw in kw_filter):
                    continue

            paper = {
                "id": paper_id,
                "title": metadata["title"],
                "authors": metadata["authors"],
                "category": metadata["category"],
                "keywords": metadata.get("keywords", ""),
                "pdf_url": metadata["pdf_url"],
                "forum_url": metadata["forum_url"],
                "similarity": results["distances"][0][idx]
                if "distances" in results
                else None,
            }
            matched_papers.append(paper)
            if len(matched_papers) >= n_results:
                break

        if output_format == "json":
            return json.dumps(matched_papers, indent=2)
        elif output_format == "csv":
            return pd.DataFrame(matched_papers).to_csv(index=False)
        else:
            if not matched_papers:
                return "No results found."
            table_data = []
            for idx, paper in enumerate(matched_papers):
                authors_disp = ", ".join(paper["authors"][:3])
                if len(paper["authors"]) > 3:
                    authors_disp += f" (+{len(paper['authors']) - 3} more)"
                score = (
                    f"{paper['similarity']:.4f}"
                    if paper["similarity"] is not None
                    else "N/A"
                )
                table_data.append(
                    [
                        idx + 1,
                        paper["title"],
                        authors_disp,
                        paper["category"].upper(),
                        score,
                    ]
                )
            return tabulate(
                table_data,
                headers=["#", "Title", "Authors", "Type", "Score"],
                tablefmt="fancy_grid",
            )


# ===================
# CLI Commands
# ===================
@click.group()
def cli():
    """ICLR Paper Search Utility - use semantic search on ICLR 2025 papers."""
    pass


@cli.command()
@click.option(
    "--force", is_flag=True, help="Force re-indexing and re-extraction of papers"
)
@click.option("--batch-size", default=50, help="Batch size for indexing")
def index(force, batch_size):
    """Extract papers from OpenReview and build the search index."""
    start_time = time.time()
    finder = OpenReviewFinder()
    finder.build_index(batch_size=batch_size, force=force)
    elapsed = time.time() - start_time
    click.echo(f"Indexing completed in {elapsed:.2f}s.")


@cli.command()
@click.argument("query")
@click.option("--num-results", "-n", default=10, help="Number of results to return")
@click.option(
    "--category",
    "-c",
    type=click.Choice(["oral", "spotlight", "poster"]),
    help="Filter by paper category",
)
@click.option(
    "--author", "-a", multiple=True, help="Filter by author name (multiple allowed)"
)
@click.option(
    "--keyword", "-k", multiple=True, help="Filter by keyword (multiple allowed)"
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "csv"]),
    default="text",
    help="Output format",
)
@click.option("--output", "-o", help="Path to save output")
def search(query, num_results, category, author, keyword, format, output):
    """Search for papers based on semantic similarity."""
    finder = OpenReviewFinder()
    results = finder.search(
        query,
        n_results=num_results,
        category=category,
        author=author,
        keyword=keyword,
        output_format=format,
    )
    if output:
        with open(output, "w") as f:
            f.write(results)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(results)


if __name__ == "__main__":
    cli()
