#!/usr/bin/env python
"""
A script to extract, index, and search NeurIPS 2025 papers using OpenReview API and SPECTER2 embeddings.
This script provides a command-line interface (CLI) for searching papers based on semantic similarity.
It also includes a Gradio web interface for user-friendly interaction.
"""

import os
import json
import time
import re
import logging
import click
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from tqdm.auto import tqdm
from tabulate import tabulate
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from pprint import pformat
import diskcache
import pickle
import zlib

# OpenReview, chromadb, transformers, adapters
from openreview import api
import chromadb
from transformers import AutoTokenizer
from adapters import AutoAdapterModel


# ===================
# Venue Configuration
# ===================
@dataclass
class VenueConfig:
    """Configuration for a specific conference venue."""

    venue_id: str  # e.g., "NeurIPS.cc/2025/Conference"
    label: str  # e.g., "NeurIPS 2025"
    collection_name: str  # e.g., "neurips2025_papers"
    db_path: str  # e.g., "./chroma_db/neurips"


# NeurIPS 2025 configuration
NEURIPS2025 = VenueConfig(
    venue_id="NeurIPS.cc/2025/Conference",
    label="NeurIPS 2025",
    collection_name="neurips2025_papers",
    db_path="./chroma_db/neurips",
)

# ===================
# Configuration
# ===================
API_CACHE_FILE = "./api_cache"

# Ensure required directories exist
os.makedirs(NEURIPS2025.db_path, exist_ok=True)

# ===================
# Logging Configuration
# ===================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("openreview_finder.log")],
)
logger = logging.getLogger(__name__)
logging.getLogger("chromadb").setLevel(logging.INFO)


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
            sanitized[key] = "; ".join(map(str, value))
        else:
            sanitized[key] = value
    return sanitized


def compress_data(obj):
    """Serialize and compress an object."""
    pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return zlib.compress(pickled, level=9)


def decompress_data(data):
    """Decompress and deserialize an object."""
    decompressed = zlib.decompress(data)
    return pickle.loads(decompressed)


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
        self.cache = diskcache.Cache(
            cache_file,
            # serializer=(compress_data, decompress_data)
        )

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

    def get_all_notes(self, **kwargs):
        """Get all notes matching the criteria (API v2 style)."""
        key = f"get_all_notes-{json.dumps(kwargs, sort_keys=True)}"
        cached_response = self.cache.get(key)
        if cached_response is not None:
            logger.info(f"Cache hit for key: {key}")
            return cached_response
        else:
            logger.info(f"Cache miss for key: {key}. Making API call...")
            result = with_retry(self.client.get_all_notes)(**kwargs)
            self.cache.set(key, result)
            return result

    def get_group(self, id):
        """Get venue group metadata."""
        key = f"get_group-{id}"
        cached_response = self.cache.get(key)
        if cached_response is not None:
            logger.info(f"Cache hit for key: {key}")
            return cached_response
        else:
            logger.info(f"Cache miss for key: {key}. Making API call...")
            result = with_retry(self.client.get_group)(id=id)
            self.cache.set(key, result)
            return result


# ===================
# OpenReview Finder
# ===================
class OpenReviewFinder:
    """
    Handles extraction, indexing, and searching of conference papers.
    No persistent checkpoint; extraction results are built in memory.
    """

    def __init__(self, config: VenueConfig = NEURIPS2025):
        self.config = config
        self.api_client = CachedOpenReviewClient()
        # Single shared embedder for all indexing and querying
        self.embedding_function = SPECTER2Embedder()

    def extract_papers(self):
        logger.info(f"Fetching accepted papers from {self.config.venue_id}...")
        papers_dict = {}

        try:
            # Use API v2 style venueid filtering to get only accepted papers
            notes = self.api_client.get_all_notes(
                content={"venueid": self.config.venue_id},
                details="original,tags,revisions",
            )
            logger.info(
                f"Fetched {len(notes)} accepted papers from {self.config.label}."
            )

            for i, paper in tqdm(
                enumerate(notes), desc="Processing papers", total=len(notes)
            ):
                if i == 0:  # Print the first note as a sample
                    logger.info(f"Sample paper structure:\n{pformat(paper, indent=4)}")

                if paper.id in papers_dict:
                    continue

                # Verify this is actually an accepted paper by checking venueid
                paper_venueid = clean_field(paper.content.get("venueid", ""))
                if paper_venueid != self.config.venue_id:
                    logger.warning(
                        f"Paper {paper.id} has unexpected venueid: {paper_venueid}"
                    )
                    continue

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
                }
                # Convert list fields to semicolon-separated strings
                paper_dict = join_list_values(paper_data)
                # Keep author names with original capitalization (filtering now done in Python)
                papers_dict[paper.id] = paper_dict

        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            raise

        papers_list = list(papers_dict.values())
        logger.info(f"Total papers extracted: {len(papers_list)}")
        return papers_list

    def _load_collection(self):
        """Load the ChromaDB collection with the SPECTER2 embedder."""
        chroma_client = chromadb.PersistentClient(path=self.config.db_path)
        try:
            # First try to get the existing collection
            try:
                logger.info(
                    f"Attempting to load collection: {self.config.collection_name}"
                )
                collection = chroma_client.get_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                )
                logger.info(
                    f"Successfully loaded collection with {collection.count()} documents"
                )
                # Test the collection to make sure it's configured correctly
                return self._check_collection(collection)
            except Exception as e:
                logger.warning(f"Could not load existing collection: {e}")

                # If collection doesn't exist, create it
                logger.info(f"Creating new collection: {self.config.collection_name}")
                collection = chroma_client.create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={
                        "description": f"{self.config.label} papers with SPECTER2 embeddings"
                    },
                )
                logger.info("New collection created. Please run indexing.")
                # Test the new collection to make sure it's configured correctly
                return self._check_collection(collection)

        except Exception as e:
            logger.error(f"Error loading/creating collection: {e}")
            return None

    def build_index(self, batch_size=50, force=False):
        """
        Build (or rebuild) the search index using chromadb and SPECTER2 embeddings.
        If force is True, clears the API cache.
        """
        if force:
            logger.info("Force enabled: Clearing API cache.")
            CachedOpenReviewClient().cache.clear()

        papers = self.extract_papers()
        chroma_client = chromadb.PersistentClient(path=self.config.db_path)

        collection = None
        try:
            collection = chroma_client.get_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
            )
            existing = collection.count()
            if force or existing < len(papers):
                logger.info(
                    f"Existing collection has {existing} docs, need {len(papers)}; deleting."
                )
                chroma_client.delete_collection(self.config.collection_name)
                collection = None
            else:
                logger.info(
                    f"Collection already has {existing} docs (>= {len(papers)}); skipping reindex."
                )
                return collection
        except Exception:
            collection = None

        if collection is None:
            collection = chroma_client.create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "description": f"{self.config.label} papers with SPECTER2 embeddings"
                },
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
                # Fail fast - don't continue if embeddings are broken
                logger.error(f"Fatal error indexing batch {i // batch_size + 1}: {e}")
                raise

        logger.info(f"Indexing complete. Indexed {len(papers)} papers.")
        return collection

    def _query_papers(self, query, num_results=10, authors=None, keywords=None):
        """
        Core search functionality: semantic search via ChromaDB, then
        post-filter results in Python for author/keyword substring matching.
        """
        collection = self._load_collection()
        if not collection:
            logger.error(
                "Index not built. Run 'openreview_finder index' to build the index."
            )
            return []

        # Normalize filters for case-insensitive matching
        authors = [a.strip().lower() for a in (authors or []) if a.strip()]
        keywords = [k.strip().lower() for k in (keywords or []) if k.strip()]

        # Request more candidates than needed to give filters room to work
        # Use 5x multiplier or minimum of 100 candidates
        candidate_k = max(num_results * 5, 100)

        if authors or keywords:
            logger.info(f"Filters requested: authors={authors}, keywords={keywords}")
            logger.info(f"Requesting {candidate_k} candidates for filtering")

        # Build query arguments - no metadata filters, pure semantic search
        query_args = dict(
            query_texts=[query],
            n_results=candidate_k,
            include=["metadatas", "documents", "distances"],
        )

        logger.info(f"Executing ChromaDB query: '{query}' with n_results={candidate_k}")
        results = collection.query(**query_args)

        # Debug logging
        logger.info(f"ChromaDB returned {len(results['ids'][0]) if results['ids'] else 0} candidates")

        if not results["ids"] or not results["ids"][0]:
            logger.warning("No results returned from ChromaDB")
            return []

        # Build candidate list with metadata and distances
        candidates = []
        for idx, paper_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][idx]
            dist = (
                results["distances"][0][idx]
                if (
                    "distances" in results
                    and results["distances"]
                    and len(results["distances"]) > 0
                    and results["distances"][0] is not None
                    and idx < len(results["distances"][0])
                )
                else None
            )
            candidates.append((paper_id, metadata, dist))

        # Python-based filtering functions
        def matches_authors(meta):
            """Check if all author filters match (case-insensitive substring)."""
            if not authors:
                return True
            auth_str = meta.get("authors", "").lower()
            return all(a in auth_str for a in authors)

        def matches_keywords(meta):
            """Check if all keyword filters match (case-insensitive substring)."""
            if not keywords:
                return True
            kw_str = meta.get("keywords", "").lower()
            return all(k in kw_str for k in keywords)

        # Apply filters
        filtered = [
            (paper_id, meta, dist)
            for (paper_id, meta, dist) in candidates
            if matches_authors(meta) and matches_keywords(meta)
        ]

        # If filters eliminated everything, fall back to unfiltered results
        if not filtered and (authors or keywords):
            logger.warning(
                f"No candidates matched filters (authors={authors}, keywords={keywords}); "
                "returning top unfiltered results"
            )
            filtered = candidates

        logger.info(f"After filtering: {len(filtered)} papers remain")

        # Truncate to requested number (candidates are already sorted by distance)
        filtered = filtered[:num_results]

        # Build final paper objects
        matched_papers = []
        for paper_id, meta, dist in filtered:
            paper = {
                "id": paper_id,
                "title": meta["title"],
                "authors": meta["authors"],
                "abstract": meta.get("abstract", ""),
                "keywords": meta.get("keywords", ""),
                "pdf_url": meta["pdf_url"],
                "forum_url": meta["forum_url"],
                "similarity": (1.0 - dist) if dist is not None else None,
            }
            matched_papers.append(paper)

        return matched_papers

    def _check_collection(self, collection):
        """Check if the collection is properly configured with embeddings."""
        if collection:
            try:
                # Run a test query to check if collection is working correctly
                test_query = "test query"
                logger.info(f"Running test query on collection: '{test_query}'")
                test_results = collection.query(
                    query_texts=[test_query],
                    n_results=1,
                    include=["metadatas", "distances"],
                )

                # Log the test query results
                logger.info(f"Test query returned keys: {list(test_results.keys())}")
                if "distances" in test_results:
                    logger.info("✓ Collection test query returned distances")
                else:
                    logger.warning("✗ Collection test query did NOT return distances")

                return collection
            except Exception as e:
                logger.error(f"Collection test query failed: {e}")
                return None
        return None

    def _format_results_text(self, papers):
        """Format the results as a plain-text table for CLI output."""
        if not papers:
            return "No results found."
        table_data = []
        for idx, paper in enumerate(papers):
            authors = paper["authors"].split("; ")
            authors_disp = authors[:3] + ["et al"] if len(authors) > 3 else authors
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
                    score,
                ]
            )
        return tabulate(
            table_data,
            headers=["#", "Title", "Authors", "Score"],
            tablefmt="fancy_grid",
        )

    def _format_results_html(self, papers, query):
        """Format the results as HTML (can be used by both the static browser output and Gradio)."""
        if not papers:
            return f"<h3>No matching papers found for query: '{query}'</h3>"

        html = f"<h2>Search Results for: '{query}'</h2>"
        html += f"<p>Found {len(papers)} matching papers</p>"

        for i, paper in enumerate(papers):
            score_display = (
                f"(Score: {paper['similarity']:.4f})"
                if paper["similarity"] is not None
                else ""
            )
            # Use consistent color for all papers
            paper_color = "#7f8c8d"
            html += f"""
            <div style="margin-bottom: 25px; padding: 15px; border-left: 5px solid {paper_color}; background-color: #f9f9f9;">
                <h3 style="margin-top: 0; color: #2c3e50;">{i + 1}. {paper["title"]} {score_display}</h3>
                <p style="color: #7f8c8d;"><b>Authors:</b> {paper["authors"]}</p>
                <p><b>Abstract:</b> {paper["abstract"]}</p>
                <p><b>Keywords:</b> {paper["keywords"]}</p>
                <div>
                    <a href="{paper["pdf_url"]}" target="_blank" style="display: inline-block; margin-right: 10px; padding: 5px 10px; background-color: #3498db; color: white; text-decoration: none; border-radius: 3px;">View PDF</a>
                    <a href="{paper["forum_url"]}" target="_blank" style="display: inline-block; padding: 5px 10px; background-color: #2ecc71; color: white; text-decoration: none; border-radius: 3px;">Discussion Forum</a>
                </div>
            </div>
            """
        return html

    def _format_results_csv(self, papers):
        """Return a CSV string (using pandas for convenience)."""
        return pd.DataFrame(papers).to_csv(index=False)


def create_gradio_interface(finder):
    import gradio as gr

    def format_history_html(history_items):
        """Format search history as clickable HTML elements.

        Args:
            history_items: List of [query, results] pairs

        Returns:
            str: HTML string with clickable history items
        """
        if not history_items:
            return "<div class='search-history-empty'>No previous searches</div>"

        html = "<div class='search-history-container'>"
        for i, (query, result) in enumerate(history_items):
            # Escape quotes in the query for the data attribute
            query_esc = query.replace('"', "&quot;")
            html += f"""
            <div class="search-history-item" onclick="rerunSearch(this)" data-query="{query_esc}">
                <div class="search-query">{query}</div>
                <div class="search-results">{result}</div>
            </div>
            """
        html += "</div>"

        # Add the JavaScript function to handle clicks
        html += """
        <script>
        function rerunSearch(element) {
            const query = element.getAttribute('data-query');
            // Find the query input element - look for the search query textbox
            const queryInputs = document.querySelectorAll('input[placeholder="Enter search query..."]');
            if (queryInputs && queryInputs.length > 0) {
                const queryInput = queryInputs[0];
                queryInput.value = query;
                // Manually trigger an input event
                queryInput.dispatchEvent(new Event('input', { bubbles: true }));

                // Find and click the search button
                const searchBtns = document.querySelectorAll('button.primary');
                if (searchBtns && searchBtns.length > 0) {
                    searchBtns[0].click();
                }
            }
        }
        </script>
        <style>
        .search-history-container {
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 400px;
            overflow-y: auto;
        }
        .search-history-item {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        .search-history-item:hover {
            background-color: #f0f0f0;
        }
        .search-query {
            font-weight: bold;
            margin-bottom: 4px;
        }
        .search-results {
            font-size: 0.9em;
            color: #666;
        }
        </style>
        """
        return html

    def search_papers(
        query, num_results, author_input, keyword_input, history_html=None
    ):
        """Search for papers and update search history.

        Args:
            query: The search query string
            num_results: Maximum number of results to return
            author_input: Comma-separated author names to filter by
            keyword_input: Comma-separated keywords to filter by
            history_html: Current search history HTML

        Returns:
            tuple: (html_results, history_html) where history_html is the HTML
                  representation of the search history
        """
        # Get persistent history from our in-memory storage
        # We need to maintain history ourselves since we're using HTML
        if not hasattr(search_papers, "history_items"):
            search_papers.history_items = []

        # Parse input filters
        authors = [a.strip() for a in author_input.split(",")] if author_input else []
        keywords = (
            [k.strip() for k in keyword_input.split(",")] if keyword_input else []
        )

        # Skip empty queries
        if not query.strip():
            # Return current state unchanged
            return None, format_history_html(search_papers.history_items)

        # Perform search
        papers = finder._query_papers(query, num_results, authors, keywords)
        html_results = finder._format_results_html(papers, query)

        # Update history if this is a new query
        history_entry = [query, f"{len(papers)} results"]

        # Remove this query if it already exists
        search_papers.history_items = [
            h for h in search_papers.history_items if h[0] != query
        ]

        # Add to top of history
        search_papers.history_items.insert(0, history_entry)

        # Limit history size
        if len(search_papers.history_items) > 15:
            search_papers.history_items = search_papers.history_items[:15]

        # Format history as HTML
        history_html = format_history_html(search_papers.history_items)

        # Return the HTML results and updated history HTML
        return html_results, history_html

    with gr.Blocks(title=f"{finder.config.label} Paper Search") as app:
        gr.Markdown(f"# {finder.config.label} Paper Search Engine")

        gr.Markdown(
            "Search for papers using semantic similarity with SPECTER2 embeddings"
        )
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Enter search query...",
                )
                num_results = gr.Slider(
                    minimum=1, maximum=50, value=10, step=1, label="Number of Results"
                )
                author_filter = gr.Textbox(
                    label="Filter by Authors", placeholder="Separate by commas"
                )
                keyword_filter = gr.Textbox(
                    label="Filter by Keywords", placeholder="Separate by commas"
                )
                search_button = gr.Button("Search", variant="primary")
            with gr.Column(scale=1):
                gr.Markdown("### Search History")
                search_history = gr.HTML(
                    value="<div class='search-history-empty'>No previous searches</div>",
                    label="Click on any previous search to run it again",
                )
        results_display = gr.HTML(label="Results")

        # Add footer with credits at the bottom of the page
        with gr.Row():
            gr.HTML(f"""
            <div style="margin-top: 30px; padding-top: 10px; border-top: 1px solid #ddd; width: 100%;">
                <p style="text-align: center; color: #666;">
                    <strong>{finder.config.label} Paper Search</strong> | Developed by
                    <a href="https://danmackinlay.name" target="_blank">Dan MacKinlay</a> |
                    <a href="https://www.csiro.au/" target="_blank">CSIRO</a>
                    (Commonwealth Scientific and Industrial Research Organisation)
                </p>
            </div>
            """)

        # Handle regular search button clicks
        search_button.click(
            fn=search_papers,
            inputs=[
                query_input,
                num_results,
                author_filter,
                keyword_filter,
                search_history,
            ],
            outputs=[results_display, search_history],
        )

        # Handle enter key press in any textbox
        query_input.submit(
            fn=search_papers,
            inputs=[
                query_input,
                num_results,
                author_filter,
                keyword_filter,
                search_history,
            ],
            outputs=[results_display, search_history],
        )

        author_filter.submit(
            fn=search_papers,
            inputs=[
                query_input,
                num_results,
                author_filter,
                keyword_filter,
                search_history,
            ],
            outputs=[results_display, search_history],
        )

        keyword_filter.submit(
            fn=search_papers,
            inputs=[
                query_input,
                num_results,
                author_filter,
                keyword_filter,
                search_history,
            ],
            outputs=[results_display, search_history],
        )

        # Note: We're now handling the search history clicks via JavaScript directly
        # This allows for a more interactive experience without full page reloads
    return app


# ===================
# CLI Commands
# ===================
@click.group()
@click.version_option(version="1.0.0")
def cli():
    """NeurIPS Paper Search Utility - use semantic search on NeurIPS 2025 papers.

    Developed by Dan MacKinlay (https://danmackinlay.name)
    CSIRO - Commonwealth Scientific and Industrial Research Organisation
    """
    # Display attribution banner on startup
    click.echo("=" * 80)
    click.echo("NeurIPS 2025 Paper Search")
    click.echo("Developed by Dan MacKinlay (https://danmackinlay.name)")
    click.echo("CSIRO - Commonwealth Scientific and Industrial Research Organisation")
    click.echo("=" * 80)
    click.echo("")  # Empty line for spacing


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
def search(query, num_results, author, keyword, format, output):
    finder = OpenReviewFinder()
    papers = finder._query_papers(query, num_results, list(author), list(keyword))
    # Choose the output format using the helper functions.
    if format == "csv":
        results = finder._format_results_csv(papers)
    elif format == "json":
        results = json.dumps(papers, indent=2)
    else:
        results = finder._format_results_text(papers)

    if output:
        with open(output, "w") as f:
            f.write(results)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(results)


@cli.command()
def web():
    """Launch a Gradio web interface for searching and exploring papers."""
    finder = OpenReviewFinder()
    app = create_gradio_interface(finder)
    import webbrowser

    webbrowser.open("http://127.0.0.1:7860/", new=2, autoraise=True)
    app.launch()


if __name__ == "__main__":
    cli()
