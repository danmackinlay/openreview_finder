When you use the `collection.query()` method in ChromaDB's Python API, it returns a dictionary containing the results of your similarity search[2]. Understanding the structure and dependability of the keys within this dictionary is crucial for effectively using the results.

## **Structure of the Query Response Dictionary**

The dictionary returned by `collection.query()` typically contains the following keys:

*   **`ids`**: A list of lists. Each inner list contains the unique identifiers (IDs) of the documents retrieved for the corresponding query[2]. The documents are ordered by similarity, with the most similar document's ID appearing first. This key is fundamental and always present in the response.
*   **`distances`**: A list of lists. Each inner list contains the distance scores between the query and the retrieved documents[2]. These scores represent similarity, where lower values indicate higher similarity. The order corresponds to the IDs and other data in the parallel lists. This key is always present.
*   **`documents`**: A list of lists. Each inner list contains the actual text content of the retrieved documents, ordered by similarity[2]. This key is present if the documents were stored with text content and if 'documents' is included in the query request (which it is by default).
*   **`metadatas`**: A list of lists. Each inner list contains the metadata dictionaries associated with the retrieved documents, ordered by similarity[2]. This key is present if metadata was stored with the documents and if 'metadatas' is included in the query request (which it is by default). If a document has no metadata, its entry might be `None` or an empty dictionary, depending on how it was added.
*   **`embeddings`**: A list of lists. Each inner list contains the vector embeddings of the retrieved documents, ordered by similarity[2]. This key is *not* typically included by default to save bandwidth and processing. You must explicitly request it using the `include` parameter in your query (e.g., `include=['embeddings', 'documents', 'distances', 'metadatas']`). If not requested or if embeddings weren't stored, the value might be `None` or omitted[1][2].

## **Dependability and Usage**

*   **Always Present**: You can depend on `ids` and `distances` always being present in the query response[2].
*   **Conditionally Present**: The presence of `documents`, `metadatas`, and `embeddings` depends on whether they were added to the collection and requested via the `include` parameter during the query. The default `include` for `query` contains `['documents', 'metadatas', 'distances']`. You can customize this list. For instance, to get only IDs, you could use `include=[]`, although this seems more relevant for `get()` as shown in search result[3] which mentions getting only IDs with `get(include=[])`. For `query`, you typically need distances at a minimum.
*   **List Structure**: Each key maps to a list of lists. The outer list has one entry for each query provided in `query_texts` or `query_embeddings`. Each inner list contains the results for that specific query, up to the number specified by `n_results`, ordered from most similar (lowest distance) to least similar[2].
*   **Accessing Results**: To use the results for the first query, you would access the first element (index 0) of each list (e.g., `results['ids']`, `results['distances']`, `results['documents']`). The first item within *these* inner lists (e.g., `results['ids']`) corresponds to the top-ranked document for that query.
*   **Interpreting Distances**: Use the `distances` values to gauge relevance. Smaller distances mean the retrieved document is semantically closer to your query[2]. The exact meaning of the distance value depends on the space used (e.g., L2, cosine)[3].

Citations:
[1] https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide
[2] https://stackoverflow.com/questions/76749728/how-chromadb-querying-system-works
[3] https://cookbook.chromadb.dev/core/collections/
[4] https://www.restack.io/p/vector-database-answer-chroma-db-usage-cat-ai
[5] https://docs.trychroma.com/reference/python/client
[6] https://github.com/chroma-core/chroma/issues/1488
[7] https://docs.trychroma.com/docs/querying-collections/query-and-get
[8] https://gist.github.com/pgolding/571aa64072d4c3d9304ee034cdcc7487


The full-text search performed by the `where_document` clause (specifically using `$contains`) in ChromaDB is **typically case-insensitive by default**.

Here's why and what that means:

1.  **Underlying Mechanism (SQLite FTS5):** ChromaDB leverages SQLite's FTS5 extension for the `where_document` functionality. Standard FTS5 tokenizers (like the default `unicode61`) perform normalization steps during indexing and querying. A key part of this normalization is **case-folding**, usually converting text to lowercase.
2.  **Effect:** Because both the indexed document text and the query term within `$contains` are typically converted to lowercase (or another normalized form) before comparison, the match becomes case-insensitive.
3.  **Result:** Searching for `{"$contains": "story"}` should match documents containing "story", "Story", "STORY", etc.

**Example Confirmation:**

Using the previous example:

```python
import chromadb

# Setup client and collection
client = chromadb.Client()
collection_name = "case_insensitive_example"
if collection_name in [c.name for c in client.list_collections()]:
    client.delete_collection(name=collection_name)
collection = client.get_or_create_collection(name=collection_name)

# Add documents with varying cases
collection.add(
    documents=[
        "This document contains the word Story.",
        "This one has STORY in uppercase.",
        "And this uses lowercase story.",
        "This document is about tales." # Should not match
    ],
    ids=["id1", "id2", "id3", "id4"]
)

# Query using lowercase
results = collection.query(
    query_texts=["narrative"], # Query text for semantic similarity (less relevant here)
    n_results=4,
    where_document={"$contains": "story"} # Case-insensitive search for 'story'
)

print("IDs found:", results['ids'])

# Query using uppercase
results_upper = collection.query(
    query_texts=["narrative"],
    n_results=4,
    where_document={"$contains": "STORY"} # Also case-insensitive
)
print("IDs found (uppercase query):", results_upper['ids'])
```

**Expected Output:**

Both queries should return the IDs of the documents containing "story" regardless of its original case:

```
IDs found: [['id1', 'id2', 'id3']]
IDs found (uppercase query): [['id1', 'id2', 'id3']]
```
*(The exact order within the inner list might vary based on distance scores, but all three relevant IDs should be present).*

**In summary:** You generally don't need to do anything extra to achieve case-insensitivity with `where_document={"$contains": "..."}`; it's the standard behavior due to the underlying FTS5 implementation used by ChromaDB.
