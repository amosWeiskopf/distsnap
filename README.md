# distsnap

A CLI tool for computing and visualizing cosine similarity between text inputs using OpenAI embeddings.

Supports:
- Distance discovery for up to 1,000 strings (could be more, haven't yet tested)
- Text-to-embedding via OpenAI API
- Local SHA1+model embedding cache (SQLite)
- Cosine similarity matrix (parallelized)
- Matrix visualization (terminal, ≤12 items)
- Top-N similarity tuples
- Agglomerative clustering
- CSV export

You're welcome to fork & extend it. 
I hope this helps you in your line of work!
