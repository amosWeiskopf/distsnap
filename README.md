# distsnap

A lightweight CLI tool for computing and visualizing cosine similarity between text inputs using OpenAI embeddings in record time.

Supports:
- Distance discovery for up to 1,000 strings (could be more, haven't yet tested)
- Text-to-embedding via OpenAI API
- Local SHA1+model embedding cache (SQLite)
- Cosine similarity matrix (parallelized)
- Matrix visualization (terminal, ≤12 items)
- Top-N similarity tuples
- Agglomerative clustering
- CSV export

Future roadmap: 
1. Model abstraction layer
2. Batch & streaming modes
3. Advanced clustering algorithms
4. Visualization enhancements
5. Metadata augmentation
6. Embedding inspector tools
7. System integrations (REST, gRPC)
8. Enterprise / Scale Features

You're welcome to fork & extend it. 
I hope this helps you in your line of work!
