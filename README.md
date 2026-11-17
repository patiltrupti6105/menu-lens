# MenuLens

MenuLens answers questions about the supplied 190-dish menu PDF. It uses LlamaIndex to make one document per dish, Sentence Transformers to create embeddings, ChromaDB to save and search those documents, Google Gemini to phrase answers, and Gradio for chat. The query flow is ordinary Python code; there is no autonomous agent or LangChain dependency.

## Install and configure

Use Python 3.12. Create and activate a virtual environment, then install the pinned project dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Add a Google AI Studio key as `GOOGLE_API_KEY` in `.env`. Never commit that file. The optional `MENULENS_GRADIO_SHARE` setting defaults to `false`; set it to `true` only when you intend to make a temporary public Gradio link.

## Build the menu index and launch

```powershell
python -m indexing.build_index
python app.py
```

The first index build downloads the embedding model if it is not already cached. Chroma data is stored in `database/chroma_db/`. Rebuild after changing the menu PDF. Each rebuild creates a new timestamped collection, checks its count and dish IDs, then atomically updates `active_collection.json`. Earlier collections stay available for rollback and are not deleted automatically. Restart a running app after a rebuild because it caches the active vector store in memory.

## How a question is answered

1. Python reads the current question and recent chat messages, and prepares a retrieval query with only the recent exchange when the message is a follow-up.
2. Supported dietary, price, cuisine, and category filters are extracted. Explicit dish IDs use deterministic lookup; other questions use semantic search in Chroma with metadata filters. Requests for all records matching structured filters return the full filtered set.
3. Gemini receives the original question, recent conversation, structured retrieved menu records, and instructions to use only those records.
4. The app adds clickable PDF page sources for dishes named in the answer. For allergen questions it also gives the menu's cross-contact warning.

If Gemini is not configured or temporarily unavailable, MenuLens displays the retrieved records rather than inventing a natural-language answer.

## PDF and data rules

The loader preserves complete dish descriptions, section categories, explicitly stated cuisines, prices, page numbers, supported dietary values, conditional preparation notes, and separate confirmed/possible allergens. `None` means the PDF did not establish a value; it is not a safety claim. Ingredient lists remain empty unless a reliable extraction rule is added. The menu-wide allergen notice is stored in the active collection manifest. Allergen notes are indicative, cross-contact is possible, and diners should confirm with the restaurant when a note is absent.

The parser expects selectable text. A scanned image-only PDF needs OCR before indexing.

## Project layout

- `app.py`: Gradio chat and PDF viewer.
- `config.py`: environment-based settings.
- `models/dish_schema.py`: validated Pydantic dish data.
- `indexing/data_loader.py`: PDF parsing and menu disclaimer extraction.
- `indexing/document_builder.py`: LlamaIndex document construction.
- `indexing/build_index.py`: safe versioned collection creation and activation.
- `retrieval/vector_store.py`: active collection and cached embedding access.
- `retrieval/search.py`: exact lookup, structured filters, and semantic search.
- `retrieval/query_processor.py`: shared retrieval flow, context preparation, and direct Gemini call.
- `utils/citations.py`: PDF page links.

## Settings

The main overrides are `MENULENS_PDF_PATH`, `MENULENS_CHROMA_PATH`, `MENULENS_CHROMA_COLLECTION`, `MENULENS_EMBEDDING_MODEL`, `MENULENS_GEMINI_MODEL` (default `gemini-3.7-flash`), `MENULENS_GEMINI_FALLBACK_MODEL` (default `gemini-3.6-flash`), `MENULENS_GEMINI_THINKING_LEVEL` (default `low`), `MENULENS_GEMINI_REQUEST_TIMEOUT` (default 30 seconds), and `MENULENS_GRADIO_SHARE` (default `false`).
