# MenuLens

MenuLens is a conversational menu search app for the prepared menu PDF. LlamaIndex reads PDF text and creates one document per dish, HuggingFace Sentence Transformers creates embeddings, ChromaDB persists the searchable records, LangChain tools orchestrate retrieval with Gemini, and Gradio provides the chat interface.

## Run it

Add your Google AI API key to the local `.env` file (copy `.env.example`). Then use the project virtual environment:

```cmd
.\.venv\Scripts\python.exe -m indexing.build_index
.\.venv\Scripts\python.exe app.py
```

The first index build downloads the embedding model. The index is saved under `database/chroma_db/`. Rebuild it after updating the source PDF.

## PDF parsing and safety

The menu uses explicit IDs such as `D001`, followed by a name, description, dietary wording, optional allergen note, and rupee price. The loader validates one Pydantic `Dish` per ID and stores the PDF page for citations. Prices, cuisine headings, and explicitly stated dietary labels become filterable metadata. An absent dietary or allergen statement stays unknown and does not pass a positive dietary filter.

Ingredient phrases remain in the dish description rather than being guessed into a normalized ingredient list. Conditional labels such as “vegan when served without fish sauce” remain unknown for the vegan filter. The parser expects selectable PDF text; scanned image-only menus need OCR first.

## Project modules

- `indexing/`: PDF parsing, dish document creation, and ChromaDB indexing.
- `retrieval/`: semantic search with price, cuisine, restaurant, and dietary filters.
- `agent/`: three read-only tools and a bounded Gemini tool-calling loop.
- `memory/`: recent session chat history conversion.
- `app.py`: Gradio chat interface.
- `models/`: Pydantic dish and tool argument schemas.

Settings can be overridden with `MENULENS_PDF_PATH`, `MENULENS_CHROMA_PATH`, `MENULENS_CHROMA_COLLECTION`, `MENULENS_EMBEDDING_MODEL`, `MENULENS_GEMINI_MODEL` (defaults to `gemini-3.7-flash`), `MENULENS_GEMINI_FALLBACK_MODEL` (defaults to `gemini-3.6-flash` for temporary provider errors), `MENULENS_GEMINI_THINKING_LEVEL` (defaults to `low`), `MENULENS_GEMINI_REQUEST_TIMEOUT` (defaults to `30` seconds), and `MENULENS_MAX_AGENT_TURNS`.
