# Research Document Intelligence Assistant

An agentic research-paper assistant that combines **PDF processing, semantic retrieval, cross-encoder reranking, tool use, and LLM reasoning** to answer questions about scientific documents.

The project was built from scratch to explore the architecture behind modern RAG and AI-agent systems without relying on high-level orchestration frameworks such as LangChain or LlamaIndex.

## Overview

The application allows a user to upload a research paper in PDF format and interact with it through a chat interface.

Instead of sending the entire document to an LLM, the system processes the paper into chunks, creates vector embeddings, retrieves relevant passages for each query, reranks them with a cross-encoder, and provides the most relevant context to an LLM through an agentic tool-use workflow.

The agent can also decide when document retrieval is unnecessary and can use specialized tools for calculations, document metadata, and abstract extraction.

## Architecture

```text
                         ┌──────────────────────┐
                         │      Streamlit       │
                         │    Chat Interface    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      AI Agent        │
                         │ Structured Decisions │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┼────────────────┐
                    │               │                │
                    ▼               ▼                ▼
             Document Search    Calculator     Document Tools
                    │
                    ▼
              Query Embedding
                    │
                    ▼
             Cosine Retrieval
                    │
                    ▼
          Cross-Encoder Reranking
                    │
                    ▼
             Relevant Context
                    │
                    └──────────────► Agent
```

For cloud deployment:

```text
GitHub
   │
   ▼
Cloud Build
   │
   ▼
Docker Image
   │
   ▼
Artifact Registry
   │
   ▼
Google Cloud Run
   │
   ├──► Embedding + Reranking Models
   │
   └──► OpenAI API

API credentials
   ▲
   │
Google Secret Manager
```

## Document Processing Pipeline

Uploaded PDFs go through a custom preprocessing pipeline before retrieval.

The pipeline:

1. Extracts text page-by-page using `pypdf`.
2. Detects and removes repeated headers and footers.
3. Repairs words broken by PDF line-ending hyphenation.
4. Normalizes whitespace.
5. Preserves page information using page markers.
6. Splits the document into sentences.
7. Associates sentences with their source pages.
8. Creates overlapping chunks of approximately 200 words.

Each chunk retains metadata including its chunk ID and source page numbers.

This allows retrieved information to remain connected to its location in the original research paper.

## Retrieval and Reranking

The retrieval pipeline uses two stages.

### 1. Semantic Retrieval

Document chunks are embedded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The user's query is embedded with the same model and compared against document chunks using cosine similarity.

The highest-scoring candidates are selected for the next stage.

### 2. Cross-Encoder Reranking

Initial retrieval results are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

Unlike embedding similarity, the cross-encoder evaluates the query and candidate passage together.

The final top-ranked passages are converted into context containing:

- document text
- source page
- chunk ID
- embedding similarity score
- reranker score

This context can then be used by the agent to answer document-specific questions.

## Agent Architecture

The project implements a custom tool-using agent loop rather than relying on an agent framework.

For every step, the LLM returns a structured decision containing an action and its arguments.

Example:

```json
{
  "action": "search_documents",
  "arguments": {
    "query": "What value of H0 was obtained using DES and Low-z supernovae?"
  }
}
```

The application executes the selected tool, returns its observation to the agent, and allows the model to decide whether another tool is necessary or whether it can produce the final answer.

The loop continues until the agent selects:

```text
final_answer
```

A maximum-step limit prevents uncontrolled tool loops.

## Available Tools

### `search_documents`

Searches the uploaded research paper using semantic retrieval followed by cross-encoder reranking.

Used for factual questions whose answers must come from the uploaded document.

### `calculate`

Evaluates mathematical expressions using a restricted Python AST evaluator.

Only explicitly permitted arithmetic operators are supported instead of using unrestricted `eval()`.

### `get_document_info`

Returns basic metadata about the uploaded document, currently including:

- filename
- page count

### `get_document_abstract`

Extracts the paper's abstract.

The agent uses this tool for questions asking for a general overview or summary of the research paper.

## LLM Backend

The LLM backend is configured through environment variables:

```text
LLM_URL
LLM_MODEL
OPENAI_API_KEY
```

This allows the same application architecture to work with different OpenAI-compatible endpoints.

During development, the project was tested with a local LLM through LM Studio.

For cloud deployment, the application was configured to use the OpenAI API.

API credentials are not stored in the source code or Docker image.

## User Interface

The application uses Streamlit to provide a simple interactive interface.

Users can:

- upload a PDF research paper
- inspect its page and chunk count
- ask general questions
- ask document-specific questions
- request summaries
- perform calculations through the agent

Conversation history is maintained using Streamlit session state.

## Docker

The application is containerized with Docker.

Example build:

```bash
docker build -t research-assistant .
```

Example local run:

```bash
docker run -p 8501:8501 research-assistant
```

The container can also communicate with an OpenAI-compatible LLM endpoint through environment variables.

For example:

```bash
docker run -p 8501:8501 \
  -e LLM_URL="<LLM_ENDPOINT>" \
  -e LLM_MODEL="<MODEL_NAME>" \
  -e OPENAI_API_KEY="<API_KEY>" \
  research-assistant
```

Do not commit API keys or `.env` files containing credentials to the repository.

## Cloud Deployment

The application was successfully deployed to **Google Cloud Run**.

The deployment pipeline used:

```text
GitHub
   ↓
Developer Connect
   ↓
Cloud Build
   ↓
Artifact Registry
   ↓
Cloud Run
```

The Cloud Run deployment was configured with:

- Docker-based deployment
- 2 GiB memory
- 1 CPU
- request-based billing
- minimum instances: 0
- maximum instances: 1
- secrets provided through Google Secret Manager

A Cloud Build trigger was also configured to automatically build and deploy changes pushed to the main branch.

After validating the deployment pipeline, the automatic build trigger was disabled because the project had reached its final portfolio state.

### Demo Status

The application was successfully deployed and tested on Google Cloud Run.

The public demo is currently **disabled** to prevent unauthorized usage of the external LLM API and associated API costs.

The source code remains public and reproducible.

## Example Queries

The deployed application was tested with both general and document-specific queries.

### General knowledge

```text
What is the FRW metric?
```

The agent answers directly using the LLM when document retrieval is unnecessary.

### Calculation

```text
What is 17*19?
```

The agent selects the calculator tool and returns:

```text
323
```

### Research-paper retrieval

```text
What was the H0 value for DES + Low-z?
```

For the test cosmology paper, the retrieval and reranking pipeline identified the relevant document context and returned:

```text
H0 = 67.8 ± 1.3 km s⁻¹ Mpc⁻¹
```

### Paper overview

```text
What is this paper about?
```

The agent selects the abstract tool and uses the extracted abstract to produce a general summary of the paper.

## Tech Stack

- Python
- Streamlit
- pypdf
- Sentence Transformers
- PyTorch
- OpenAI-compatible LLM APIs
- LM Studio
- Docker
- Git / GitHub
- Google Cloud Run
- Google Cloud Build
- Artifact Registry
- Google Secret Manager

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

The current implementation intentionally keeps most of the architecture visible in `app.py` rather than hiding retrieval and agent behavior behind high-level frameworks. This makes the project useful as both a working application and a demonstration of how the individual components of a RAG and tool-using agent system interact.

## Limitations

This project was designed primarily as a learning and portfolio project rather than a production document platform.

Current limitations include:

- one active uploaded document at a time
- embeddings are generated in memory rather than stored in a persistent vector database
- no user authentication layer inside the Streamlit application
- PDF parsing depends on extractable PDF text
- abstract detection assumes a conventional research-paper structure
- retrieval parameters are manually configured
- no persistent conversation or document storage

These choices keep the architecture understandable while leaving clear directions for future extensions.

## Possible Future Improvements

Potential extensions include:

- persistent vector storage
- support for multiple documents
- improved scientific PDF parsing
- richer citation rendering
- evaluation metrics for retrieval quality
- automated RAG evaluation
- streaming LLM responses
- additional research-oriented tools
- improved mathematical/LaTeX rendering
- user authentication and usage limits for a public deployment

## Purpose

The main goal of this project was to learn the individual components behind a modern AI application by implementing them directly:

**document processing → embeddings → retrieval → reranking → RAG → tool use → agent loop → containerization → cloud deployment**

Rather than treating RAG, agents, Docker, and cloud deployment as separate concepts, the project connects them into a single end-to-end system.