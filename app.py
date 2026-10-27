from pypdf import PdfReader
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
from collections import Counter
import re
import requests
import json
from io import BytesIO
import os
import ast
import operator

def normalize_boundary_line(line):

    line = line.strip()

    # Removing the first or last page number
    line = re.sub(r'^\d+\s+', '', line)
    line = re.sub(r'\s+\d+$', '', line)

    return line

def find_repeated_boundary_lines(
    pages,
    boundary_size=3,
    min_count=3
):

    boundary_counts = Counter()

    for page in pages:

        lines = [
            line.strip()
            for line in page["text"].splitlines()
            if line.strip()
        ]

        boundary_lines = (
            lines[:boundary_size]
            + lines[-boundary_size:]
        )
        
        # The same line should not be counted twice on the same page.
        boundary_lines = set(boundary_lines)

        for line in boundary_lines:

            normalized = normalize_boundary_line(line)

            if normalized:
                boundary_counts[normalized] += 1

    repeated_lines = {
        line
        for line, count in boundary_counts.items()
        if count >= min_count
    }

    return repeated_lines

 


def clean_pages(pages, repeated_lines):

    cleaned_pages = []

    for page in pages:

        page_number = page["page"]

        lines = page["text"].splitlines()

        cleaned_lines = []

        for line in lines:

            stripped = line.strip()

            if not stripped:
                continue

            # Don't clean the first page(not header but Metadata)
            if page_number != 1:

                normalized = normalize_boundary_line(
                    stripped
                )

                if normalized in repeated_lines:
                    continue

            cleaned_lines.append(stripped)

        cleaned_pages.append({
            "text": "\n".join(cleaned_lines),
            "page": page_number
        })

    return cleaned_pages




def fix_hyphenation(text):

    lines = text.splitlines()

    fixed_lines = []

    i = 0

    while i < len(lines):

        current = lines[i].strip()

        # Sonraki satır var mı?
        if i + 1 < len(lines):

            next_line = lines[i + 1].strip()

            # Mevcut satır "-" ile bitiyorsa
            if current.endswith("-") and next_line:

                # Hyphen'i kaldırıp kelimeleri birleştir
                current = current[:-1] + next_line

                fixed_lines.append(current)

                i += 2
                continue

        fixed_lines.append(current)

        i += 1

    return "\n".join(fixed_lines)








def normalize_whitespace(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def protect_abbreviations(text):

    abbreviations = [
        "et al.",
        "e.g.",
        "i.e.",
        "fig.",
        "figs.",
        "eq.",
        "eqs."
    ]

    for abbreviation in abbreviations:

        protected = abbreviation.replace(".", "<DOT>")

        text = text.replace(
            abbreviation,
            protected
        )

    return text


def combine_pages(normalized_pages):

    parts = []

    for page in normalized_pages:

        page_marker = f"<PAGE_{page['page']}>"

        parts.append(
            page_marker + " " + page["text"]
        )

    return " ".join(parts)




def split_sentences(text):

    text = protect_abbreviations(text)

    sentences = re.split(
        r'(?<=[.!?])\s+',
        text
    )

    sentences = [
        sentence.replace("<DOT>", ".")
        for sentence in sentences
    ]

    return sentences



def add_page_metadata(sentences):

    sentence_data = []

    current_page = None

    for sentence in sentences:

        pages = []

        parts = re.split(
            r"(<PAGE_\d+>)",
            sentence
        )

        for part in parts:

            page_match = re.fullmatch(
                r"<PAGE_(\d+)>",
                part
            )

            if page_match:

                current_page = int(
                    page_match.group(1)
                )

            elif part.strip():

                if current_page is not None:
                    pages.append(current_page)

        clean_text = re.sub(
            r"<PAGE_\d+>\s*",
            "",
            sentence
        ).strip()

        sentence_data.append({
            "text": clean_text,
            "pages": sorted(set(pages))
        })

    return sentence_data




def make_chunk(sentences, chunk_id):

    chunk_text = " ".join(
        sentence["text"]
        for sentence in sentences
    )

    pages = sorted(
        set(
            page
            for sentence in sentences
            for page in sentence["pages"]
        )
    )

    return {
        "text": chunk_text,
        "metadata": {
            "pages": pages,
            "chunk_id": chunk_id
        }
    }


def create_document_chunks(
    sentence_data,
    chunk_size=200,
    overlap_sentences=2
):

    chunks = []

    current_sentences = []
    current_words = 0

    chunk_id = 0

    for sentence in sentence_data:

        current_sentences.append(sentence)

        current_words += len(
            sentence["text"].split()
        )

        if current_words >= chunk_size:

            chunk = make_chunk(
                current_sentences,
                chunk_id
            )

            chunks.append(chunk)

            chunk_id += 1

            # Son birkaç cümleyi overlap olarak koru
            current_sentences = current_sentences[
                -overlap_sentences:
            ]

            current_words = sum(
                len(sentence["text"].split())
                for sentence in current_sentences
            )

    # 200 kelimeye ulaşmadan kalan son bölüm
    if current_sentences:

        chunk = make_chunk(
            current_sentences,
            chunk_id
        )

        chunks.append(chunk)

    return chunks












@st.cache_data
def process_document(pdf_bytes):
    print("PROCESSING PDF...")
    reader = PdfReader(
        BytesIO(pdf_bytes)
    )

    page_count = len(reader.pages)


    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        pages.append({
            "text": page_text,
            "page": page_number
        })

    repeated_lines = find_repeated_boundary_lines(pages)

    cleaned_pages = clean_pages(
        pages,
        repeated_lines
    )

    fixed_pages = []

    for page in cleaned_pages:

        fixed_text = fix_hyphenation(
            page["text"]
        )

        fixed_pages.append({
            "text": fixed_text,
            "page": page["page"]
        })

    normalized_pages = []

    for page in fixed_pages:

        normalized_text = normalize_whitespace(
            page["text"]
        )

        normalized_pages.append({
            "text": normalized_text,
            "page": page["page"]
        })

    document_text = combine_pages(
        normalized_pages
    )

    sentences = split_sentences(
        document_text
    )

    sentence_data = add_page_metadata(
        sentences
    )

    chunks = create_document_chunks(
        sentence_data,
        chunk_size=200,
        overlap_sentences=2
    )

    return page_count, chunks, document_text







@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return embedding_model, reranker


embedding_model, reranker = load_models()



@st.cache_data
def embed_document(chunks):
    print("CREATING DOCUMENT EMBEDDINGS...")
    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = embedding_model.encode(
        chunk_texts
    )

    return chunk_embeddings






response_format = {
    "type": "json_schema",

    "json_schema": {
        "name": "action_selection",

        "schema": {
            "type": "object",

            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "calculate",
                        "explain",
                        "answer",
                        "search_documents"
                    ]
                },

                "input": {
                    "type": "string"
                }
            },

            "required": [
                "action",
                "input"
            ],

            "additionalProperties": False
        }
    }
}





url = os.getenv(
    "LLM_URL",
    "http://localhost:1234/v1/chat/completions"
)

model = os.getenv(
    "LLM_MODEL",
    "local-model"
)

api_key = os.getenv("OPENAI_API_KEY")







def get_assistant_response(
    url,
    messages,
    model="local-model",
    temperature=None,
    response_format=None,
    timeout=120,
    api_key=None       
):# default values are given
   
    data = {
        "model": model,
        "messages": messages,
        }

    if temperature is not None:
        data["temperature"] = temperature
    
    if response_format is not None:
        data["response_format"] = response_format
    

    headers = {}

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=timeout
    )

    if not response.ok:# this part was added in copy 6 to see specifics of the error
        print("Status code:", response.status_code)
        print("Response:", response.text) 
    
    response.raise_for_status()

    result = response.json()

    if "choices" not in result:
        raise ValueError(f"Unexpected API response: {result}")

    assistant_message = result["choices"][0]["message"]

    return assistant_message, result



def search_documents(query, threshold=0.4, top_k=10):
    # threshold filters by relevance
    # top_k limits the maximum number of retrieved chunks
    
    query_embedding = embedding_model.encode(query) #embedding the query( vector)

    similarities = []

    for chunk, chunk_embedding in zip(
        chunks,
        chunk_embeddings #matching chunk to chunk_embedding
    ):

        similarity = util.cos_sim(
            query_embedding,
            chunk_embedding
        ) #similarity between the embedding vectors of the query and the chunks

        similarities.append(
            similarity.item() # append the cos similarity value into the similarities
        )

    ranked_indices = sorted(
        range(len(similarities)),
        key=lambda i: similarities[i],
        reverse=True
    ) # rank the values in similarities

    results = []

    for i in ranked_indices:

        if similarities[i] < threshold:
            continue

        results.append({
            "chunk": chunks[i]["text"],
            "similarity": similarities[i],
            "metadata": chunks[i]["metadata"]
        }) # like the name suggests we are ranking the similarities

        if len(results) >= top_k:
            break

        
    return results


def rerank_documents(query, results, top_k=5):

    pairs = []

    for result in results:
        pairs.append(
            (query, result["chunk"])
        )

    scores = reranker.predict(pairs)

    reranked_results = []

    for result, score in zip(results, scores):

        reranked_results.append({
            "chunk": result["chunk"],
            "similarity": result["similarity"],
            "reranker_score": float(score),
            "metadata": result["metadata"]
        })

    reranked_results.sort(
        key=lambda x: x["reranker_score"],
        reverse=True
    )

    return reranked_results[:top_k]




def build_context(results):

    context_parts = []

    for result in results:

        chunk = result["chunk"]
        similarity = result["similarity"]
        metadata = result["metadata"]

        reranker_score = result.get("reranker_score") #if no "reranker_score" -> None 

        context_part = (
            f"[Page: {metadata['pages']}]\n"
            f"[Chunk ID: {metadata['chunk_id']}]\n"
            f"[Similarity: {similarity:.3f}]\n"
        )

        if reranker_score is not None:
            context_part += (
                f"[Reranker Score: {reranker_score:.3f}]\n"
            )

        context_part += chunk

        context_parts.append(context_part)

    return "\n\n".join(context_parts)





allowed_operators = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod
}

def calculate(expression):

    tree = ast.parse(expression, mode="eval")

    def evaluate(node):

        if isinstance(node, ast.Expression):
            return evaluate(node.body)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numbers are allowed.")

        elif isinstance(node, ast.BinOp):
            operator_type = type(node.op)

            if operator_type not in allowed_operators:
                raise ValueError("Operator not allowed.")

            left = evaluate(node.left)
            right = evaluate(node.right)

            return allowed_operators[operator_type](left, right)

        elif isinstance(node, ast.UnaryOp):

            value = evaluate(node.operand)

            if isinstance(node.op, ast.USub):
                return -value

            elif isinstance(node.op, ast.UAdd):
                return value

            raise ValueError("Unary operator not allowed.")

        else:
            raise ValueError("Invalid mathematical expression.")

    return evaluate(tree)


def document_search_tool(
    query,
    threshold=0.2,
    top_k=10,
    rerank_top_k=5
):
    if document_text is None:
        return "No document has been uploaded."
        
    results = search_documents(
        query,
        threshold=threshold,
        top_k=top_k
    )

    if not results:
        return "No relevant information was found in the research document."

    reranked_results = rerank_documents(
        query,
        results,
        top_k=rerank_top_k
    )

    context = build_context(reranked_results)

    return context




document_info = None

def get_document_info(field):

    if document_info is None:
        return "No document has been uploaded."

    return document_info[field]






def extract_abstract(document_text):

    lower_text = document_text.lower()

    abstract_start = lower_text.find("abstract")

    if abstract_start == -1:
        return None

    introduction_start = lower_text.find(
        "1 introduction",
        abstract_start
    )

    if introduction_start == -1:
        return None

    abstract = document_text[
        abstract_start + len("abstract"):
        introduction_start
    ]

    return abstract.strip()


def get_document_abstract():

    if document_text is None:
        return "No document has been uploaded."

    abstract = extract_abstract(document_text)

    if abstract is None:
        return "No abstract could be found in the document."

    return abstract





tools = {
    "calculate": {
        "function": calculate,
        "description": (
            "Performs mathematical operations. "
            "Use this tool for arithmetic, percentages, ratios, "
            "differences, and other numerical calculations."
        ),
        "parameters": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to calculate."
            }
        }
    },

    "search_documents": {
        "function": document_search_tool,
        "description": (
            "Retrieves factual information from the research document. "
            "Use this tool when information must be obtained "
            "from the content of the document. "
            "Do not use it for mathematical calculations."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "description": "The query used to search the research document."
            }
        }
    },

    "get_document_info": {
        "function": get_document_info,
        "description": (
            "Returns general information about the uploaded research document, "
            "such as its filename or page count."
        ),
        "parameters": {
            "field": {
                "type": "string",
                "enum": ["filename", "page_count"],
                "description": "The document information to retrieve."
            }
        }
    },

    "get_document_abstract": {
        "function": get_document_abstract,
        "description": (
            "Returns the abstract of the uploaded research document. "
            "Use this tool when the user asks what the paper is about, "
            "requests an overview or general summary, or asks for the abstract. "
            "Do not use it for specific factual questions that require searching "
            "the document."
        ),
        "parameters": {}
    }
}




def build_tool_descriptions(tools):

    descriptions = []

    for name, tool in tools.items():

        descriptions.append(
            f"{name}:\n{tool['description']}"
        )

    return "\n\n".join(descriptions)

tool_descriptions = build_tool_descriptions(tools)    




def build_action_schemas(tools):

    action_schemas = []

    for name, tool in tools.items():

        parameters = tool["parameters"]

        action_schema = {
            "type": "object",
            "properties": {
                "action": {
                    "const": name
                },
                "arguments": {
                    "type": "object",
                    "properties": parameters,
                    "required": list(parameters.keys()),
                    "additionalProperties": False
                }
            },
            "required": ["action", "arguments"],
            "additionalProperties": False
        }

        action_schemas.append(action_schema)

    return action_schemas


final_answer_schema = {
    "type": "object",
    "properties": {
        "action": {
            "const": "final_answer"
        },
        "arguments": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string"
                }
            },
            "required": ["answer"],
            "additionalProperties": False
        }
    },
    "required": ["action", "arguments"],
    "additionalProperties": False
}

#Now we combine them all

action_schemas = build_action_schemas(tools)
action_schemas.append(final_answer_schema)


agent_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "agent_decision",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "calculate",
                        "search_documents",
                        "get_document_info",
                        "get_document_abstract",
                        "final_answer"
                    ]
                },
                "arguments": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string"
                                }
                            },
                            "required": ["expression"],
                            "additionalProperties": False
                        },
                        {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string"
                                }
                            },
                            "required": ["query"],
                            "additionalProperties": False
                        },
                        {
                            "type": "object",
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "enum": [
                                        "filename",
                                        "page_count"
                                    ]
                                }
                            },
                            "required": ["field"],
                            "additionalProperties": False
                        },
                        {
                            "type": "object",
                            "properties": {},
                            "required": [],
                            "additionalProperties": False
                        },
                        {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string"
                                }
                            },
                            "required": ["answer"],
                            "additionalProperties": False
                        }
                    ]
                }
            },
            "required": [
                "action",
                "arguments"
            ],
            "additionalProperties": False
        }
    }
}


agent_system_prompt = f"""
You are an AI agent.

You have access to the following tools:

{tool_descriptions}

You must choose exactly one action:

- A tool from the available tools
- final_answer: Give the final answer to the user's question.

Rules:

1. Use tools only when they are needed.

2. If the user asks for factual information from the research document
   and that information is not already available in the conversation,
   use search_documents before answering.

3. If the required document information is already available in the
   conversation, do not search for it again unless additional information
   from the document is needed.

4. If a calculation depends on information from the document that is not
   already available in the conversation, first search the document,
   observe the result, and then use calculate.

5. For any arithmetic calculation, always use the calculate tool.
   Do not perform arithmetic calculations yourself.

6. Use get_document_info for basic document information such as the
   document filename or page count.

7. If the user asks what the research document is about, requests an
   overview, or asks for a general summary of the entire document,
   use get_document_abstract.

8. You may use your own knowledge and reasoning for general explanations,
   analysis, and interpretation unless the user specifically asks for
   factual information from the research document.

9. After receiving a tool observation, use the information in the
   observation to decide the next action.

10. When using final_answer, the answer must directly address the user's
   question based on the information available so far.
   Do not use a search query or tool instruction as the final answer.

11. Do not invent document-specific information. Document-specific factual
    claims must come from the conversation or a document tool observation.

After receiving an observation, decide whether another tool call is needed
or whether you can give the final answer.

Always respond using the required JSON format.
"""




def run_agent(user_input, conversation_history, max_steps=10):

    agent_messages = [
        {
            "role": "system",
            "content": agent_system_prompt
        }
    ]

    agent_messages.extend(conversation_history)

    agent_messages.append({
        "role": "user",
        "content": user_input
    })

    for step in range(max_steps):

        assistant_message, result = get_assistant_response(
            url,
            messages=agent_messages,
            model=model,
            response_format=agent_response_format,
            api_key=api_key
        )

        decision = json.loads(
            assistant_message["content"]
        )

        print("Agent decision:", decision)

        action = decision["action"]
        arguments = decision["arguments"]

        if action in tools:

            tool_function = tools[action]["function"]

            try:
                tool_result = tool_function(**arguments)

            except Exception as e:
                tool_result = f"Tool error: {e}"

            print(f"{action} result:")

            if action == "search_documents":
                print(tool_result[:1000])

                if len(tool_result) > 1000:
                    print(
                        f"... [truncated, total {len(tool_result)} characters]"
                    )
            else:
                print(tool_result)

            agent_messages.append(assistant_message)

            agent_messages.append({
                "role": "user",
                "content": f"Observation from {action} tool:\n{tool_result}"
            })
        elif action == "final_answer":

            answer = arguments["answer"]

            conversation_history.append({
                "role": "user",
                "content": user_input
            })

            conversation_history.append({
                "role": "assistant",
                "content": answer
            })

            return answer, conversation_history

    raise RuntimeError("Agent reached the maximum number of steps.")
        





document_text = None



st.title("Research Document Intelligence Assistant")

uploaded_file = st.file_uploader(
    "Upload a research paper",
    type="pdf"
)

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    page_count, chunks, document_text = process_document(
        pdf_bytes
    )

    chunk_embeddings = embed_document(
        chunks
    )

    document_info = {
        "filename": uploaded_file.name,
        "page_count": page_count
    }


    st.write("File:", uploaded_file.name)
    st.write("Pages:", page_count)
    st.write("Chunks:", len(chunks))


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask something...")

if user_input:

    with st.chat_message("user"):
        st.write(user_input)

    assistant_response, st.session_state.messages = run_agent(
        user_input,
        st.session_state.messages
    )

    with st.chat_message("assistant"):
        st.write(assistant_response)