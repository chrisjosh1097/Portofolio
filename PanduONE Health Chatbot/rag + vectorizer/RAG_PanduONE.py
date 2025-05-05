import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# === Configurations ===
GENAI_API_KEY = "AIzaSyCCAyL9PYMyBMnwZYbo1P6B9a3JPlqr58o" #AIzaSyDRmWBcjJ8-WKtlNhlUda5W5Cp8bo3P5GQ"  # Replace with your key
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RULES_PATH = "C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\pandu.one-sehat\\PanduOne_Resources\\rules.txt"  
VECTOR_INDEX_PATH = "C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\pandu.one-sehat\\PanduOne_Resources\\Chunks\\pdf_index.faiss"
CHUNKS_PATH = "C:\\Users\\chris\\OneDrive\\Documents\\GitHub\\pandu.one-sehat\\PanduOne_Resources\\FAISS\\doc_chunks.txt"
TOP_K = 8

# === Setup Functions ===   
class RAGConversationalAgent:
    def __init__(self, genai_api_key, vector_index_path, chunks_path, embedding_model_name="all-MiniLM-L6-v2", top_k=3, rules_path=None):
        self.embed_model = SentenceTransformer(embedding_model_name)
        self.vector_index = faiss.read_index(vector_index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = f.readlines()

        self.genai_model = self._init_gemini(genai_api_key)

        # Load the rules from file if a path is given
        if RULES_PATH:
            with open(RULES_PATH, "r", encoding="utf-8") as f:
                self.rules = f.read().strip()

        self.top_k = top_k
        self.history = []  # List of (user_prompt, model_response)


    def _init_gemini(self, api_key):
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name="gemini-2.0-pro-exp")


    def _embed(self, text):
        return self.embed_model.encode([text])

    def _retrieve_context(self, vector):
        D, I = self.vector_index.search(np.array(vector), k=self.top_k)
        context_chunks = []
        for i in I[0]:
            line = self.chunks[i].strip()
            if "\t" in line:
                filename, chunk = line.split("\t", 1)
                context_chunks.append(f"[{filename}]: {chunk}")
            else:
                context_chunks.append(line)  # Fallback in case format is old
        return context_chunks


    def _build_prompt(self, user_prompt, retrieved_context):
        history_str = "\n".join(
            [f"User: {q}\nAssistant: {a}" for q, a in self.history]
        )
        context_str = "\n".join(retrieved_context)
        return f"{self.rules}\n\nContext:\n{context_str}\n\nConversation:\n{history_str}\nUser: {user_prompt}"

    def _generate_response(self, full_prompt):
        response = self.genai_model.generate_content(full_prompt)
        return response.text

    def chat(self, user_prompt):
        # Combine user prompt with history for embedding
        if self.history:
            last_q, last_a = self.history[-1]
            combined_text = f"{last_q}\n{last_a}\n{user_prompt}"
        else:
            combined_text = user_prompt

        query_vector = self._embed(combined_text)
        retrieved_context = self._retrieve_context(query_vector)
        full_prompt = self._build_prompt(user_prompt, retrieved_context)
        response = self._generate_response(full_prompt)

        # Save to history
        self.history.append((user_prompt, response))
        return response

# === Main Handler ===
def rag_gemini_pipeline(user_input, embed_model, vector_index, chunks, gemini_model):
    query_vector = embed_query(user_input, embed_model)
    top_indices = search_vector_db(query_vector, vector_index)
    context_chunks = []
    for i in top_indices:
        line = chunks[i].strip()
        if "\t" in line:
            filename, chunk = line.split("\t", 1)
            context_chunks.append(f"[{filename}]: {chunk}")
        else:
            context_chunks.append(line)


    full_prompt = build_prompt(user_input, context_chunks)
    return generate_response(full_prompt, gemini_model)

# === Example Usage ===
if __name__ == "__main__":
    agent = RAGConversationalAgent(
        genai_api_key=GENAI_API_KEY,
        vector_index_path=VECTOR_INDEX_PATH,
        chunks_path=CHUNKS_PATH
    )

    print("💬 RAG-Gemini Chat | Type 'exit' to quit\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in {"exit", "quit"}:
            print("👋 Exiting chat. Goodbye!")
            break

        response = agent.chat(user_input)
        print("Gemini:", response, "\n")


# import google.generativeai as genai

# genai.configure(api_key="AIzaSyDRmWBcjJ8-WKtlNhlUda5W5Cp8bo3P5GQ")

# models = genai.list_models()
# for model in models:
#     print(f"{model.name} | {model.supported_generation_methods}")
