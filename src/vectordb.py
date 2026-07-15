import faiss
import numpy as np
import json
import os
import re
import math
from collections import Counter


class TextVectorizer:
    """
    Converts text into dense vectors using TF-IDF weighting.
    Builds a vocabulary from all stored documents, then produces
    fixed-dimension vectors via hashing + dimensionality reduction.
    """

    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim
        self.vocab = {}
        self.idf = {}
        self.doc_count = 0
        self._built = False

    def _tokenize(self, text):
        text = text.lower().strip()
        return re.findall(r"[a-zA-Z]+|[0-9]+", text)

    def fit(self, documents):
        doc_freq = Counter()
        self.doc_count = len(documents)

        all_tokens = set()
        for doc in documents:
            tokens = set(self._tokenize(doc))
            doc_freq.update(tokens)
            all_tokens.update(tokens)

        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        self.idf = {}
        for token, freq in doc_freq.items():
            self.idf[token] = math.log((self.doc_count + 1) / (freq + 1)) + 1

        self._built = True

    def partial_fit(self, new_documents):
        if not self._built:
            self.fit(new_documents)
            return

        doc_freq = Counter()
        self.doc_count += len(new_documents)

        for doc in new_documents:
            tokens = set(self._tokenize(doc))
            doc_freq.update(tokens)
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = len(self.vocab)

        for token, freq in doc_freq.items():
            old_df = 0
            if token in self.idf:
                old_df = (self.doc_count - len(new_documents) + 1) / (math.exp(self.idf[token] - 1) + 1e-10)
            new_df = old_df + freq
            self.idf[token] = math.log((self.doc_count + 1) / (new_df + 1)) + 1

    def transform(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.embed_dim, dtype=np.float32)

        tf = Counter(tokens)
        total = len(tokens)

        raw_vector = np.zeros(self.embed_dim, dtype=np.float32)

        for token, count in tf.items():
            tf_val = count / total
            idf_val = self.idf.get(token, 1.0)
            weight = tf_val * idf_val

            hash_idx = hash(token) % self.embed_dim
            sign = 1 if hash(token + "_sign") % 2 == 0 else -1
            raw_vector[hash_idx] += sign * weight

            hash_idx2 = hash(token + "_2") % self.embed_dim
            raw_vector[hash_idx2] += weight * 0.5

        norm = np.linalg.norm(raw_vector)
        if norm > 0:
            raw_vector = raw_vector / norm

        return raw_vector.astype(np.float32)


class FAISSVectorDB:
    """
    FAISS-backed Vector Database for storing and retrieving past queries,
    answers, KG facts, and any text data.
    
    Features:
        - Fast cosine similarity search using FAISS IndexFlatIP
        - TF-IDF text vectorization (no external model dependencies)
        - Persistent storage: saves index + metadata to disk
        - Incremental updates: add new entries without rebuilding
        - Tracks query history, answers, timestamps, and KG context
    """

    def __init__(self, embed_dim=128, db_path="vector_store"):
        self.embed_dim = embed_dim
        self.db_path = db_path
        self.vectorizer = TextVectorizer(embed_dim=embed_dim)

        self.index = faiss.IndexFlatIP(embed_dim)

        self.metadata = []

        self._load_from_disk()

    def _ensure_dir(self):
        os.makedirs(self.db_path, exist_ok=True)

    def _save_to_disk(self):
        self._ensure_dir()

        faiss.write_index(self.index, os.path.join(self.db_path, "faiss_index.bin"))

        with open(os.path.join(self.db_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        vectorizer_state = {
            "vocab": self.vectorizer.vocab,
            "idf": self.vectorizer.idf,
            "doc_count": self.vectorizer.doc_count,
            "embed_dim": self.vectorizer.embed_dim,
        }
        with open(os.path.join(self.db_path, "vectorizer.json"), "w", encoding="utf-8") as f:
            json.dump(vectorizer_state, f, indent=2)

    def _load_from_disk(self):
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        meta_path = os.path.join(self.db_path, "metadata.json")
        vec_path = os.path.join(self.db_path, "vectorizer.json")

        if os.path.exists(index_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(index_path)

                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

                if os.path.exists(vec_path):
                    with open(vec_path, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    self.vectorizer.vocab = state["vocab"]
                    self.vectorizer.idf = state["idf"]
                    self.vectorizer.doc_count = state["doc_count"]
                    self.vectorizer.embed_dim = state["embed_dim"]
                    self.vectorizer._built = True

                print(f"    [VectorDB] Loaded {len(self.metadata)} entries from disk ({self.db_path}/)")
            except Exception as e:
                print(f"    [VectorDB] Warning: Could not load from disk: {e}. Starting fresh.")
                self.index = faiss.IndexFlatIP(self.embed_dim)
                self.metadata = []

    def _build_combined_text(self, entry):
        parts = []
        if entry.get("query"):
            parts.append(entry["query"])
        if entry.get("answer"):
            parts.append(entry["answer"])
        if entry.get("kg_facts"):
            parts.append(entry["kg_facts"])
        if entry.get("teacher_answer"):
            parts.append(entry["teacher_answer"])
        return " ".join(parts)

    def add_entry(self, query, answer="", teacher_answer="", kg_facts="",
                  entry_type="query", extra_metadata=None):
        import datetime

        entry = {
            "id": len(self.metadata),
            "query": query,
            "answer": answer,
            "teacher_answer": teacher_answer,
            "kg_facts": kg_facts,
            "type": entry_type,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if extra_metadata:
            entry.update(extra_metadata)

        combined_text = self._build_combined_text(entry)

        self.vectorizer.partial_fit([combined_text])

        vector = self.vectorizer.transform(combined_text)
        vector = vector.reshape(1, -1)

        self.index.add(vector)
        self.metadata.append(entry)

        self._save_to_disk()

        return entry["id"]

    def add_kg_facts(self, graph):
        facts_added = 0
        for u, v, data in graph.edges(data=True):
            rel = data.get('type', 'related to')
            conf = data.get('confidence', 1.0)
            fact_text = f"{u} {rel} {v}"

            self.add_entry(
                query=fact_text,
                answer="",
                kg_facts=fact_text,
                entry_type="kg_fact",
                extra_metadata={"source": str(u), "target": str(v),
                                "relation": rel, "confidence": conf}
            )
            facts_added += 1

        return facts_added

    def search(self, query_text, top_k=5, entry_type_filter=None):
        if self.index.ntotal == 0:
            return []

        if not self.vectorizer._built:
            return []

        query_vector = self.vectorizer.transform(query_text).reshape(1, -1)

        k = min(top_k * 3 if entry_type_filter else top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            entry = self.metadata[idx].copy()
            entry["similarity_score"] = float(dist)

            if entry_type_filter and entry.get("type") != entry_type_filter:
                continue

            results.append(entry)

            if len(results) >= top_k:
                break

        return results

    def search_past_queries(self, query_text, top_k=5):
        return self.search(query_text, top_k=top_k, entry_type_filter="query")

    def search_kg_facts(self, query_text, top_k=5):
        return self.search(query_text, top_k=top_k, entry_type_filter="kg_fact")

    def get_all_past_queries(self):
        return [m for m in self.metadata if m.get("type") == "query"]

    def get_stats(self):
        type_counts = Counter(m.get("type", "unknown") for m in self.metadata)
        return {
            "total_entries": len(self.metadata),
            "index_size": self.index.ntotal,
            "vocab_size": len(self.vectorizer.vocab),
            "embed_dim": self.embed_dim,
            "entry_types": dict(type_counts),
            "db_path": self.db_path,
        }

    def format_search_results(self, results, max_display=5):
        if not results:
            return "  No relevant past entries found."

        lines = []
        for i, r in enumerate(results[:max_display]):
            score = r.get("similarity_score", 0.0)
            entry_type = r.get("type", "unknown")
            timestamp = r.get("timestamp", "N/A")

            if entry_type == "query":
                lines.append(f"  [{i+1}] (score: {score:.3f}) Q: \"{r.get('query', '')}\"")
                if r.get("answer"):
                    ans_preview = r["answer"][:100] + "..." if len(r.get("answer", "")) > 100 else r.get("answer", "")
                    lines.append(f"      A: {ans_preview}")
                lines.append(f"      Time: {timestamp}")
            elif entry_type == "kg_fact":
                lines.append(f"  [{i+1}] (score: {score:.3f}) Fact: {r.get('query', '')}")
                conf = r.get("confidence", "N/A")
                lines.append(f"      Confidence: {conf}")
            else:
                lines.append(f"  [{i+1}] (score: {score:.3f}) [{entry_type}] {r.get('query', '')}")

        return "\n".join(lines)
