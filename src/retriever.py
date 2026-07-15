import networkx as nx
import numpy as np
import faiss
import re
import math
from collections import Counter


class KGFAISSIndex:
   

    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim
        self.index = None
        self.edge_data = []
        self.vocab = {}
        self.idf = {}
        self._built = False

    def _tokenize(self, text):
        return re.findall(r"[a-zA-Z]+|[0-9]+", text.lower().strip())

    def _build_tfidf(self, documents):
        doc_freq = Counter()
        all_tokens = set()
        for doc in documents:
            tokens = set(self._tokenize(doc))
            doc_freq.update(tokens)
            all_tokens.update(tokens)

        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}
        n_docs = len(documents)
        self.idf = {t: math.log((n_docs + 1) / (f + 1)) + 1 for t, f in doc_freq.items()}

    def _vectorize(self, text):
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.embed_dim, dtype=np.float32)

        tf = Counter(tokens)
        total = len(tokens)
        vec = np.zeros(self.embed_dim, dtype=np.float32)

        for token, count in tf.items():
            weight = (count / total) * self.idf.get(token, 1.0)
            h1 = hash(token) % self.embed_dim
            sign = 1 if hash(token + "_s") % 2 == 0 else -1
            vec[h1] += sign * weight
            h2 = hash(token + "_2") % self.embed_dim
            vec[h2] += weight * 0.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.astype(np.float32)

    def build_from_graph(self, graph):
        self.edge_data = []
        documents = []

        for u, v, data in graph.edges(data=True):
            rel = data.get('type', 'related to')
            conf = data.get('confidence', 1.0)
            text = f"{u} {rel} {v}"
            documents.append(text)
            self.edge_data.append({
                "source": str(u),
                "target": str(v),
                "relation": rel,
                "confidence": float(conf),
                "text": text
            })

        for node, attrs in graph.nodes(data=True):
            node_type = attrs.get('type', '')
            if node_type:
                text = f"{node} is a {node_type}"
                documents.append(text)
                self.edge_data.append({
                    "source": str(node),
                    "target": node_type,
                    "relation": "is_a",
                    "confidence": 1.0,
                    "text": text
                })

        if not documents:
            return 0

        self._build_tfidf(documents)

        vectors = np.array([self._vectorize(doc) for doc in documents], dtype=np.float32)

        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(vectors)

        self._built = True
        return len(documents)

    def search(self, query, top_k=5):
        if not self._built or self.index is None or self.index.ntotal == 0:
            return []

        query_vec = self._vectorize(query).reshape(1, -1)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.edge_data):
                entry = self.edge_data[idx].copy()
                entry["similarity_score"] = float(dist)
                results.append(entry)

        return results


class KGRetrievalModule:
   

    def __init__(self, graph):
        self.graph = graph
        self.faiss_index = KGFAISSIndex(embed_dim=128)

        n_indexed = self.faiss_index.build_from_graph(graph)
        print(f"    [FAISS] Indexed {n_indexed} KG facts for fast retrieval")

    def retrieve_subgraph(self, query_entities, hops=1):
        subgraph_nodes = set()
        for ent in query_entities:
            matching_nodes = [n for n in self.graph.nodes if str(ent).lower() in str(n).lower()]
            for matched_node in matching_nodes:
                subgraph_nodes.add(matched_node)
                if hops >= 1:
                    subgraph_nodes.update(self.graph.successors(matched_node))
                    subgraph_nodes.update(self.graph.predecessors(matched_node))

        return self.graph.subgraph(list(subgraph_nodes))

    def faiss_retrieve(self, query, top_k=5):
        return self.faiss_index.search(query, top_k=top_k)

    def generate_improved_answer_context(self, question, query_entities):
        subgraph = self.retrieve_subgraph(query_entities)
        context_lines: list[str] = []
        for u, v, data in subgraph.edges(data=True):
            context_lines.append(f"- {u} is related to {v} (Relation: {data.get('type')}, Confidence: {data.get('confidence', 1.0)})")

        faiss_results = self.faiss_retrieve(question, top_k=5)
        faiss_lines = []
        for r in faiss_results:
            fact = f"- [FAISS] {r['source']} → {r['relation']} → {r['target']} (Confidence: {r['confidence']}, Similarity: {r['similarity_score']:.3f})"
            faiss_lines.append(fact)

        if not context_lines and not faiss_lines:
            return "No relevant Knowledge Graph context found."

        result = ""
        if context_lines:
            result += "Knowledge Graph Context (Graph Traversal):\n" + "\n".join(context_lines)
        if faiss_lines:
            if result:
                result += "\n\n"
            result += "Knowledge Graph Context (FAISS Semantic Search):\n" + "\n".join(faiss_lines)

        return result
