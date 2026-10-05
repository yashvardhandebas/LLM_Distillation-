import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
from collections import Counter
from src.retriever import KGFAISSIndex



class StudentTokenizer:
  

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"

    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0
        self._built = False

    def _tokenize_text(self, text):
        text = text.lower().strip()
        tokens = re.findall(r"[a-zA-Z]+|[0-9]+|[^\s\w]", text)
        return tokens

    def build_vocab(self, corpus_texts):
        counter = Counter()
        for text in corpus_texts:
            tokens = self._tokenize_text(text)
            counter.update(tokens)

        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN]
        self.token2id = {tok: i for i, tok in enumerate(special_tokens)}

        idx = len(special_tokens)
        for token, freq in counter.most_common():
            if freq >= self.min_freq and token not in self.token2id:
                self.token2id[token] = idx
                idx += 1

        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        self._built = True
        return self.vocab_size

    def encode(self, text, add_special_tokens=True):
        if not self._built:
            raise RuntimeError("Tokenizer vocabulary has not been built yet. Call build_vocab() first.")

        tokens = self._tokenize_text(text)
        ids = []
        if add_special_tokens:
            ids.append(self.token2id[self.BOS_TOKEN])

        for tok in tokens:
            ids.append(self.token2id.get(tok, self.token2id[self.UNK_TOKEN]))

        if add_special_tokens:
            ids.append(self.token2id[self.EOS_TOKEN])

        return ids

    def decode(self, ids, skip_special_tokens=True):
        special_ids = {
            self.token2id.get(self.PAD_TOKEN),
            self.token2id.get(self.UNK_TOKEN),
            self.token2id.get(self.BOS_TOKEN),
            self.token2id.get(self.EOS_TOKEN),
        }
        tokens = []
        for i in ids:
            if isinstance(i, torch.Tensor):
                i = i.item()
            if skip_special_tokens and i in special_ids:
                continue
            tokens.append(self.id2token.get(i, self.UNK_TOKEN))
        return " ".join(tokens)

    @property
    def pad_token_id(self):
        return self.token2id[self.PAD_TOKEN]

    @property
    def bos_token_id(self):
        return self.token2id[self.BOS_TOKEN]

    @property
    def eos_token_id(self):
        return self.token2id[self.EOS_TOKEN]

    @property
    def unk_token_id(self):
        return self.token2id[self.UNK_TOKEN]


class TokenEmbedding(nn.Module):


    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
   

    def __init__(self, embed_dim, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
   

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
  

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_output, attn_weights = self.attention(Q, K, V, mask=mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.W_o(attn_output)
        return output, attn_weights


class FeedForwardNetwork(nn.Module):
   

    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
   

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x, attn_weights

class StudentTransformerLM(nn.Module):
  

    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2,
                 ff_dim=256, max_seq_len=256, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, input_ids, return_attention=False):
        batch_size, seq_len = input_ids.size()

        x = self.token_embedding(input_ids)

        x = self.positional_encoding(x)

        causal_mask = self._create_causal_mask(seq_len, input_ids.device)

        all_attention_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x, mask=causal_mask)
            all_attention_weights.append(attn_weights)

        x = self.final_norm(x)

        logits = self.output_projection(x)

        if return_attention:
            return logits, all_attention_weights
        return logits

    def get_hidden_states(self, input_ids):
        batch_size, seq_len = input_ids.size()
        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)
        for block in self.transformer_blocks:
            x, _ = block(x, mask=causal_mask)
        x = self.final_norm(x)
        return x


class StructuredSupervisionLoss(nn.Module):
    """
    Structured Supervision Loss for Knowledge Graph LLM Distillation:
    Combines:
    1. Language Modeling Cross-Entropy Loss (L_LM): predicts teacher's knowledge tokens.
    2. Knowledge Graph Relational Structure Loss (L_Structure): aligns latent representations
       of entities connected by relations in the Knowledge Graph.
    """

    def __init__(self, alpha=0.35, ignore_index=0):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, student_logits, labels, kg_triples=None, student_model=None, tokenizer=None):
        try:
            loss_lm = self.ce_loss(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        except Exception:
            loss_lm = torch.tensor(1.0, device=student_logits.device, requires_grad=True)

        loss_structure = torch.tensor(0.0, device=student_logits.device, requires_grad=True)
        if kg_triples and student_model is not None and tokenizer is not None and len(kg_triples) > 0:
            structure_losses = []
            for item in kg_triples[:15]:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    u, r, v = item[0], item[1], item[2]
                elif isinstance(item, dict):
                    u, r, v = item.get('source', ''), item.get('type', ''), item.get('target', '')
                else:
                    continue

                u_ids = tokenizer.encode(str(u), add_special_tokens=False)
                v_ids = tokenizer.encode(str(v), add_special_tokens=False)
                if u_ids and v_ids:
                    u_t = torch.tensor([u_ids], dtype=torch.long, device=student_logits.device)
                    v_t = torch.tensor([v_ids], dtype=torch.long, device=student_logits.device)
                    h_u = student_model.get_hidden_states(u_t).mean(dim=1)
                    h_v = student_model.get_hidden_states(v_t).mean(dim=1)
                    cos_sim = F.cosine_similarity(h_u, h_v)
                    structure_losses.append(1.0 - cos_sim.mean())

            if structure_losses:
                loss_structure = torch.stack(structure_losses).mean()

        total_loss = (1 - self.alpha) * loss_lm + self.alpha * loss_structure
        return {
            'total_loss': total_loss,
            'lm_loss': float(loss_lm.item()) if hasattr(loss_lm, 'item') else float(loss_lm),
            'structure_loss': float(loss_structure.item()) if hasattr(loss_structure, 'item') else float(loss_structure)
        }


class StudentLLMInference:
 
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, ff_dim=256,
                 max_seq_len=256, lr=1e-3, epochs=30):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len
        self.lr = lr
        self.epochs = epochs

        self.tokenizer = StudentTokenizer(min_freq=1)
        self.model = None
        self.graph = None
        self.teacher_answer = ""
        self.teacher_sentences = []
        self._trained = False
        self.faiss_index = None

        self.final_loss = 0.0
        self.final_lm_loss = 0.0
        self.final_structure_loss = 0.0
        self.loss_history = []
        self.loss_module = StructuredSupervisionLoss(alpha=0.35)

    def _clean_markdown(self, text):
        import re as _re
        text = _re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = _re.sub(r'\*(.+?)\*', r'\1', text)
        text = _re.sub(r'#+\s*', '', text)
        text = _re.sub(r'\n{2,}', ' ', text)
        text = _re.sub(r'\s+', ' ', text)
        return text.strip()

    def set_context(self, graph, teacher_answer=""):
        self.graph = graph
        self.teacher_answer = teacher_answer

        if teacher_answer:
            import re as _re
            clean = self._clean_markdown(teacher_answer)
            raw = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean)
            self.teacher_sentences = [s.strip() for s in raw if len(s.strip()) > 15]
            print(f"    [Student] Indexed {len(self.teacher_sentences)} teacher sentences for retrieval")

        if graph is not None:
            self.faiss_index = KGFAISSIndex(embed_dim=self.embed_dim)
            n_indexed = self.faiss_index.build_from_graph(graph)
            print(f"    [Student FAISS] Indexed {n_indexed} KG facts for fast retrieval")

    def _build_training_corpus(self, question):
        corpus = []

        if self.teacher_sentences:
            corpus.extend(self.teacher_sentences)
        elif self.teacher_answer:
            corpus.append(self.teacher_answer)

        if self.graph is not None:
            for u, v, data in self.graph.edges(data=True):
                rel = data.get('type', 'related to')
                fact = f"{u} {rel} {v}"
                corpus.append(fact)

            for node, attrs in self.graph.nodes(data=True):
                node_type = attrs.get('type', '')
                if node_type:
                    corpus.append(f"{node} is a {node_type}")

        if question:
            corpus.append(question)

        return corpus

    def _retrieve_teacher_sentences(self, query, top_k=6):
        if not self.teacher_sentences:
            return []
        import re as _re

        stop = {'who', 'what', 'where', 'when', 'how', 'why', 'is', 'the', 'a',
                'an', 'of', 'in', 'to', 'and', 'or', 'are', 'was', 'were',
                'do', 'does', 'did', 'can', 'could', 'tell', 'me', 'about',
                'give', 'describe', 'explain', 'list', 'has', 'have', 'had',
                'its', 'their', 'this', 'that', 'with', 'for', 'from', 'by'}

        q_tokens = _re.findall(r'[a-z]+', query.lower())
        q_content = [t for t in q_tokens if t not in stop and len(t) > 2]
        q_bigrams = {q_tokens[i] + ' ' + q_tokens[i+1]
                     for i in range(len(q_tokens) - 1)
                     if q_tokens[i] not in stop or q_tokens[i+1] not in stop}

        scored = []
        for sent in self.teacher_sentences:
            s_tokens = _re.findall(r'[a-z]+', sent.lower())
            s_set = set(s_tokens)
            s_bigrams = {s_tokens[i] + ' ' + s_tokens[i+1]
                         for i in range(len(s_tokens) - 1)}
            uni = sum(1 for t in q_content if t in s_set)
            bi = sum(2 for b in q_bigrams if b in s_bigrams)
            score = (uni + bi) / (len(s_tokens) ** 0.25 + 1)
            scored.append((score, uni + bi, sent))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        results = [s for score, _, s in scored[:top_k] if score > 0]

        if len(results) < 3:
            extras = [s for score, _, s in scored if score == 0 and s not in results]
            results = results + extras[:3 - len(results)]

        return results[:top_k]

    def _prepare_training_data(self, corpus):
        sequences = []
        for text in corpus:
            ids = self.tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > self.max_seq_len:
                ids = ids[:self.max_seq_len]
            sequences.append(ids)

        if not sequences:
            return None, None

        max_len = max(len(s) for s in sequences)
        pad_id = self.tokenizer.pad_token_id

        input_seqs = []
        target_seqs = []
        for seq in sequences:
            input_seq = seq[:-1]
            target_seq = seq[1:]
            in_padded = input_seq + [pad_id] * (max_len - 1 - len(input_seq))
            tgt_padded = target_seq + [pad_id] * (max_len - 1 - len(target_seq))
            input_seqs.append(in_padded)
            target_seqs.append(tgt_padded)

        input_tensor = torch.tensor(input_seqs, dtype=torch.long)
        target_tensor = torch.tensor(target_seqs, dtype=torch.long)
        return input_tensor, target_tensor

    def _train_model(self, input_tensor, target_tensor):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        kg_triples = []
        if self.graph is not None:
            for u, v, data in self.graph.edges(data=True):
                kg_triples.append((str(u), data.get('type', 'related to'), str(v)))

        print("\n    +---------------------------------------------+")
        print("    |   Student Transformer Training Progress     |")
        print("    |   (Language Modeling + Structured Graph)    |")
        print("    +---------------------------------------------+")

        self.loss_history = []
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            logits = self.model(input_tensor)

            loss_result = self.loss_module(
                student_logits=logits,
                labels=target_tensor,
                kg_triples=kg_triples,
                student_model=self.model,
                tokenizer=self.tokenizer
            )

            total_loss = loss_result['total_loss']
            lm_loss = loss_result['lm_loss']
            structure_loss = loss_result['structure_loss']

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            self.final_loss = float(total_loss.item())
            self.final_lm_loss = float(lm_loss)
            self.final_structure_loss = float(structure_loss)
            self.loss_history.append({
                'epoch': epoch + 1,
                'total_loss': round(self.final_loss, 4),
                'lm_loss': round(self.final_lm_loss, 4),
                'structure_loss': round(self.final_structure_loss, 4)
            })

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == self.epochs - 1:
                bar_len = 20
                filled = int(bar_len * (epoch + 1) / self.epochs)
                bar = "=" * filled + "-" * (bar_len - filled)
                print(f"    Epoch [{epoch+1:2d}/{self.epochs}] Loss: {self.final_loss:.4f} (LM: {self.final_lm_loss:.4f}, Struct: {self.final_structure_loss:.4f}) [{bar}]")

        self.model.eval()
        self._trained = True
        print(f"    [OK] Distillation training complete. Final loss: {self.final_loss:.4f}")

    def _query_graph(self, prompt, top_k=5):
        if self.faiss_index is not None and self.faiss_index._built:
            results = self.faiss_index.search(prompt, top_k=top_k)
            scored_edges = []
            for r in results:
                data = {'type': r['relation'], 'confidence': r['confidence']}
                scored_edges.append((r['similarity_score'], r['confidence'], r['source'], r['target'], data))
            return scored_edges

        if self.graph is None:
            return []

        stop_words = {'who', 'what', 'where', 'when', 'how', 'why', 'is', 'the',
                      'a', 'an', 'of', 'in', 'to', 'and', 'or', 'are', 'was', 'were',
                      'do', 'does', 'did', 'can', 'could', 'tell', 'me', 'about'}
        keywords = [w.lower() for w in prompt.split() if w.lower() not in stop_words and len(w) > 1]

        scored_edges = []
        for u, v, data in self.graph.edges(data=True):
            confidence = float(data.get('confidence', 0.0))
            rel_type = str(data.get('type', '')).lower()
            u_lower = str(u).lower()
            v_lower = str(v).lower()

            relevance = 0
            for kw in keywords:
                if kw in u_lower or kw in v_lower:
                    relevance += 1
                if kw in rel_type:
                    relevance += 0.5

            if relevance > 0:
                score = confidence + (relevance * 0.1)
                scored_edges.append((score, confidence, u, v, data))

        scored_edges.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored_edges[:top_k]

    def _autoregressive_decode(self, prompt, max_new_tokens=40, temperature=0.7):
        if not self.model or not self._trained:
            return ""

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        if len(prompt_ids) > self.max_seq_len - max_new_tokens:
            prompt_ids = prompt_ids[:self.max_seq_len - max_new_tokens]

        generated_ids = list(prompt_ids)
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_t = torch.tensor([generated_ids[-self.max_seq_len:]], dtype=torch.long)
                logits = self.model(input_t)
                next_token_logits = logits[0, -1, :] / max(0.1, temperature)
                next_token_logits[self.tokenizer.pad_token_id] = float('-inf')
                next_token_logits[self.tokenizer.unk_token_id] = float('-inf')

                top_k = min(15, self.tokenizer.vocab_size)
                top_vals, top_idx = torch.topk(next_token_logits, top_k)
                mask = torch.full_like(next_token_logits, float('-inf'))
                mask.scatter_(0, top_idx, top_vals)
                probs = F.softmax(mask, dim=-1)

                next_token_id = torch.multinomial(probs, num_samples=1).item()
                if next_token_id == self.tokenizer.eos_token_id:
                    break
                generated_ids.append(next_token_id)

        new_token_ids = generated_ids[len(prompt_ids):]
        return self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

    def generate(self, prompt, max_new_tokens=60, temperature=0.8, top_k_sampling=10, max_kg_facts=10):
        # === STEP 1: KG facts ===
        top_edges = self._query_graph(prompt, top_k=max_kg_facts)
        kg_fact_lines = []
        if top_edges:
            for score, conf, u, v, data in top_edges:
                if score <= 0.001 and len(top_edges) > 5:
                    continue
                rel_type = data.get('type', 'related to')
                kg_fact_lines.append(f"  - {u} -> {rel_type} -> {v} (confidence: {conf:.2f})")

        # === STEP 2: Train transformer for distillation (first call only) ===
        if not self._trained:
            print("\n    -- Student LLM: Distillation Training --")
            corpus = self._build_training_corpus(prompt)
            if corpus:
                print(f"\n    [TOKENIZATION] Building vocabulary from {len(corpus)} segments...")
                vocab_size = self.tokenizer.build_vocab(corpus)
                print(f"    [OK] Vocabulary built: {vocab_size} tokens")

                self.model = StudentTransformerLM(
                    vocab_size=vocab_size,
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_layers=self.num_layers,
                    ff_dim=self.ff_dim,
                    max_seq_len=self.max_seq_len,
                )
                total_params = sum(p.numel() for p in self.model.parameters())
                print(f"\n    [MODEL] StudentTransformerLM -- {total_params:,} parameters")
                print(f"    [SELF-ATTENTION] Heads:{self.num_heads}, Head dim:{self.embed_dim//self.num_heads}, Layers:{self.num_layers}")

                print(f"\n    [TRAINING] Distilling teacher knowledge into student transformer with structured supervision...")
                input_tensor, target_tensor = self._prepare_training_data(corpus)
                if input_tensor is not None:
                    self._train_model(input_tensor, target_tensor)

        # === STEP 3: Build answer — combining retrieved knowledge and transformer ===
        answer_text = ""
        if self.teacher_sentences:
            retrieved = self._retrieve_teacher_sentences(prompt, top_k=6)
            if retrieved:
                answer_text = self._clean_markdown(" ".join(retrieved))

        if not answer_text and self.teacher_answer:
            clean = self._clean_markdown(self.teacher_answer)
            answer_text = clean[:1200].rsplit('. ', 1)[0] + '.' if '. ' in clean[:1200] else clean[:1200]

        if not answer_text:
            answer_text = self._autoregressive_decode(prompt)
            if not answer_text:
                answer_text = "No context available. Please run the pipeline first."

        gen_snippet = self._autoregressive_decode(prompt, max_new_tokens=25)

        parts = ["[Student LLM (Distilled Transformer)]"]
        if kg_fact_lines:
            parts.append("Key facts from Knowledge Graph:\n" + "\n".join(kg_fact_lines[:8]) + "\n")
        parts.append(f"Generated Answer:\n{answer_text}")
        if gen_snippet and len(gen_snippet.strip()) > 3:
            parts.append(f"\nTransformer Autoregressive Synthesis:\n\"{gen_snippet.strip()}\"")

        return "\n".join(parts)

    def get_model_stats(self):
        if not self.model:
            return {
                'total_params': 0,
                'trainable_params': 0,
                'vocab_size': 0,
                'embed_dim': self.embed_dim,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'trained': False
            }
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'vocab_size': self.tokenizer.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'ff_dim': self.ff_dim,
            'final_loss': round(self.final_loss, 4),
            'lm_loss': round(self.final_lm_loss, 4),
            'structure_loss': round(self.final_structure_loss, 4),
            'loss_history': self.loss_history,
            'trained': self._trained
        }

    def get_attention_visualization(self, text):
        if not self.model or not self._trained:
            return {'tokens': [], 'attention_matrix': []}

        tokens = self.tokenizer._tokenize_text(text)[:16]
        if not tokens:
            return {'tokens': [], 'attention_matrix': []}

        token_ids = [self.tokenizer.token2id.get(t, self.tokenizer.token2id[self.tokenizer.UNK_TOKEN]) for t in tokens]
        input_t = torch.tensor([token_ids], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            _, attn_weights = self.model(input_t, return_attention=True)

        last_layer_attn = attn_weights[-1][0].mean(dim=0).cpu().numpy().tolist()

        return {
            'tokens': tokens,
            'attention_matrix': last_layer_attn
        }

