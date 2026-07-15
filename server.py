import os
import sys
import json
import threading
import io
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import torch
from src.extractor import KnowledgeExtractor
from src.builder import KGBuilder, KGSerializer
from src.retriever import KGRetrievalModule
from src.generator import DatasetGenerator
from src.student import StructuredSupervisionLoss, StudentLLMInference
from src.evaluate import EvaluationFramework
from src.vectordb import FAISSVectorDB

app = Flask(__name__, static_folder='static')
CORS(app)

vector_db = FAISSVectorDB(embed_dim=128, db_path="vector_store")

extractor = None
if os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_API_KEY") != "your_groq_api_key_here":
    extractor = KnowledgeExtractor()

student = None
current_graph = None
pipeline_logs = []


def log(msg):
    pipeline_logs.append(msg)
    print(msg)


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = vector_db.get_stats()
    past_queries = vector_db.get_all_past_queries()
    stats['past_query_count'] = len(past_queries)
    stats['has_extractor'] = extractor is not None
    stats['student_trained'] = student is not None and student._trained
    return jsonify(stats)


@app.route('/api/history', methods=['GET'])
def get_history():
    past = vector_db.get_all_past_queries()
    past.reverse()
    return jsonify(past[:50])


@app.route('/api/pipeline', methods=['POST'])
def run_pipeline():
    global student, current_graph
    data = request.json
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    result = {
        'question': question,
        'steps': [],
        'graph_data': None,
        'teacher_answer': '',
        'student_answer': '',
        'metrics': {},
        'similar_past': [],
        'vectordb_stats': {},
    }

    result['steps'].append({'name': 'VectorDB Search', 'status': 'running'})
    past_results = vector_db.search_past_queries(question, top_k=3)
    result['similar_past'] = past_results
    result['steps'][-1]['status'] = 'done'
    result['steps'][-1]['detail'] = f"Found {len(past_results)} similar past queries"

    result['steps'].append({'name': 'Information Retrieval', 'status': 'running'})
    if extractor:
        document_text = extractor.get_information(question)
    else:
        document_text = f"Information about: {question}. This is placeholder text because GROQ_API_KEY is not configured."
    result['steps'][-1]['status'] = 'done'
    result['steps'][-1]['detail'] = f"Retrieved {len(document_text)} chars"

    result['steps'].append({'name': 'Entity Extraction', 'status': 'running'})
    if extractor:
        extraction_data = extractor.extract(document_text)
    else:
        extraction_data = {
            "entities": [
                {"name": "Apple Inc.", "type": "Organization"},
                {"name": "Steve Jobs", "type": "Person"},
                {"name": "Cupertino, California", "type": "Location"}
            ],
            "relations": [
                {"source": "Apple Inc.", "target": "Steve Jobs", "type": "founded by", "confidence": 0.99},
                {"source": "Apple Inc.", "target": "Cupertino, California", "type": "headquartered in", "confidence": 0.95}
            ]
        }
    result['steps'][-1]['status'] = 'done'
    entities = extraction_data.get('entities', [])
    relations = extraction_data.get('relations', [])
    result['steps'][-1]['detail'] = f"{len(entities)} entities, {len(relations)} relations"

    result['steps'].append({'name': 'Knowledge Graph Construction', 'status': 'running'})
    kg_builder = KGBuilder()
    graph = kg_builder.build_from_extraction(extraction_data)
    current_graph = graph

    nodes = []
    for n, attr in graph.nodes(data=True):
        nodes.append({'id': str(n), 'label': str(n), 'type': attr.get('type', 'Unknown')})
    edges = []
    for u, v, attr in graph.edges(data=True):
        edges.append({
            'from': str(u), 'to': str(v),
            'label': attr.get('type', ''),
            'confidence': float(attr.get('confidence', 1.0))
        })
    result['graph_data'] = {'nodes': nodes, 'edges': edges}
    result['steps'][-1]['status'] = 'done'
    result['steps'][-1]['detail'] = f"{len(nodes)} nodes, {len(edges)} edges"

    try:
        KGSerializer.to_graphml(graph, "kg_output.graphml")
        KGSerializer.to_image(graph, title=f"Knowledge Graph: {question}", filepath="kg_output.png")
    except Exception:
        pass

    result['steps'].append({'name': 'FAISS Indexing', 'status': 'running'})
    kg_facts_added = vector_db.add_kg_facts(graph)
    result['steps'][-1]['status'] = 'done'
    result['steps'][-1]['detail'] = f"Indexed {kg_facts_added} KG facts"

    result['steps'].append({'name': 'Context Retrieval (Graph + FAISS)', 'status': 'running'})
    retriever = KGRetrievalModule(graph)
    top_entities = []
    if isinstance(entities, list):
        for ent in entities:
            if isinstance(ent, dict) and 'name' in ent:
                top_entities.append(ent['name'])
            if len(top_entities) >= 2:
                break
    kg_context = retriever.generate_improved_answer_context(question, top_entities)
    result['steps'][-1]['status'] = 'done'

    result['steps'].append({'name': 'Teacher LLM Answer', 'status': 'running'})
    full_context = f"{document_text}\n\n{kg_context}"
    if extractor:
        teacher_answer = extractor.generate_teacher_answer(question, full_context)
    else:
        teacher_answer = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne. It is headquartered in Cupertino, California. (Mock teacher answer)"
    result['teacher_answer'] = teacher_answer
    result['steps'][-1]['status'] = 'done'

    result['steps'].append({'name': 'Student Transformer (Tokenize → Embed → Attend → Generate)', 'status': 'running'})
    student = StudentLLMInference(
        embed_dim=128, num_heads=4, num_layers=2,
        ff_dim=256, max_seq_len=256, lr=1e-3, epochs=100
    )
    student.set_context(graph, teacher_answer)
    student_response = student.generate(question)
    result['student_answer'] = student_response
    result['steps'][-1]['status'] = 'done'

    model_info = {}
    if student.model:
        total_params = sum(p.numel() for p in student.model.parameters())
        model_info = {
            'total_params': total_params,
            'vocab_size': student.tokenizer.vocab_size,
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 2,
        }
    result['model_info'] = model_info

    result['steps'].append({'name': 'Evaluation', 'status': 'running'})
    eval_framework = EvaluationFramework()
    mock_gt = [{"source": entities[0]['name'] if entities else "X", "target": entities[1]['name'] if len(entities) > 1 else "Y", "type": relations[0]['type'] if relations else "related"}]
    extractor_metrics = eval_framework.evaluate_extractor(relations, mock_gt)
    student_metrics = eval_framework.evaluate_student([student_response], [teacher_answer])
    result['metrics'] = {
        'extractor': extractor_metrics,
        'student_similarity': round(student_metrics, 4)
    }
    result['steps'][-1]['status'] = 'done'

    result['steps'].append({'name': 'Storing in VectorDB', 'status': 'running'})
    kg_facts_str = "; ".join([f"{u} → {d.get('type', '')} → {v}" for u, v, d in graph.edges(data=True)])
    entry_id = vector_db.add_entry(
        query=question, answer=student_response,
        teacher_answer=teacher_answer, kg_facts=kg_facts_str,
        entry_type="query"
    )
    result['vectordb_stats'] = vector_db.get_stats()
    result['steps'][-1]['status'] = 'done'
    result['steps'][-1]['detail'] = f"Stored as entry #{entry_id}"

    return jsonify(result)


@app.route('/api/chat', methods=['POST'])
def chat():
    global student
    data = request.json
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    if student is None or not student._trained:
        return jsonify({'error': 'Student LLM not trained yet. Run the pipeline first.'}), 400

    similar = vector_db.search_past_queries(question, top_k=3)
    answer = student.generate(question, max_kg_facts=5)

    kg_facts_str = ""
    if current_graph:
        kg_facts_str = "; ".join([f"{u} → {d.get('type', '')} → {v}" for u, v, d in current_graph.edges(data=True)])

    vector_db.add_entry(
        query=question, answer=answer,
        teacher_answer="", kg_facts=kg_facts_str,
        entry_type="query"
    )

    return jsonify({
        'answer': answer,
        'similar_past': similar,
        'stats': vector_db.get_stats()
    })


@app.route('/api/search', methods=['POST'])
def search_vectordb():
    data = request.json
    query = data.get('query', '')
    top_k = data.get('top_k', 5)
    results = vector_db.search(query, top_k=top_k)
    return jsonify(results)


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    if os.environ.get("GROQ_API_KEY") and os.environ.get("GROQ_API_KEY") != "your_groq_api_key_here":
        extractor = KnowledgeExtractor()
        print("✓ Groq Teacher LLM connected")
    else:
        print("⚠ GROQ_API_KEY not set. Using mock data.")

    print("\n" + "="*50)
    print("  AccuLLM Server Starting...")
    print("  Open http://localhost:5000 in your browser")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
