import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

import torch
from src.extractor import KnowledgeExtractor
from src.builder import KGBuilder, KGSerializer
from src.retriever import KGRetrievalModule
from src.generator import DatasetGenerator
from src.student import StructuredSupervisionLoss, StudentLLMInference
from src.evaluate import EvaluationFramework
from src.vectordb import FAISSVectorDB

def main():
    print("="*60)
    print("Knowledge Graph Teacher-Student System Pipeline")
    print("="*60)
    
    print("\n[0] Initializing FAISS Vector Database...")
    vector_db = FAISSVectorDB(embed_dim=128, db_path="vector_store")
    db_stats = vector_db.get_stats()
    print(f"    VectorDB loaded: {db_stats['total_entries']} entries, "
          f"{db_stats['vocab_size']} vocab tokens")
    if db_stats['total_entries'] > 0:
        print(f"    Entry types: {db_stats['entry_types']}")
    
    print("\n[1] Initializing Teacher LLM (Groq) Extractor...")
    if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_groq_api_key_here":
        print("WARNING: GROQ_API_KEY not set properly in .env! Teacher extraction will fail. Please set it.")
        extractor = None
    else:
        extractor = KnowledgeExtractor()
        
    question = input("\nEnter a search query (e.g., 'Tell me about the history of Artificial Intelligence'): ")
    if not question.strip():
        question = "Who founded Apple Inc. and where is it headquartered?"

    print("\n[1.5] Searching VectorDB for similar past queries...")
    past_results = vector_db.search_past_queries(question, top_k=3)
    if past_results:
        print("    Found similar past queries:")
        print(vector_db.format_search_results(past_results))
    else:
        print("    No similar past queries found. This is a new topic.")

    print("\n[2] Fetching information from Groq based on the query...")
    if extractor:
        document_text = extractor.get_information(question)
        print("\n--- Gathered Text ---")
        print(document_text)
        print("---------------------\n")
    else:
        document_text = "Dummy text about Apple Inc."
        
    print("\n[3] Extracting Entities and Relations from document...")
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
    print("Extracted Data:", extraction_data)
    
    print("\n[4] Building and Serializing Knowledge Graph...")
    kg_builder = KGBuilder()
    graph = kg_builder.build_from_extraction(extraction_data)
    
    json_ld_repr = KGSerializer.to_json_ld(graph)
    graphml_path = KGSerializer.to_graphml(graph, "kg_output.graphml")
    print(f"Graph serialized to GraphML: {graphml_path}")
    
    image_path = KGSerializer.to_image(graph, title=f"Knowledge Graph: {question}", filepath="kg_output.png")
    print(f"Graph image saved to: {image_path}")
    
    print("Sample JSON-LD:")
    print(json_ld_repr)

    print("\n[4.5] Indexing KG facts into VectorDB...")
    kg_facts_added = vector_db.add_kg_facts(graph)
    print(f"    ✓ Indexed {kg_facts_added} KG facts into VectorDB")

    print("\n    Searching VectorDB for KG facts related to query...")
    kg_fact_results = vector_db.search_kg_facts(question, top_k=3)
    if kg_fact_results:
        print(vector_db.format_search_results(kg_fact_results))
    
    print("\n[5] Retrieving Context from KG (Graph Traversal + FAISS)...")
    retriever = KGRetrievalModule(graph)
    
    entities_list = extraction_data.get('entities')
    top_entities = []
    if isinstance(entities_list, list):
        for ent in entities_list:
            if isinstance(ent, dict) and 'name' in ent:
                top_entities.append(ent['name'])
            if len(top_entities) >= 2:
                break

    print(f"Querying graph for context around: {top_entities}")
    kg_context = retriever.generate_improved_answer_context(question, top_entities)
    print(kg_context)
    
    print("\n[6] Teacher generating high-quality answer...")
    full_context = f"{document_text}\n\n{kg_context}"
    if extractor:
        teacher_answer = extractor.generate_teacher_answer(question, full_context)
    else:
        teacher_answer = "Apple Inc. was founded by Steve Jobs. It is headquartered in Cupertino, California. (Mock teacher answer)"
    print("Teacher Answer:", teacher_answer)
    
    print("\n[7] Adding sample to Dataset and saving...")
    dataset_gen = DatasetGenerator()
    dataset_gen.add_sample(question, document_text, extraction_data, teacher_answer)
    dataset_gen.save_dataset("training_dataset.json")
    dataset_gen.save_csv("training_dataset.csv")
    print("Dataset saved to training_dataset.json and training_dataset.csv")
    
    print("\n[8] Initializing Student LLM with Real Transformer Architecture... (Work in Progress)")
    student_response = "(Student LLM inference will be available in Milestone 2)"
    print("\n[9] Running Student LLM Inference (full pipeline)... (Work in Progress)")

    print("\n[9.5] Storing query + answers in VectorDB...")
    kg_facts_str = ""
    if graph:
        facts = []
        for u, v, data in graph.edges(data=True):
            facts.append(f"{u} → {data.get('type', 'related to')} → {v}")
        kg_facts_str = "; ".join(facts)

    entry_id = vector_db.add_entry(
        query=question,
        answer=student_response,
        teacher_answer=teacher_answer,
        kg_facts=kg_facts_str,
        entry_type="query"
    )
    db_stats = vector_db.get_stats()
    print(f"    ✓ Stored as entry #{entry_id} in VectorDB")
    print(f"    VectorDB now has {db_stats['total_entries']} total entries")
    print(f"    Entry types: {db_stats['entry_types']}")
    
    print("\n[10] Computing Structured Supervision Loss... (Work in Progress)")
    
    print("\n[11] Running Evaluation Framework... (Work in Progress)")
    
    print("\n[12] Pipeline Complete.")
    print("    In a real scenario, metrics from [11] are used to update Student or refine Teacher Prompts.")
    
    print("\n" + "="*60)
    print("Interactive Q&A with Student LLM (FAISS-augmented)")
    print("Past queries are stored in VectorDB for retrieval.")
    print("Type 'history' to see past queries, 'stats' for DB stats.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*60)
    
    while True:
        user_q = input("\nYou: ").strip()
        if not user_q or user_q.lower() in ('exit', 'quit', 'q'):
            print("Exiting interactive Q&A. Goodbye!")
            break

        if user_q.lower() == 'history':
            past = vector_db.get_all_past_queries()
            if past:
                print(f"\n  📋 Query History ({len(past)} entries):")
                for i, p in enumerate(past):
                    print(f"    [{i+1}] \"{p['query']}\" ({p.get('timestamp', 'N/A')})")
            else:
                print("  No past queries found.")
            continue

        if user_q.lower() == 'stats':
            stats = vector_db.get_stats()
            print(f"\n  📊 VectorDB Statistics:")
            print(f"    Total entries:  {stats['total_entries']}")
            print(f"    Index size:     {stats['index_size']}")
            print(f"    Vocabulary:     {stats['vocab_size']} tokens")
            print(f"    Embedding dim:  {stats['embed_dim']}")
            print(f"    Entry types:    {stats['entry_types']}")
            print(f"    Storage path:   {stats['db_path']}/")
            continue

        print("\n  🔍 Searching VectorDB for similar past queries...")
        similar = vector_db.search_past_queries(user_q, top_k=2)
        if similar:
            print(vector_db.format_search_results(similar, max_display=2))
        else:
            print("    No similar past queries found.")
        
        student_answer = "(Interactive Student LLM chat is WIP)"
        print(f"\nStudent LLM: {student_answer}")

        vector_db.add_entry(
            query=user_q,
            answer=student_answer,
            teacher_answer="",
            kg_facts=kg_facts_str,
            entry_type="query"
        )

if __name__ == "__main__":
    main()
