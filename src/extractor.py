import os
import sys
import json
import time
from groq import Groq
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

load_dotenv()

class KnowledgeExtractor:
    FALLBACK_MODELS = ["qwen/qwen3.8-27b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "allam-2-7b"]

    def __init__(self, model=None):
        self.client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
            timeout=60.0
        )
        self.model = os.environ.get("GROQ_MODEL") or model or "qwen/qwen3.8-27b"
        print(f"[KnowledgeExtractor] Initialized with model: {self.model}")

    def _execute_with_retry(self, api_call_func, max_retries=4, initial_delay=3):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return api_call_func(self.model)
            except Exception as e:
                err_msg = str(e)
                is_rate_limit = "429" in err_msg or "rate limit" in err_msg.lower() or "rate_limit_exceeded" in err_msg
                is_timeout = "timeout" in err_msg.lower() or "time out" in err_msg.lower()
                is_model_error = "model_not_found" in err_msg.lower() or "decommissioned" in err_msg.lower() or "not found" in err_msg.lower()
                
                if is_model_error:
                    for fb in self.FALLBACK_MODELS:
                        if fb != self.model:
                            print(f"[!] [Groq Model Fallback] Switching from {self.model} to {fb} due to error: {err_msg}")
                            self.model = fb
                            break
                    try:
                        return api_call_func(self.model)
                    except Exception as inner_e:
                        err_msg = str(inner_e)

                if (is_rate_limit or is_timeout) and attempt < max_retries - 1:
                    print(f"[!] [Groq API Warning] Attempt {attempt+1}/{max_retries} failed ({err_msg}). Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise e

    def extract(self, document_text):
        prompt = f"""You are an expert Knowledge Graph construction system. Your task is to extract ALL possible entities and ALL possible relationships from the text below. Be EXHAUSTIVE — do not omit any entity or relationship, even minor ones.

RULES:
1. Extract EVERY named entity: people, organizations, locations, dates, events, laws, concepts, products, titles, roles, policies, awards, countries, institutions.
2. Extract EVERY relationship between ANY two entities mentioned. Include direct and implied relationships.
3. Entity types to use: Person, Organization, Location, Date, Event, Law, Policy, Product, Concept, Role, Country, Institution, Award, Legislation.
4. Relation types examples: Founded, CEO_Of, Born_In, Located_In, Married_To, Parent_Of, Child_Of, Employee_Of, Appointed_By, Passed_Law, Signed_Agreement, Won_Election, Lost_To, Member_Of, Acquired, Invested_In, Attended, Graduated_From, Implemented_Policy, Allied_With, Opposed_By, Succeeded_By, Preceded_By, Part_Of, Owns, Authored, Led, Joined, Left, Met_With, Negotiated_With, Imposed_Sanctions_On.
5. Assign confidence: 1.0 = explicitly stated, 0.9 = strongly implied, 0.7 = reasonably inferred.
6. Aim for at least 15+ entities and 20+ relations if the text supports it.

Text:
{document_text}

Output ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "Entity Name", "type": "EntityType"}}
  ],
  "relations": [
    {{"source": "Entity A", "target": "Entity B", "type": "RelationType", "confidence": 0.95}}
  ]
}}"""
        
        def api_call(m=None):
            return self.client.chat.completions.create(
                model=m or self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2048  # Increased from 1000 to avoid JSON truncation
            )
            
        try:
            response = self._execute_with_retry(api_call, max_retries=4, initial_delay=3)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error during extraction or parsing JSON: {e}")
            return {"entities": [], "relations": []}

    def generate_teacher_answer(self, question, context):
        prompt = f"""
Given the following context, please answer the question thoroughly and accurately.

Context:
{context}

Question:
{question}
"""
        
        def api_call(m=None):
            return self.client.chat.completions.create(
                model=m or self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048  # Increased to prevent truncation
            )
            
        try:
            response = self._execute_with_retry(api_call, max_retries=4, initial_delay=3)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating teacher answer: {e}")
            return "Failed to generate answer from Teacher LLM due to rate limit or API error."

    def get_information(self, query):
        prompt = f"""Provide a DETAILED, FACTUAL, and COMPREHENSIVE overview about: {query}

Requirements:
- Cover background, history, key people, organizations, locations, dates, and events
- Mention specific relationships between entities (who founded what, who worked with whom, where things happened, when events occurred)
- Include at least 3-4 paragraphs with concrete facts and named entities
- Write in clear prose (no bullet points), rich with entity names so a knowledge graph can be built from it
- Be specific: use full names, official titles, and precise dates where known"""
        
        def api_call(m=None):
            return self.client.chat.completions.create(
                model=m or self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048  # Increased to prevent truncation
            )
            
        try:
            response = self._execute_with_retry(api_call, max_retries=4, initial_delay=3)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error fetching information: {e}")
            return "No information could be retrieved."
