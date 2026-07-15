import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class KnowledgeExtractor:
    def __init__(self, model="llama-3.1-8b-instant"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model

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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4096
            )
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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2048
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error fetching information: {e}")
            return "No information could be retrieved."
