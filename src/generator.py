import json
import pandas as pd

class DatasetGenerator:
    def __init__(self):
        self.dataset = []

    def add_sample(self, question, original_text, extracted_kg, teacher_answer):
        kg_str = json.dumps(extracted_kg)
        
        sample = {
            "question": question,
            "context_text": original_text,
            "kg_representation": kg_str,
            "teacher_answer": teacher_answer
        }
        self.dataset.append(sample)

    def save_dataset(self, filepath="dataset.json"):
        with open(filepath, 'w') as f:
            json.dump(self.dataset, f, indent=2)
            
    def save_csv(self, filepath="dataset.csv"):
        df = pd.DataFrame(self.dataset)
        df.to_csv(filepath, index=False)
