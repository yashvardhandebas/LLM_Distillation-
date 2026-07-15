class EvaluationFramework:
    def evaluate_extractor(self, extracted_rels, ground_truth_rels):
        ext_set = set([(r.get('source'), r.get('target'), r.get('type')) for r in extracted_rels])
        gt_set = set([(r.get('source'), r.get('target'), r.get('type')) for r in ground_truth_rels])
        
        if not ext_set or not gt_set:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        tp = len(ext_set.intersection(gt_set))
        fp = len(ext_set - gt_set)
        fn = len(gt_set - ext_set)
        
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        
        return {"precision": precision, "recall": recall, "f1": f1}
        
    def evaluate_student(self, student_answers, teacher_answers):
        scores: list[float] = []
        for s, t in zip(student_answers, teacher_answers):
            s_words = set(s.lower().split())
            t_words = set(t.lower().split())
            if not s_words or not t_words:
                scores.append(0.0)
                continue
            intersection = len(s_words.intersection(t_words))
            union = len(s_words.union(t_words))
            scores.append(intersection / union)
        return sum(scores) / len(scores) if scores else 0.0
