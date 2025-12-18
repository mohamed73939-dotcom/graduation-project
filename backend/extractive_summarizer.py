"""
Simple extractive summarizer using frequency & sentence scoring (TextRank-like heuristic).
"""
import re
import math
from collections import Counter
from logger_config import summarizer_logger

class ExtractiveSummarizer:
    def __init__(self):
        pass
    
    def extract_top_sentences(self, text, top_n=25, language='ar'):
        # 1. Split into sentences
        sentences = re.split(r'(?<=[.!?؟])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip().split()) > 4]
        
        if not sentences:
            return []
            
        n = len(sentences)
        if n <= top_n:
            return sentences

        # 2. Build Similarity Matrix (Common Words Ratio)
        # Using simple set overlap to be lightweight (no heavy embeddings)
        def sentence_similarity(s1, s2):
            words1 = set(s1.lower().split())
            words2 = set(s2.lower().split())
            if not words1 or not words2:
                return 0.0
            avg_len = (math.log(len(words1)) + math.log(len(words2))) / 2.0
            if avg_len == 0: return 0.0
            return len(words1.intersection(words2)) / (avg_len + 1.0)

        # 3. Compute PageRank Scores (Iterative)
        scores = [1.0] * n
        # We only consider local context (window) to favor flow, or full graph? 
        # For lectures, full graph is better for global "centrality" but window is better for "coverage".
        # Let's use a fully connected weighted graph for best results.
        
        # Precompute weights to save time
        graph = {}
        for i in range(n):
            graph[i] = []
            for j in range(n):
                if i == j: continue
                # Optimization: Only compare if somewhat close or similar length? No, full compare for quality.
                sim = sentence_similarity(sentences[i], sentences[j])
                if sim > 0:
                    graph[i].append((j, sim))

        damping = 0.85
        for _ in range(10): # 10 iterations usually enough for convergence on small graphs
            new_scores = [1 - damping] * n
            for i in range(n):
                for j, weight in graph[i]:
                    # Contribution from node j to i
                    width_j = sum([w for _, w in graph[j]])
                    if width_j > 0:
                        new_scores[i] += damping * (weight / width_j) * scores[j]
            scores = new_scores

        # 4. Rank and Select
        ranked_sentences = sorted(((scores[i], i) for i in range(n)), reverse=True)
        top_indices = sorted([i for _, i in ranked_sentences[:top_n]]) # Sort by Index to keep Chronological Order
        
        selected_sentences = [sentences[i] for i in top_indices]
        
        summarizer_logger.info(f"TextRank selected {len(selected_sentences)} sentences from {n}")
        return selected_sentences