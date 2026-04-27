import logging
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Any

class SemanticEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def cluster_segments(self, segments: List[Dict[str, Any]], max_clusters=5) -> List[Dict[str, Any]]:
        """
        Groups transcript segments into semantic clusters.
        """
        if not segments:
            return []

        texts = [s["text"] for s in segments]
        embeddings = self.model.encode(texts)

        # Dynamic cluster count
        n_clusters = min(len(segments), max_clusters)
        if n_clusters < 2:
            for s in segments: s["cluster_id"] = 0
            return segments

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        for i, segment in enumerate(segments):
            segment["cluster_id"] = int(cluster_labels[i])
            segment["topic"] = f"Topic {cluster_labels[i]}" # Placeholder for future logic

        self.logger.info(f"Clustered {len(segments)} segments into {n_clusters} clusters.")
        return segments
