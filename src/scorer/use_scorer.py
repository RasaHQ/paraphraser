from rasa.shared.nlu.training_data.message import Message
from whatlies.language import TFHubLanguage
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.distance import cosine


class USEScorer:
    def __init__(self):
        self.model = TFHubLanguage(
            "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        )

    @staticmethod
    def compute_similarity_score(feature_vec_a: np.ndarray, feature_vec_b: np.ndarray):
        return float(1 - cosine(feature_vec_a, feature_vec_b))

    def compute_similarity_for_pair(self, a: Message, b: Message):
        features_a = a.get("sentence_features").vector
        features_b = b.get("sentence_features").vector

        return self.compute_similarity_score(features_a, features_b)

    def compute_features(self, example: Message):
        features = self.model[example.get("text")]
        example.set("sentence_features", features)

    def compute_similarity_with_paraphrases(self, example: Message):

        # Set features for text of example itself first.
        self.compute_features(example)

        paraphrases = example.get("metadata").get("example").get("paraphrases")

        similarity_scores = []

        # construct individual message for each paraphrase
        for paraphrase in paraphrases:
            message = Message.build(text=paraphrase)
            self.compute_features(message)
            similarity = self.compute_similarity_for_pair(example, message)
            similarity_scores.append(similarity)

        return similarity_scores

    def compute_similarities(self, examples: List[Message]) -> List[List[float]]:

        scores_for_collection = []

        for example in examples:

            similarity_scores = self.compute_similarity_with_paraphrases(example)
            scores_for_collection.append(similarity_scores)

        return scores_for_collection


if __name__ == "__main__":
    a = Message.build(text="hellO, how are you doing?", intent="hello")
    a.set(
        "metadata",
        {"example": {"paraphrases": ["how are you doing too?", "good to meet you"]}},
    )
    scorer = USEScorer()
    scorer.compute_similarity_with_paraphrases(a)
