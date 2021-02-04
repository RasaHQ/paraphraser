from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.featurizers.dense_featurizer.lm_featurizer import LanguageModelFeaturizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from typing import List, Dict, Any
import numpy as np
from scipy.spatial.distance import cosine


class LaBSEScorer:
    def __init__(self):
        self.featurizer = LanguageModelFeaturizer(
            {"model_name": "bert", "model_weights": "rasa/LaBSE"}
        )
        self.tokenizer = WhitespaceTokenizer()

    @staticmethod
    def compute_similarity_score(feature_vec_a: np.ndarray, feature_vec_b: np.ndarray):
        return 1 - cosine(feature_vec_a, feature_vec_b)

    def compute_similarity_for_pair(self, a: Message, b: Message):
        features_a = a.features[0].features
        features_b = b.features[0].features

        return self.compute_similarity_score(features_a, features_b)

    def compute_features(self, example: Message):
        self.tokenizer.process(example)
        self.featurizer.process(example)

    def compute_similarity_with_paraphrases(self, example: Message):

        # Set features for text of example itself first.
        self.featurizer.process(example)

        paraphrases = example.get("metadata").get("paraphrases")

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
