import torch
from torch.utils.data import Dataset

from data import normalize_review_weight


class ReviewsDataset(Dataset):
    def __init__(self, reviews, scores):
        super().__init__()
        self.reviews = reviews
        self.scores = scores

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = self.reviews['reviewTextTensor'][idx]
        summary = self.reviews['summaryTensor'][idx]
        score = self.label_to_float(self.scores[idx])
        weight = normalize_review_weight(self.reviews['helpful'][idx])

        return {
            'review': review,
            'summary': summary,
            'review_weight': torch.tensor(weight, dtype=torch.float),
            'score': score
        }

    @staticmethod
    def label_to_float(val):
        return torch.tensor(float(val) - 1, dtype=torch.long)