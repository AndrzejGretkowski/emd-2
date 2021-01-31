import torch.nn.functional as F
import torch.nn as nn
import torch


class BasicNet(nn.Module):
    def __init__(self, inputs_tuple, outputs):
        super().__init__()
        self.fc = nn.Linear(sum(inputs_tuple), 32)
        self.fc_out = nn.Linear(32, outputs)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def forward(self, review, summary, weight, device='cpu'):
        return self.fc_out(self.fc(torch.cat((review, summary, weight.unsqueeze(1)), dim=1).to(device)))

    @staticmethod
    def collate_fn(batch):
        review, summary, weight, score = [], [], [], []
        for r in batch:
            review.append(torch.mean(r['review'], dim=0))
            summary.append(torch.mean(r['summary'], dim=0))
            weight.append(r['review_weight'])
            score.append(r['score'])
        return {
            'review': torch.stack(review),
            'summary': torch.stack(summary),
            'review_weight': torch.stack(weight),
            'score': torch.stack(score)
        }


class LSTMNet(nn.Module):
    def __init__(self, inputs_tuple, outputs):
        super().__init__()
        self.lstm_review = nn.Sequential(
            nn.LSTM(
                inputs_tuple[0],
                32,
                num_layers=4,
                bidirectional=True,
                dropout=0.2,
                batch_first=True)
        )
        self.lstm_summary = nn.Sequential(
            nn.LSTM(
                inputs_tuple[1],
                32,
                num_layers=2,
                bidirectional=True,
                dropout=0.2,
                batch_first=True)
        )
        self.fc = nn.Linear(32 * 4 + 1, 32)
        self.fc_out = nn.Linear(32, outputs)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()

    def forward(self, review, summary, weight, device='cpu'):
        _, (hidden_review, _) = self.lstm_review(review.to(device))
        _, (hidden_summary, _) = self.lstm_summary(summary.to(device))

        hidden = torch.cat((
            hidden_review[-2,:,:],
            hidden_review[-1,:,:],
            hidden_summary[-2,:,:],
            hidden_summary[-1,:,:],
            weight.unsqueeze(1).to(device)), dim = 1)
        return self.fc_out(self.fc(hidden))

    @staticmethod
    def collate_fn(batch):
        review, summary, weight, score = [], [], [], []
        for r in batch:
            review.append(r['review'])
            summary.append(r['summary'])
            weight.append(r['review_weight'])
            score.append(r['score'])
        return {
            'review': nn.utils.rnn.pad_sequence(review, batch_first=True),
            'summary': nn.utils.rnn.pad_sequence(summary, batch_first=True),
            'review_weight': torch.stack(weight),
            'score': torch.stack(score)
        }