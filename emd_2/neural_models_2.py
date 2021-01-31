import time
from torch.utils.data import DataLoader
from data import get_all_data, score_metric, normalize_review_weight
from neural import BasicNet, LSTMNet
from dataset import ReviewsDataset
import torch
import tqdm
import spacy
import pickle
import os

BATCH_SIZE = 64
EMBED_DIM = 96
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 50
min_valid_loss = float('inf')
max_valid_acc = 0.0

model = LSTMNet((EMBED_DIM, EMBED_DIM, 1), 5)
model.to(device)

crit_weight = torch.tensor([0.89177587, 0.93929913, 0.88501973, 0.79031866, 0.49358662])
criterion = torch.nn.CrossEntropyLoss(reduction='sum', weight=crit_weight).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


def custom_criterion(output, target):
    return criterion(output, target) * torch.abs(target.float() - output.argmax(1)).mean()

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, collate_fn=model.collate_fn)
    for i, batch in enumerate(tqdm.tqdm(data)):
        optimizer.zero_grad()
        output = model(batch['review'], batch['summary'], batch['review_weight'], device)
        loss = custom_criterion(output, batch['score'].to(device))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) ==  batch['score'].to(device)).sum().item()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    test_loss = 0
    test_acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=model.collate_fn)
    for batch in data:
        with torch.no_grad():
            output = model(batch['review'], batch['summary'], batch['review_weight'], device)
            loss = custom_criterion(output, batch['score'].to(device))
            test_loss += loss.item()
            test_acc += (output.argmax(1) ==  batch['score'].to(device)).sum().item()
    return test_loss / len(data_), test_acc / len(data_)


if __name__ == '__main__':
    train_dataset_pkl = 'model/train_dataset.pkl'
    test_dataset_pkl = 'model/test_dataset.pkl'

    if os.path.exists(train_dataset_pkl) and os.path.exists(test_dataset_pkl):
        print('Loading train set...')
        with open(train_dataset_pkl, 'rb') as train_pkl:
            train_dataset = pickle.load(train_pkl)
        print('Loading test set...')
        with open(test_dataset_pkl, 'rb') as test_pkl:
            test_dataset = pickle.load(test_pkl)
        print('Loaded.')

    else:
        X_train, y_train, X_test, y_test = get_all_data()

        # Processing to tensors
        nlp = spacy.load('en_core_web_md')

        reviewTensors = []
        summaryTensors = []
        for i, row in tqdm.tqdm(X_train.iterrows(), total=len(X_train)):
            reviewTensors.append(torch.from_numpy(nlp(row['reviewText'].lower()).tensor))
            summaryTensors.append(torch.from_numpy(nlp(row['summary'].lower()).tensor))
        X_train['reviewTextTensor'] = reviewTensors
        X_train['summaryTensor'] = summaryTensors

        reviewTensors = []
        summaryTensors = []
        for i, row in tqdm.tqdm(X_test.iterrows(), total=len(X_test)):
            reviewTensors.append(torch.from_numpy(nlp(row['reviewText'].lower()).tensor))
            summaryTensors.append(torch.from_numpy(nlp(row['summary'].lower()).tensor))
        X_test['reviewTextTensor'] = reviewTensors
        X_test['summaryTensor'] = summaryTensors

        train_dataset = ReviewsDataset(X_train, y_train)
        test_dataset = ReviewsDataset(X_test, y_test)

        with open(train_dataset_pkl, 'wb') as train_pkl:
            pickle.dump(train_dataset, train_pkl)

        with open(test_dataset_pkl, 'wb') as test_pkl:
            pickle.dump(test_dataset, test_pkl)


    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_func(train_dataset)
        valid_loss, valid_acc = test(test_dataset)
        # Adjust the learning rate
        scheduler.step(valid_loss)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        #save the best model
        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model/lstm-net.pt')

        print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')