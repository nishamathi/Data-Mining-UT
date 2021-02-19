import torch
from rnn_utils import *
import os
import argparse
import progressbar

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='Code to train RNN and intepretable RNN models')
parser.add_argument('--path', help='Path to dataset', type=str, required=True)
parser.add_argument('--epochs', help='Number of epochs for which to train the model', type=int, default=10)
# parser.add_argument('--batch_size', help='Number of examples to use per update', type=int, default=10)
parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
parser.add_argument('--no_cuda', dest='use_cuda', help='Flag to not use CUDA', action='store_false')
parser.set_defaults(use_cuda=True)

args = parser.parse_args()
assert os.path.exists(args.path), 'Path to dataset does not exist'

widgets = [
            progressbar.ETA(),
            progressbar.Bar(),
            ' ', progressbar.DynamicMessage('Error')
            ]

args.batch_size = 1
embedding_size = 40
hidden_size = 40

data_generator = DataGenerator(args.path, args.batch_size, mode='train', use_cuda=args.use_cuda)
model = BiLSTMModel(21, embedding_size, hidden_size)
if args.use_cuda:
    model = model.cuda()
    model.tensors_to_cuda()

opt = torch.optim.Adam(model.parameters(), lr=args.lr)

for epoch in range(args.epochs):
    tot_loss = 0.0

    print("Epoch {}/{}".format(epoch+1, args.epochs))
    with progressbar.ProgressBar(max_value = data_generator.steps_per_epoch, widgets=widgets) as bar:
        for i in range(data_generator.steps_per_epoch):
            xs, ys = data_generator.next()
            
            y_preds = []
            loss = 0.0
            for x,y in zip(xs, ys):
                y_hat = model.forward(x)
                loss += torch.mean((y - y_hat)**2) # MSE
                y_preds.append(y_hat)
            loss /= args.batch_size

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot_loss += loss.detach().item()

            bar.update(i, Error=tot_loss/(i+1))

save_model(model, 'models', {'embedding_size': embedding_size, 'hidden_size': hidden_size, 'lr': args.lr, 'epochs':args.epochs, 'batch_size':args.batch_size})

