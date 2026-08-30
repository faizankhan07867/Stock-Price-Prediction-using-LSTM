import torch
import torch.nn as nn

from config import *


class StockLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.hidden_size = HIDDEN_SIZE

        self.num_layers = NUM_LAYERS

        self.lstm = nn.LSTM(

            input_size=1,

            hidden_size=HIDDEN_SIZE,

            num_layers=NUM_LAYERS,

            batch_first=True,

            dropout=DROPOUT

        )

        self.dropout = nn.Dropout(

            DROPOUT

        )

        self.fc = nn.Linear(

            HIDDEN_SIZE,

            1

        )

    def forward(self, x):

        h0 = torch.zeros(

            self.num_layers,

            x.size(0),

            self.hidden_size

        ).to(DEVICE)

        c0 = torch.zeros(

            self.num_layers,

            x.size(0),

            self.hidden_size

        ).to(DEVICE)

        out, _ = self.lstm(

            x,

            (h0, c0)

        )

        out = out[:, -1, :]

        out = self.dropout(out)

        out = self.fc(out)

        return out


def create_model():

    model = StockLSTM()

    return model.to(DEVICE)


if __name__ == "__main__":

    model = create_model()

    print(model)

    total = sum(

        p.numel()

        for p in model.parameters()

    )

    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )

    print("=" * 50)

    print("Total Parameters :", total)

    print("Trainable :", trainable)

    print("=" * 50)
