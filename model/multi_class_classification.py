# use mlp for prediction on multi-label classification
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam


class NeuralNet(object):
    def __init__(self, conf: dict, n_inputs: int, n_outputs: int) -> None:
        self.conf = conf
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def create_model(self, hp):
        num_layer = 1
        num_units = 8
        dropout_rate = 0.1
        learning_rate = 0.01

        if hp != None:
            num_layer = hp.Choice("num_layer", values=[1, 2, 3])
            num_units = hp.Choice("num_units", values=[20, 200, 2000])
            dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5)
            learning_rate = hp.Float("learning_rate", min_value=0.0001, max_value=0.01)

        model = Sequential()
        model.add(Dense(input_dim=self.n_inputs, units=num_units, activation="relu"))

        for _ in range(0, num_layer - 1):
            model.add(Dense(units=num_units, activation="relu"))
            model.add(Dropout(rate=dropout_rate))

        model.add(Dense(self.n_outputs, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )

        return model

    def __repr__(self) -> str:
        return str(self.create_model(None).summary())
