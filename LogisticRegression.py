import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt


class LogisticRegression:
    """docstring for LogisticRegression"""

    def __init__(self, learning_rate=0.01, epochs=50000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    # define sigmoid function
    def sigmoid(self, z):
        '''
        Compute the sigmoid activation of the input
        param z: input data coming out from one layer of the model ()
        return: apply sigmoid
        '''
        sig = 1.0 / (1 + np.exp(-z))
        return sig

    def compute_accuracy(self, a, Y):
        y_pred = a.round()
        val = y_pred == Y
        return np.sum(val) / Y.shape[0]

    def train(self, X, Y):
        '''
        train the model
        param: X: input features of size (number of training data * number of features) = (m, n)
               Y: ground truth value of size (number of training data, 1 = (m, 1)
        '''

        # weight and bias initialization
        # weight dim [number of features, 1] = (n, 1)
        # bias is a scalar value but python can broadcast while calculating
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0

        losses = []
        accuracy_data = []

        for i in range(self.epochs):
            m = X.shape[0]

            # forward propagation
            z = np.dot(X, self.w) + self.b
            # print('z val', z)
            a = self.sigmoid(z)
            # print('a ', a)

            loss = - 1 / m * (np.sum(Y * np.log(a) + (1 - Y) * np.log(1 - a)))
            accuracy = self.compute_accuracy(a, Y)

            # gradient or back propagation
            # Y = Y.reshape((Y.shape[0], 1))
            dZ = a - Y
            # print('dZ dim', dZ.shape)
            dW = (1 / m) * np.dot(X.T, dZ)
            db = (1 / m) * np.sum(dZ)

            # update weight and bias
            self.w -= self.learning_rate * dW
            self.b -= self.learning_rate * db

            losses.append(loss)
            accuracy_data.append(accuracy)

            print(f'epoch: {i} \t loss: {loss} \t accuracy: {accuracy} \t')

        self.plot(losses, accuracy_data, self.epochs)

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.w) + self.b)
        return y_pred.round()

    def plot(self, loss, accuracy, epoch):
        # Ref https://www.pluralsight.com/guides/data-visualization-deep-learning-model-using-matplotlib
        fig1 = plt.figure(1)
        epochs = range(1, epoch + 1)
        plt.plot(epochs, loss, 'g', label='Training loss')
        plt.title('Loss Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        # fig1.savefig('loss_curve.png')
        plt.show()

        fig2 = plt.figure(2)
        plt.plot(epochs, accuracy, 'g', label='Training Accuracy')
        # plt.plot(epochs, valid_acc_history, 'b', label='validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        # fig2.savefig('accuracy_curve.png')
        plt.show()


def main():
    # prepare the input features and label
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    # standardize the data
    new_X = preprocessing.scale(X)

    y = y.reshape((y.shape[0], 1))

    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.20, random_state=42)
    print('X dim', X_train.shape)
    print('y dim', y_train.shape)

    # setup model
    model = LogisticRegression(learning_rate=0.1, epochs=200)

    # train the model
    model.train(X_train, y_train)

    # predict the model
    y_pred = model.predict(X_test)

    test_accuracy = (y_pred == y_test).mean()

    print(f'Model test accuracy: {test_accuracy} \t')


if __name__ == "__main__":
    main()