from mnist_loader import load_data_wrapper
from network2 import Network

training_data, validation_data, test_data = load_data_wrapper()

net = Network([784, 500, 800, 500, 10])

net.SGD(training_data=training_data, epochs=10, mini_batch_size=500, eta=0.01, evaluation_data=test_data,
        monitor_evaluation_accuracy=True, monitor_evaluation_cost=True, monitor_training_accuracy=True, monitor_training_cost=True)