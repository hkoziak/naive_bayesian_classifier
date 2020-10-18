from bayesian_classifier import BayesianClassifier

import pandas as pd

TRAIN_DATA = "train.csv"
TEST_DATA = "test.csv"
stop_words = "stop_words.txt"
punctuation = """!.,&*"?/:;'"""
filter_type = "sentiment"


def process_data(data_file):
    """
    Function for data processing and split it into X and y sets.
    :param data_file: str - train data
    :return: pd.DataFrame|list, pd.DataFrame|list - X and y data frames or lists
    """
    with open(stop_words) as stop_f:
        stop_content = stop_f.readlines()
    stop_content = [x.strip() for x in stop_content]
    with open(data_file) as f:
        data = pd.read_csv(f)
        for line in range(len(data["text"])):
            data["text"][line] = [x for x in data["text"][line].lower().split()
                                  if x not in stop_content and x not in punctuation]
    return data["text"], data[filter_type]


if __name__ == "__main__":
    train_x, train_y = process_data(TRAIN_DATA)
    test_x, test_y = process_data(TEST_DATA)

    classifier = BayesianClassifier()
    classifier.fit(train_x, train_y)
    print(classifier.predict(test_x[23]))
    print(classifier.predict_prob(test_x[0], test_y[0]))

    print("model score: ", classifier.score(test_x, test_y))
