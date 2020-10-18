class BayesianClassifier:
    """
    Implementation of Naive Bayes classification algorithm.
    """
    def __init__(self):
        self._vocab_freq = None
        self._neutp = None
        self._posp = None
        self._negp = None

    def fit(self, content, labels):
        """
        Fit Naive Bayes parameters according to train data X and y.
        :param content: pd.DataFrame|list - train input/messages
        :param labels: pd.DataFrame|list - train output/labels
        :return: None
        """
        self._getps(labels)
        num = len(labels)
        vocab = []
        for line in content:
            for word in line:
                vocab.append(word)
        vocab = list(set(vocab))

        self._vocab_freq = dict()
        for word in vocab:
            self._vocab_freq[word] = {"neutral": 0, "positive": 0,
                                      "negative": 0, "total": 0}
        for i in range(len(content)):
            message = content[i]
            for word in vocab:
                if word in message:
                    self._vocab_freq[word]["total"] += 1
                    k = labels[i]
                    self._vocab_freq[word][k] += 1

        for key in self._vocab_freq.keys():
            total = self._vocab_freq[key]["total"]
            self._vocab_freq[key]["neutral"] /= total
            self._vocab_freq[key]["positive"] /= total
            self._vocab_freq[key]["negative"] /= total
            self._vocab_freq[key]["total"] = total / num

    def predict_prob(self, message, label):
        """
        Calculate the probability that a given label can be assigned to a given message.
        :param message: str - input message; list is message prepared
        :param label: str - label - "positive", "negative" or "neutral"
        :return: float - probability P(label|message)
        """
        if isinstance(message, str):
            message = message.lower().split()
        words = [x for x in message if x not in """!.,&*?/:;"'"""]
        numerator = 1
        pos_denum_p = 1
        neg_denum_p = 1
        neut_denum_p = 1
        for word in words:
            wdict = self._vocab_freq.get(word, 0)
            if wdict:
                numerator *= wdict[label]
                pos_denum_p *= wdict["positive"] + 1
                neg_denum_p *= wdict["negative"] + 1
                neut_denum_p *= wdict["neutral"] + 1
        denumerator = pos_denum_p * self._posp + neg_denum_p * self._negp + neut_denum_p * self._neutp
        result = numerator / denumerator
        if label == "positive":
            result *= self._posp
        if label == "neutral":
            result *= self._neutp
        else:
            result *= self._negp
        return result

    def predict(self, message):
        """
        Predict label for a given message.
        :param message: str - message
        :return: str - label that is most likely to be truly assigned to a given message
        """
        prob_neut = self.predict_prob(message, "neutral")
        prob_neg = self.predict_prob(message, "negative")
        prob_pos = self.predict_prob(message, "positive")
        if prob_neut > prob_neg and prob_neut > prob_pos:
            return "neutral"
        elif prob_neg > prob_neut and prob_neg > prob_pos:
            return "negative"
        else:
            return "positive"

    def score(self, content, labels):
        """
        Return the mean accuracy on the given test data and labels - the efficiency of a trained model.
        :param сontent: pd.DataFrame|list - test data - messages
        :param labels: pd.DataFrame|list - test labels
        :return: float - percentage of coincidences
        """
        coincidences = 0
        num_of_messages = len(content)
        for i in range(num_of_messages):
            if self.predict(content[i]) == labels[i]:
                coincidences += 1
        return round(coincidences/num_of_messages, 2)

    def _getps(self, marker_set):
        pos, neg, neut = 0, 0, 0
        length = len(marker_set)
        for i in range(length):
            quality = marker_set[i]
            if quality == "positive":
                pos += 1
            elif quality == "neutral":
                neut += 1
            else:
                neg += 1
        self._posp, self._negp, self._neutp = pos / length, neg / length, neut / length
