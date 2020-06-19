# Named-Entity-Recognition
Named Entity Recognition (NER) model using Bidirectional LSTM and ELMo embedding.

**Named entity recognition (NER)**

Also called entity identification or entity extraction, NER is an information extraction technique that automatically identifies named entities in a text and classifies them into predefined categories. Entities can be names of people, organizations, locations, times, quantities, monetary values, percentages, and more.

**Bi-directional LSTMs**

we need to use bi-directional LSTMs because using a standard LSTM to make predictions will only take the “past” information in a sequence of the text into account. For NER, since the context covers past and future labels in a sequence, we need to take both the past and the future information into account. A bidirectional LSTM is a combination of two LSTMs — one runs forward from “right to left” and one runs backward from “left to right”.

**ELMo Embedding**

ELMo goes beyond traditional embedding techniques. It uses a deep, bi-directional LSTM model to create word representations.
Rather than a dictionary of words and their corresponding vectors, ELMo analyses words within the context that they are used. It is also character based, allowing the model to form representations of out-of-vocabulary words.
This therefore means that the way ELMo is used is quite different to word2vec or fastText. Rather than having a dictionary ‘look-up’ of words and their corresponding vectors, ELMo instead creates vectors on-the-fly by passing text through the deep learning model.

**Dataset**

https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/data


