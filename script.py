from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import pickle
import keras.backend as K
import argparse

model = None
tokenizer = None

def load_model():

    global model

    with open('models/model_architecture.json', 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    model.load_weights('models/model_weights.h5')
    
def load_tokenizer():
    
    global tokenizer
    
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

def preprocess(x):
    x = str(x).lower()
    x = x.replace("won't", "will not")
    x = x.replace("cannot", "can not")
    x = x.replace("can't", "can not")
    x = x.replace("n't", " not")
    x = x.replace("what's", "what is")
    x = x.replace("it's", "it is")
    x = x.replace("'ve", " have")
    x = x.replace("i'm", "i am")
    x = x.replace("'re", " are")
    x = x.replace("he's", "he is")
    x = x.replace("she's", "she is")
    x = x.replace("'s", " own")
    x = x.replace("%", " percent ")
    x = x.replace("₹", " rupee ")
    x = x.replace("$", " dollar ")
    x = x.replace("€", " euro ")
    x = x.replace("'ll", " will")
    x = x.strip()
    x = ' '.join(x.split())

    return x

def prepare_text_feat(question_1, question_2):
    
    question_1 = preprocess(question_1)
    question_2 = preprocess(question_2)
    
    encoded_question_1 = tokenizer.texts_to_sequences([question_1])
    encoded_question_2 = tokenizer.texts_to_sequences([question_2])
    padded_question_1 = pad_sequences(encoded_question_1, maxlen=50, padding='post')
    padded_question_2 = pad_sequences(encoded_question_2, maxlen=50, padding='post')
    
    X = [padded_question_1, padded_question_2]
    
    return X

def predict(question_1, question_2):
    
    X = prepare_text_feat(question_1, question_2)
    y = model.predict(X)
    
    if y > 0.5:
        return 1
    else:
        return 0
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence1")
    parser.add_argument("sentence2")
    args = parser.parse_args()
    
    load_model()
    load_tokenizer()
    
    print()
    print('Is Duplicate : ')
    print(predict(args.sentence1, args.sentence2))