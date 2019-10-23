import pickle
import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]

def transform_to_dataset(tagged_sentences):

    X, y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            #print(tagged)
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
 
    return X, y


def get_train_and_test(word_list):
    train_size = int(.75*len(word_list))
    train_sent = word_list[:train_size]
    test_sent = word_list[train_size:]

    train_X, train_y = transform_to_dataset(train_sent)
    test_X, test_y = transform_to_dataset(test_sent)
    return train_X, train_y, test_X, test_y

with open('ph_corpus', 'rb') as ph:
    word_list_ph = pickle.load(ph)

with open('proiel_corpus', 'rb') as proiel:
    word_list_proiel = pickle.load(proiel)


random.shuffle(word_list_ph)
random.shuffle(word_list_proiel)

short_length_ph = int(0.2*len(word_list_ph))
short_length_proiel = int(0.2*len(word_list_proiel))

word_list_ph_short = word_list_ph[:short_length_ph]
word_list_proiel_short = word_list_proiel[:short_length_proiel]

train_X_ph, train_y_ph, test_X_ph, test_y_ph = get_train_and_test(word_list_ph_short)
train_X_proiel, train_y_proiel, test_X_proiel, test_y_proiel = get_train_and_test(word_list_proiel_short)

clf_ph = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', XGBClassifier())
])

print('Starting training for ph corpus')

print('Number of training words: ' + str(len(train_X_ph)))

clf_ph.fit(train_X_ph, train_y_ph)   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)
 
print ('Training completed')
 
print ("Accuracy: " + str(clf_ph.score(test_X_ph, test_y_ph)))

print('Starting training for proiel corpus')


classifier_param = {'n_estimator: 2'}

clf_proiel = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', XGBClassifier(**classifier_param))
])

clf_proiel.fit(train_X_proiel, train_y_proiel)

print('Training completed')

print("Accuracy: " + str(clf_proiel.score(test_X_proiel, test_y_proiel)))

with open('ph_gradient_boost_model', 'wb') as ph:
    pickle.dump(clf_ph, ph)

with open('proiel_gradient_boost_model', 'wb') as proiel:
    pickle.dump(clf_proiel, proiel)