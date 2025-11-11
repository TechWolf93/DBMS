# =========================
# Practical 5: CBOW implementation (Colab-ready)
# Code extracted from Prac_5.pdf (formatting fixed for Colab)
# =========================


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense, Input
import matplotlib.pyplot as plt


corpus = """Natural language processing enables computers to understand human language.
Deep learning models like CBOW help us learn word representations.
Word embeddings capture semantic meaning in vector space.
This makes NLP applications like translation and sentiment analysis possible.
The CBOW model predicts a word based on its context words."""


sentences = [s.strip().lower() for s in corpus.split('.') if s.strip()]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)


vocab_size = len(tokenizer.word_index) + 1
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
index_word[0] = "<PAD>"

print("Vocabulary:", word_index)
print("Sequences:", sequences)
print("Vocab Size:", vocab_size)


def generate_cbow_pairs(sequences, window_size=2):
    contexts, targets = [], []
    context_len = window_size * 2

    for seq in sequences:
        for i, target in enumerate(seq):
            context = []
            for j in range(i - window_size, i + window_size + 1):
                if j == i:
                    continue
                if 0 <= j < len(seq):
                    context.append(seq[j])
                else:
                    context.append(0)
            contexts.append(context)
            targets.append(target)

    return np.array(contexts), np.array(targets)


X, y = generate_cbow_pairs(sequences, window_size=2)

print("\nContext shape:", X.shape)
print("Target shape:", y.shape)
print("Example contexts -> targets:")
for i in range(min(5, len(X))):
    print([index_word[idx] for idx in X[i]], "->", index_word[y[i]])


embedding_dim = 50
context_len = X.shape[1]


model = Sequential()
model.add(Input(shape=(context_len,)))
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=context_len, name="embedding"))
model.add(Lambda(lambda x: tf.reduce_mean(x, axis=1)))
model.add(Dense(vocab_size, activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X, y, epochs=80, batch_size=16, verbose=2)


embeddings = model.get_layer("embedding").get_weights()[0]
print("\nEmbedding matrix shape:", embeddings.shape)


def predict_word(context_words):

    context_indices = [word_index.get(w, 0) for w in context_words]
    context_indices = np.array(context_indices).reshape(1, -1)
    probs = model.predict(context_indices, verbose=0)[0]
    pred_idx = np.argmax(probs)
    return index_word[pred_idx], float(probs[pred_idx])


context = ["deep", "models", "cbow", "help"]
pred_word, prob = predict_word(context)
print("\nContext:", context)
print("Predicted Word:", pred_word, "with probability:", prob)


def nearest_words(word, top_k=5):
    if word not in word_index:
        return []
    w_idx = word_index[word]
    vec = embeddings[w_idx]
    norms = np.linalg.norm(embeddings, axis=1)
    sims = embeddings.dot(vec) / (norms * np.linalg.norm(vec) + 1e-9)
    top = np.argsort(-sims)[1: top_k+1]  # skip the word itself
    return [(index_word[i], float(sims[i])) for i in top]

print("\nNearest words to 'learning':", nearest_words("learning"))


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
