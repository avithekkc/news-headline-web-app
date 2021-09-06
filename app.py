from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


# HELPER  FUNCTION
def decode_sequence(input_seq):
    # LOADING PKL FILE
    loaded_model = tf.keras.models.load_model('models/news_pre.h5')

    with open('models/x_tokenizer.pickle', 'rb') as handle:
        x_tokenizer = pickle.load(handle)
    with open('models/y_tokenizer.pickle', 'rb') as handle:
        y_tokenizer = pickle.load(handle)

    encoder_model = tf.keras.models.load_model('models/encoder.h5')
    decoder_model = tf.keras.models.load_model('models/decoder.h5')

    reverse_target_word_index = y_tokenizer.index_word
    reverse_source_word_index = x_tokenizer.index_word
    target_word_index = y_tokenizer.word_index

    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Populate the first word of target sequence with the start word.
    target_seq[0, 0] = target_word_index['sostok']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if (sampled_token != 'eostok'):
            decoded_sentence += ' ' + sampled_token

        # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (100 - 1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/news_headline', methods=['GET'])
def news_headline():
    return render_template('news_headline.html')


@app.route('/predict', methods=['POST'])
def predict():
    # GETTING REQUEST
    review = None
    if request.method == "POST":
        text = request.get_json('data')

    with open('models/x_tokenizer.pickle', 'rb') as handle:
        loaded_x_tokenizer = pickle.load(handle)

    with open('models/y_tokenizer.pickle', 'rb') as handle:
        loaded__y_tokenizer = pickle.load(handle)

    seq = loaded_x_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100, padding='post')

    summary = decode_sequence(padded.reshape(1, 100))


    # JSON
    review_prediction = {
        'summary': summary.title(),
    }

    return jsonify(review_prediction)


if __name__ == '__main__':
    app.run()
