import random
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text # Note: is necessary for BERT even it appears as unused!
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_knock_detector_model():
    characters = 'abcdefghijklmnopqrstuvwxyz'
    characters_and_punctuation = 'abcdefghijklmnopqrstuvwxyz ,.!?'
    random.seed(1)

    data = []

    # Pattern we want to teach.
    for x in range(500):
        new_word = ''
        for l in range(5):
            new_word = new_word + random.choice(characters)
        data.append([new_word + ', ' + new_word + '! ', True])

    # Noise.
    for x in range(500):
        new_word = ''
        for l in range(14):
            new_word = new_word + random.choice(characters_and_punctuation)
        data.append([new_word, False])

    # Generate sets.
    df = pd.DataFrame(data, columns=['Text', 'Pattern'])
    train_input, test_input, train_labels, test_labels = train_test_split(df['Text'],
                                                                          df['Pattern'],
                                                                          stratify=df['Pattern'],
                                                                          random_state=1)

    # Create model.
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='input_layer')
    bert_preprocessor_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
                                             name='bert_preprocessor_layer')(input_layer)
    bert_encoder_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4',
                                        name='bert_encoder_layer')(bert_preprocessor_layer)
    dropout_layer = tf.keras.layers.Dropout(0.1, name='dropout_layer')(bert_encoder_layer['pooled_output'])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(dropout_layer)

    model = tf.keras.Model(inputs=[input_layer], outputs=[output_layer])
    modelMetrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
    model.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.01), loss='binary_crossentropy', metrics=[modelMetrics])

    # Train the model.
    model.fit(train_input, train_labels, epochs=30)
    model.evaluate(test_input, test_labels)
    return model


def load_knock_detector_model(path):
    return tf.keras.models.load_model(path)


# model = generate_knock_detector_model()
# shouldSave = input('Save model? y/n\n')
# if shouldSave == 'y':
#     model.save('KnockPatternDetector2.tf')

model = load_knock_detector_model('models/KnockPatternDetector2.tf')

ending_punctuation_samples = [
    'Knock, knock!',
    'Knock, knock.',
    'Knock, knock,',
    'Knock, knock ',
    'Knock, knock?'
]

middle_punctuation_samples = [
    'Knock, knock!',
    'Knock  knock!',
    'Knock. knock!',
    'Knock! knock!',
    'Knock? knock!'
]

length_samples = [
    'asdfg, asdfg!',
    'asdfgh, asdfgh!',
    'asdfghj, asdfghj!',
    'asdfghjk, asdfghjk!',
    'asdfghjkl, asdfghjkl!'
]

identical_words_samples = [
    'Knock, knock!',
    'Bark, bark!',
    'Knock, bark!',
    'Bark, knock!'
]

# Check if the model picks up on the last character needing to be '!' .
print(model.predict(identical_words_samples))
