import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

df = pd.read_csv('data.csv')
df = df[df.Gibberish == 0]
trainingInput, testingInput, trainingLabels, testingLabels = train_test_split(df['Text'], df['Joke'], stratify=df['Joke'], random_state=1)

inputLayer = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Input_Layer')

bertPreprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3', name='bert_preprocessor_layer')
bertEncoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4', name='bert_encoder_layer')
bertLayer = bertEncoder(bertPreprocessor(inputLayer))

dropoutLayer = tf.keras.layers.Dropout(0.2, name='dropout_layer')(bertLayer['pooled_output'])
outputLayer = tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')(dropoutLayer)

model = tf.keras.Model(inputs=[inputLayer], outputs=[outputLayer])
modelMetrics = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[modelMetrics])

model.fit(trainingInput, trainingLabels, epochs=40)
model.evaluate(testingInput, testingLabels)

model.save('model1.tf')