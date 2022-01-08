import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

model = tf.keras.models.load_model('model1.tf')

newInput = ['I said if he wanted to take a broad view of the thing, it really began with Andrew Jackson.',
            'When enough years had gone by to enable us to look back on them, we sometimes discussed the events leading to his accident.',
            'When he was nearly thirteen, my brother Jem got his arm badly broken at the elbow.',
            'Hello, Hello. How are you? Good. Very good? That is good to hear!',
            'Knock, Knock. Who’s there? Radio. Radio who? Radio not, here I come!',
            'Knock, knock. Who’s there? A broken pencil. A broken pencil who? Never mind. It’s pointless.',
            'Knock, Knock. Who’s there? Kanga. Kanga who? Actually, it’s kangaroo!']
prediction = model.predict(newInput)

print(prediction)