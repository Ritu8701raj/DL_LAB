from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ["I love AI", "I hate bugs"]
labels = [1, 0]

encodings = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="tf"
)

model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased'
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=model.compute_loss,
    metrics=['accuracy']
)

model.fit(encodings['input_ids'], labels, epochs=2)
