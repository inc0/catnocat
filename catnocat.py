
# coding: utf-8

import argparse
import tensorflow as tf
from tqdm import tqdm
import time
from sklearn.model_selection import train_test_split
import os
from flask import Flask, request, render_template


app = Flask(__name__)

parser = argparse.ArgumentParser(description='Cat or no cat?')
parser.add_argument('-s', required=False, help='Serve http', action='store_true')
serve = parser.parse_args().s


square_size = 200
img_path = "../../datasets/imagenet/parsed_cats_200/"  # Dir with training data. File name starts from 1 or 0 determine class
n_epochs = 25
model_fpath = "../../models/kotniekot.ckpt"  # Saved model path
model_name = "cnn-{}".format(int(time.time()))


n_inputs = square_size ** 2
filenames = os.listdir(img_path)
filenames_0 = [f for f in filenames if f.startswith("0")][:5000]  # To balance dataset
filenames_1 = [f for f in filenames if f.startswith("1")]
filenames = filenames_0 + filenames_1

length = len(filenames)
labels = [int(img[0]) for img in filenames]


X_train, X_test, y_train, y_test = train_test_split(filenames, labels, test_size=0.25, random_state=42)


def parse_image(image_string):
    image_decoded = tf.image.decode_image(image_string, channels=1)
    image_decoded = tf.image.resize_image_with_crop_or_pad(image_decoded, square_size, square_size)
    image_decoded = tf.reshape(image_decoded, [-1])
    return image_decoded

def _load_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = parse_image(image_string)
    return image_decoded


def _get_image(filename, label):
    image_string = tf.read_file(img_path + filename)
    image_decoded = parse_image(image_string)
    return image_decoded, label

def get_dataset(data, labels):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.map(_get_image)
    dataset = dataset.batch(200)
    dataset = dataset.repeat()
    dataset = dataset.make_one_shot_iterator()
    return dataset.get_next()

with tf.name_scope("dataset"):
    X_train = tf.convert_to_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)

    X_test = tf.convert_to_tensor(X_test)
    y_test = tf.convert_to_tensor(y_test)

    train_iter = get_dataset(X_train, y_train)
    test_iter = get_dataset(X_test, y_test)


X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("conv_net_definition"):
    hidden1 = tf.layers.dense(X, 2000, name="hidden1",
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, 1000, name="hidden2",
                              activation=tf.nn.relu)
    hidden3 = tf.layers.dense(hidden2, 1000, name="hidden3",
                              activation=tf.nn.relu)
    hidden4 = tf.layers.dense(hidden3, 1000, name="hidden4",
                              activation=tf.nn.relu)
    hidden5 = tf.layers.dense(hidden4, 700, name="hidden5",
                              activation=tf.nn.relu)
    hidden6 = tf.layers.dense(hidden5, 500, name="hidden6",
                              activation=tf.nn.relu)
    logits = tf.layers.dense(hidden6, 2, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                              logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")

learning_rate = 0.0001

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    tf.summary.scalar('loss', loss)

with tf.name_scope("eval"):
    cls = tf.nn.top_k(logits)
    precision = tf.metrics.precision(cls[1], y)
    tf.summary.scalar('precision', precision[1])
    accuracy = tf.metrics.accuracy(cls[1], y)
    tf.summary.scalar('accuracy', accuracy[1])
    recall = tf.metrics.recall(cls[1], y)
    tf.summary.scalar('recall', recall[1])



def train(sess, train_writer, test_writer):
    init_time = time.time()
    for j in range(n_epochs):
        for i in tqdm(range(int(length * 0.75 / 200))):
            input_data = sess.run(train_iter)
            summary_train, _ = sess.run([merged, training_op], feed_dict={X: input_data[0], y: input_data[1]})
            train_writer.add_summary(summary_train, i)
        test_data = sess.run(test_iter)
        summary_test, acc = sess.run([merged, accuracy], feed_dict={X: test_data[0], y: test_data[1]})
        print("epoch: {} elapsed: {} acc {}".format(j, (time.time() - init_time), acc[0]))
        test_writer.add_summary(summary_test, j)


sess = tf.Session()
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('/logs/dnn-{}/train'.format(model_name), sess.graph)
test_writer = tf.summary.FileWriter('/logs/dnn-{}/test'.format(model_name), sess.graph)
sess.run(init)
sess.run(tf.local_variables_initializer())
saver = tf.train.Saver()
if os.path.isfile(model_fpath):
    saved = saver.restore(sess, model_fpath)

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print("trainable params ", total_parameters)


@app.route("/", methods=["GET", "POST"])
def cat_detect():
    if request.method == 'POST':
        file_string = request.files['fileToUpload'].read()
        image = parse_image(file_string)
        image = sess.run([image])
        cnc = sess.run([cls], feed_dict={X: image})
        if cnc[0][1][0][0]:
            return "Cat!"
        else:
            return "Not cat:("
    return render_template('upload_form.html.j2')

if serve:
    app.run(host="0.0.0.0")
else:
    train(sess, train_writer, test_writer)
    saved = saver.save(sess, model_fpath)

