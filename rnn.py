# -*- coding: utf-8 -*-
from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
import sys
import time
import math

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

MAXLEN_Q = 7
MAXLEN_A = 6

# All the numbers, plus sign and space for padding.
chars = '0123456789+-*/%# '
ctable = CharacterTable(chars)

def gen_number(min_digits=1, max_digits=3):
	return int(''.join(np.random.choice(list('0123456789')) for i in range(np.random.randint(min_digits, max_digits + 1))))

def gen_add_problem(min_digits=1, max_digits=3):
	a = gen_number(min_digits, max_digits)
	b = gen_number(min_digits, max_digits)
	query = '{}+{}'.format(a, b)
	ans = '{}'.format(int(a) + int(b))
	return query, ans

def gen_sub_problem(size):
	a = gen_number(1, size)
	b = gen_number(1, size)
	query = '{}-{}'.format(a, b)
	ans = '{}'.format(int(a) - int(b))
	return query, ans

def gen_add_sub_problem(size):
	return np.random.choice([gen_add_problem, gen_sub_problem])(size)

def gen_mul_problem(size):
	a = gen_number(1, size-1)
	b = gen_number(1, size-1)
	query = '{}*{}'.format(a, b)
	ans = '{}'.format(int(a) * int(b))
	return query, ans

def gen_div_problem(size):
	a = gen_number(1, size+1)
	b = gen_number(1, size-1)
	query = '{}/{}'.format(a, b)
	ans = '#' if int(b) == 0 else '{}'.format(int(a) / int(b))
	return query, ans

def gen_mod_problem(size):
	a = gen_number(1, size+1)
	b = gen_number(1, size)
	query = '{}%{}'.format(a, b)
	ans = '#' if int(b) == 0 else '{}'.format(int(a) % int(b))
	return query, ans

def gen_math_problem(size):
	return np.random.choice([gen_add_problem, gen_sub_problem, gen_mul_problem, gen_div_problem, gen_mod_problem])(size)

def check(label, s):
	for c in s:
		if c not in chars:
			print('Invalid {}: {}' % (label, s))
			sys.exit(1)


def gen_dataset(data_size, problem_size, fn):
	print('Generating data...')
	questions = []
	answers = []
	seen = set()
	start_time = time.time()
	i = 0
	while len(questions) < data_size:
		i += 1
		if i == 10000:
			i = 0
			elapsed_time = time.time() - start_time
			if elapsed_time >= 2:
				start_time = time.time()
				print('dataset %s / %s' % (len(questions), data_size))

		query, answer = fn(problem_size)
		if len(query) > MAXLEN_Q or len(answer) > MAXLEN_A or query in seen:
			continue
		check('query', query)
		check('answer', answer)
		seen.add(query)
		questions.append(query + ' ' * (MAXLEN_Q - len(query)))
		answers.append(answer + ' ' * (MAXLEN_A - len(answer)))
	return questions, answers	

def vectorize_and_shuffle(questions, answers):
	print('Vectorization...')
	x = np.zeros((len(questions), MAXLEN_Q, len(chars)), dtype=np.bool)
	y = np.zeros((len(questions), MAXLEN_A, len(chars)), dtype=np.bool)
	for i, sentence in enumerate(questions):
	    x[i] = ctable.encode(sentence, MAXLEN_Q)
	for i, sentence in enumerate(answers):
	    y[i] = ctable.encode(sentence, MAXLEN_A)
	# Shuffle (x, y) in unison as the later parts of x will almost all be larger
	# digits.
	indices = np.arange(len(y))
	np.random.shuffle(indices)
	x = x[indices]
	y = y[indices]
	return x, y


# Try replacing GRU, or SimpleRNN.
RNN = layers.LSTM
HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

print('Build model...')
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE.
# Note: In a situation where your input sequences have a variable length,
# use input_shape=(None, num_feature).
model.add(RNN(HIDDEN_SIZE, input_shape=(MAXLEN_Q, len(chars))))
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
model.add(layers.RepeatVector(MAXLEN_A))
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    # By setting return_sequences to True, return not only the last output but
    # all the outputs so far in the form of (num_samples, timesteps,
    # output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(len(chars))))
model.add(layers.Activation('softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model each generation and show predictions against the validation
# dataset.
for iteration in range(0, 10000):
	poor_questions = []
	poor_answers = []
	if iteration != 0:
		print('Computing examples to retrain...')
		print(x_train.shape)
		preds = model.predict_classes(x_train, verbose=0)		
		print(preds.shape)
		seen = set()
		n = 0
		d = 0
		for i in xrange(split_at):
			d += 1
			correct = y_train[i].argmax(axis=-1)
			guess = preds[i]
			if np.any(correct != guess):
				n += 1
				if questions[i] in seen:
					continue
				seen.add(questions[i])
				poor_questions.append(questions[i])
				poor_answers.append(answers[i])
		print('Keeping {} unique questions from train set (train acc {})'.format(len(poor_questions), (d-n)*1.0/d))
		n = 0
		d = 0
		for i in xrange(100000-split_at):
			d += 1
			correct = y_train[i].argmax(axis=-1)
			guess = preds[i]
			if np.any(correct != guess):
				n += 1
				if questions[i] in seen:
					continue
				seen.add(questions[i])
				poor_questions.append(questions[i])
				poor_answers.append(answers[i])
		print('(val acc {})'.format((d-n)*1.0/d))
		reruns = split_at // 3
		poor = len(poor_questions)
		for j in xrange(4):
			for i in xrange(poor):
				poor_questions.append(poor_questions[i])
				poor_answers.append(poor_answers[i])
				if len(poor_questions) >= reruns:
					break
		print("before {} after {}".format(poor, len(poor_questions)))
			
	print('-' * 10)
	questions, answers = gen_dataset(data_size=100000-len(poor_questions), problem_size=3, fn=gen_math_problem)
	questions += poor_questions
	answers += poor_answers
	x, y = vectorize_and_shuffle(questions, answers)

	# Explicitly set apart 10% for validation data that we never train over.
	split_at = len(x) - len(x) // 10
	x_train, x_val = x[:split_at], x[split_at:]
	y_train, y_val = y[:split_at], y[split_at:]

	print('Training Data: {} {}'.format(x_train.shape, y_train.shape))
	print('Validation Data: {} {}'.format(x_val.shape, y_val.shape))

	print('-' * 10)
	print('Iteration', iteration)
	results = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=4, validation_data=(x_val, y_val))
	print('10 random')
	for i in range(10):
		ind = np.random.randint(0, len(x_val))
		rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
		preds = model.predict_classes(rowx, verbose=0)
		q = ctable.decode(rowx[0])
		correct = ctable.decode(rowy[0])
		guess = ctable.decode(preds[0], calc_argmax=False)
		print('Q', q, end=' ')
		print('T', correct, end=' ')
		if correct == guess:
			print(colors.ok + '☑' + colors.close, end=' ')
		else:
			print(colors.fail + '☒' + colors.close, end=' ')
		print(guess)
	print('10 fails')
	i = 0
	j = 0
	while i < 10 and j < len(x_val) / 2:
		ind = np.random.randint(0, len(x_val))
		rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]
		preds = model.predict_classes(rowx, verbose=0)
		q = ctable.decode(rowx[0])
		correct = ctable.decode(rowy[0])
		guess = ctable.decode(preds[0], calc_argmax=False)
		if correct == guess:
			j += 1
			continue
		print('Q', q, end=' ')
		print('T', correct, end=' ')
		print(colors.fail + '☒' + colors.close, end=' ')
		print(guess)
		i += 1
