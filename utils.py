import numpy as np
import os
import cv2
from PIL import Image

from pulser import Pulse, Sequence, Register
from pulser.simulation import Simulation
from pulser.devices import Chadoq2
import qutip
from qutip import expect
from itertools import product


def load_image(filename) :
    img = Image.open(filename)
    img.load()
    data = np.asarray(img)
    return data


def load_dataset(num_samples=200):
    neg_images = []
    pos_images = []

    for file in os.listdir('dataset/Negative')[0:num_samples]:
        neg_images.append(load_image('dataset/Negative/' + file))

    for file in os.listdir('dataset/Positive')[0:num_samples]:
        pos_images.append(load_image('dataset/Positive/' + file))

    return neg_images, pos_images

def preprocess_images(images, output_shape=(3,3)):
    images_rescaled = np.array([cv2.resize(np.sum(image/255, axis=2)/3, output_shape) for image in images])
    images_rescaled[np.abs(images_rescaled)<.2] = 0
    return images_rescaled


def load_image_in_seq(seq, image):
    size = image.shape[0]
    for i, j in product(range(size), repeat=2):
        index = i*size + j
        if np.abs(image[i, j])>0:
            if image[i, j] > 0:
                phase = np.pi
            else:
                phase = -np.pi
            seq.target(index, 'ch1')
            pulse = Pulse.ConstantPulse(100*np.abs(image[i, j]), 1., 0., phase)
            seq.add(pulse, 'ch1')


def magnetization(j, total_sites):
    prod = [qutip.qeye(2) for _ in range(total_sites)]
    prod[j] = qutip.sigmaz()
    return qutip.tensor(prod)


def quantum_evol(image, times, pulses, distance_reg=7.):

    size = image.shape[0]

    reg = Register.square(size, distance_reg)
    seq = Sequence(reg, Chadoq2)
    seq.declare_channel('ch0','rydberg_global')
    seq.declare_channel('ch1','rydberg_local')

    load_image_in_seq(seq, image)

    for t, p in zip(times, pulses):
        pulse_1 = Pulse.ConstantPulse(1000*t, 1., 0., 0)
        pulse_2 = Pulse.ConstantPulse(1000*p, 0., 0., 0)

        seq.add(pulse_1, 'ch0')
        seq.add(pulse_2, 'ch0')

    simul = Simulation(seq, sampling_rate=.01)
    results = simul.run()
    state = results.get_final_state()

    return state

def sigmoid(x):
    return np.exp(x)/(1 + np.exp(x))

def probas_pred(images, times, pulses, distance_reg=7.):

    size = images[0].shape[0]
    magn_list = [magnetization(j, size**2) for j in range(size**2)]

    states = [quantum_evol(image, times, pulses, distance_reg) for image in images]
    logits_pred = [np.mean(expect(magn_list, state)) for state in states]

    return sigmoid(logits_pred)

def cost(images, labels, times, pulses, distance_reg=7.):
    preds = probas_pred(images, times, pulses, distance_reg)
    return  - np.sum(labels * np.log(preds + 1e-6) + (1 - labels) * np.log(1- preds + 1e-6))/ len(preds)

def labels_pred(probas):
    return (probas > .5)*1

def accuracy(preds, targets):
    return np.sum((preds==targets)*1)/len(preds)




