
import re
import sys
import pandas as pd
import numpy as np
import os.path
import pytest

from click.testing import CliRunner
from main import cli, train, predict, process

def test_training():
    runner = CliRunner()
    tmp = runner.invoke(train, ['--data', '../data/testing_data.csv', '--model', 'test.pkl', '--split', 0.2])
    assert os.path.isfile("test_model.pkl")
    os.remove("test_model.pkl")

def test_predicting():
    runner = CliRunner()
    tmp = runner.invoke(predict, ['--model', 'model.pkl', '--data', '../data/testing_data.csv'])
    assert tmp.exit_code == 0

def test_process():
    text_a = 'I hate pineapples. They are disgusting!'
    text_b = 'In my opinion, you are wrong! Pineapples are incredible!'
    correct_a = 'hate pineapple disgusting'
    correct_b = 'opinion wrong pineapple incredible'
    assert process(text_a) == correct_a
    assert process(text_b) == correct_b
