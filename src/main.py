import os
import sys
import time
import utils
import numpy as np
from model import Model
import tensorflow as tf
from parsers import Parser
from config import Config
from network import Network
from dataset import DataSet
from eval_performance import evaluate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#tensorflow只显示error


def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(0)
    with tf.variable_scope('Model', reuse=None) as scope:
        model = Model(config)
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    if config.load == True:
        print("=> Loading model from checkpoint")
        model.saver.restore(sess, config.ckptdir_path)
    else:
        print("=> No model loaded from checkpoint")
    return model, sess

def train(model, sess):
    with sess:
        summary_writers = model.init_summaries(sess)
        loss_dict = model.fit(sess, summary_writers)
    return loss_dict

def test(model, sess):
    with sess:
        loss_dict = model.run_eval(sess, 'test')
    return loss_dict

if __name__ == '__main__' :
    args = Parser().get_parser().parse_args()
    config = Config(args)
    model, sess = init_model(config)

    if config.load == True:
        print("\033[92m=>\033[0m Testing Model")
        test_loss, test_metrics = train(model, sess)
        output = "=> Test Loss : {}".format(test_loss)
    else:
        print("\033[92m=>\033[0m Training Model")
        loss_dict = train(model, sess)
        test_metrics = loss_dict['test_metrics']
        output = "=> Best Train Loss : {}, Test Loss : {}".format(loss_dict["train_loss"], loss_dict["test_loss"])


    # output += "\n=> Test : Coverage = {}, Average Precision = {}, Micro Precision = {}, Micro Recall = {}, Micro F Score = {}".format(metrics['coverage'], metrics['average_precision'], metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'])
    # output += "\n=> Test : Hamming Loss = {}, Macro Precision = {}, Macro Recall = {}, Macro F Score = {}".format(metrics['hamming_loss'], metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'])
    output += "\n=>Test :Micro F Score = {}".format(test_metrics['micro_f1'])
    output += "\n=>Test :Macro F Score = {}".format(test_metrics['macro_f1'])

    with open("{}test_log.log".format(config.project_prefix_path), "a+") as f:
        f.write(output)
    print("\033[1m\033[92m{}\033[0m\033[0m".format(output))
