import os
from enum import Enum

from tensorflow.compat.v1 import variable_scope, placeholder
from tensorflow.compat.v1.summary import image, FileWriter, scalar


class Phase(Enum):
    TRAIN = 'TRAIN'
    VAL = 'VAL'
    TEST = 'TEST'


class Logger:
    def __init__(self, sess, summary_dir):
        self.sess = sess
        self.summary_placeholders = {}
        self.summary_ops = {}
        self.train_summary_writer = FileWriter(os.path.join(summary_dir, Phase.TRAIN.value), self.sess.graph)
        self.val_summary_writer = FileWriter(os.path.join(summary_dir, Phase.VAL.value))
        self.test_summary_writer = FileWriter(os.path.join(summary_dir, Phase.TEST.value))

    # it can summarize scalars and images.
    def summarize(self, step, phase: Phase = Phase.TRAIN, scope="", summaries_dict=None):
        """
        :param step: the step of the summary
        :param phase: use the train summary writer or the test one
        :param scope: variable scope
        :param summaries_dict: the dict of the summaries values (tag,value)
        :return:
        """
        if phase == Phase.TRAIN:
            summary_writer = self.train_summary_writer
        elif phase == Phase.VAL:
            summary_writer = self.val_summary_writer
        elif phase == Phase.TEST:
            summary_writer = self.test_summary_writer
        else:
            raise ValueError(f'Illegal Argument for summarizer: {phase.value}')

        with variable_scope(scope):

            if summaries_dict is not None:
                summary_list = []
                for tag, value in summaries_dict.items():
                    if value is None:
                        continue
                    if tag not in self.summary_ops:
                        if len(value.shape) <= 1:
                            self.summary_placeholders[tag] = placeholder('float32', value.shape, name=tag)
                            self.summary_ops[tag] = scalar(tag, self.summary_placeholders[tag])
                        else:
                            self.summary_placeholders[tag] = placeholder('float32', [None] + list(value.shape[1:]), name=tag)
                            self.summary_ops[tag] = image(tag, self.summary_placeholders[tag], max_outputs=100)

                    summary_list.append(self.sess.run(self.summary_ops[tag], {self.summary_placeholders[tag]: value}))

                for summary in summary_list:
                    summary_writer.add_summary(summary, step)
                summary_writer.flush()
