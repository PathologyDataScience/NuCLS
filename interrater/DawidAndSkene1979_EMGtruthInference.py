# --------------
# SOURCES:
# --------------
#
# This is code by Zheng et al, implementing the Expectation-Maximization
# based method for ground truth inference from multi-observer datasets, as
# proposed by Dawid and Skene in 1979:
#  Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm
#  Author(s): A. P. Dawid and A. M. Skene
#  Source: Journal of the Royal Statistical Society. Series C (Applied
#  Statistics), Vol. 28, No. 1 (1979), pp. 20-28
#  Published by: Wiley for the Royal Statistical Society
#  Stable URL: https://www.jstor.org/stable/2346806
#
# ** All I did was document it and refactor a bit for my own use. **
#
# Original code source:
#  https://zhydhkcws.github.io/crowd_truth_inference/index.html
#  https://github.com/zhydhkcws/crowd_truth_infer
# - Yudian Zheng, Guoliang Li, Yuanbing Li, Caihua Shan, Reynold Cheng.
#     Truth Inference in Crowdsourcing: Is the Problem Solved?
#     In VLDB 2017, Vol 10, Isuue 5, Pages 541-552, Full Paper, Present in
#     VLDB 2017, Aug 28 - Sep 1, Munich, Germany.
# - Guoliang Li, Yudian Zheng, Ju Fan, Jiannan Wang, Reynold Cheng.
#     Crowdsourced Data Management: Overview and Challenges.
#     In SIGMOD 2017, Tutorial (3 hours), May 14th - 19th, Chicago, IL, USA.
#

# TODO -- If I have time
# All of this codebase uses dicts and python loops instead of numpy arrays
# or pandas dataframes. This makes it pretty slow when the dataset is big.
# If I have time, and there's enough motivation/reason, I should refactor
# this to use matrix operations.
#

import math
import csv
import random
# import sys
# import os
from os.path import join as opj


class EM:
    def __init__(self, e2wl, w2el, label_set, initquality):
        """Initialize the EM instance.

        Parameters
        ----------
        e2wl: dict
            Indexed by question
            Each value is a list of workers and labels assigned by them.
            For example, e2wl['1'][3] -> ['4', '3']
            means that questoin '1', was answered as '3' by worker '4'
        w2el: dict
            Indexed by worker name
            Each value is a list of questions and labels assigned by the worker
            For example, w2el['4'][0] -> ['1', '3']
            means that worker '4', when asked question '1',
            assigned the label '3'.
        label_set: list
            list of unique labels in the dataset
        initquality: float
            Initial quality score assigned to each worker.
        """
        self.e2wl = e2wl
        self.w2el = w2el
        self.workers = self.w2el.keys()
        self.label_set = label_set
        self.initalquality = initquality
        self.likelihoods = []

    # E-step
    def Update_e2lpd(self):
        """Get the label probability distribution for each question/example.

        i.e. Determine our predicted label probabilities for each question,
        GIVEN our current model of:
        a. Prior probability that any particular label is the true response.
           i.e. the proportions of various true labels in the dataset.
        b. Worker qualities for each type of question (our current
        version of the worker confusion matrices).
        """
        self.e2lpd = {}

        for example, worker_label_set in self.e2wl.items():
            lpd = {}
            total_weight = 0

            for tlabel, prob in self.l2pd.items():
                # Current prob that "tlabel" is the true label for this example
                # The assumption here is that this probability has been
                # inferred from the worker labels themselves from the past
                # interation (unless we just started, then it's whatever
                # initial value we set)
                weight = prob

                # The "weight" we assign to this "prediction" that "tlabel"
                # is the true label is determined by how "reliable" are the
                # workers who derived that prediction when they:
                # a. Answer questions whose true label is "tlabel" and
                # b. Assign them the label that was used to derive our
                #    prediction (even if it was wrong)
                # In effect, we multiply the entry in each worker's confusion
                # matrix that contributed to the overall predicted probability
                # assigned to "tlabel" as being the true label for this
                # question.
                # Note: when this is the first iteration, the worker confusion
                #   matrix is just a matrix where diagonal elements are the
                #   initial quality scores and all off-diagonal elements
                #   are equal, but small values.
                for (w, label) in worker_label_set:
                    weight *= self.w2cm[w][tlabel][label]

                lpd[tlabel] = weight
                total_weight += weight

            # normalize the probabilities
            for tlabel in lpd:
                if total_weight == 0:
                    # uniform distribution
                    lpd[tlabel] = 1.0 / len(self.label_set)
                else:
                    lpd[tlabel] = lpd[tlabel] * 1.0 / total_weight

            self.e2lpd[example] = lpd

    # M-step

    def Update_l2pd(self):
        """Update prior probability that any particular label is the
        true response. i.e. the proportions of various TRUE labels in
        the dataset."""
        # reset to zeros
        for label in self.l2pd:
            self.l2pd[label] = 0

        # Get the counts for each label. Note that these are "soft" counts.
        # i.e. a question whose answer, according to our current model, is
        # '1' with a 70% probability contributes 0.7 to the count of true
        # labels of '1' in our dataset
        for _, lpd in self.e2lpd.items():
            for label in lpd:
                self.l2pd[label] += lpd[label]

        # normalize
        for label in self.l2pd:
            self.l2pd[label] *= 1.0 / len(self.e2lpd)

    def Update_w2cm(self):
        """Init the workers confusion matrix."""
        # reset to zeros
        for w in self.workers:
            for tlabel in self.label_set:
                for label in self.label_set:
                    self.w2cm[w][tlabel][label] = 0

        w2lweights = {}
        for w in self.w2el:
            w2lweights[w] = {}
            # init with zero
            for label in self.label_set:
                w2lweights[w][label] = 0
            # go through the questions answered by this worker, and for
            # each question, add the "soft" counts of their true labels
            # to this worker's count tally.
            for example, _ in self.w2el[w]:
                for label in self.label_set:
                    w2lweights[w][label] += self.e2lpd[example][label]

            for tlabel in self.label_set:

                # If the tally is zero, just re-initialize this worker
                # as was done in the self.init_w2cm() method.
                if w2lweights[w][tlabel] == 0:
                    for label in self.label_set:
                        if tlabel == label:
                            self.w2cm[w][tlabel][label] = self.initalquality
                        else:
                            self.w2cm[w][tlabel][label] = (
                                1 - self.initalquality) * 1.0 / (
                                len(self.label_set) - 1)

                    continue

                # go through the questions answered by this worker, and for
                # each question, add the "soft" count that the label placed
                # placed by this worker is the true one, and normalize
                for example, label in self.w2el[w]:
                    self.w2cm[w][tlabel][label] += self.e2lpd[example][
                       tlabel] * 1.0 / w2lweights[w][tlabel]

        return self.w2cm

    # initialization
    def Init_l2pd(self):
        """Initialize prior probability that any particular label is the
        true response. i.e. the proportions of various TRUE labels in
        the dataset."""
        # uniform probability distribution -- Assume there are equal number
        # of questions that have any particular true label.
        l2pd = {}
        for label in self.label_set:
            l2pd[label] = 1.0 / len(self.label_set)
        return l2pd

    def Init_w2cm(self):
        """Init the workers confusion matrix.

        We initialize with a matrix where self.initalquality (eg. 0.7) are
        the diagonal elements, and the off diagonal elements are all
        equal to a small value. I.e. Our initial model assumes all workers
        are right 70% of the time about any particular question.
        """
        w2cm = {}
        for worker in self.workers:
            w2cm[worker] = {}
            for tlabel in self.label_set:
                w2cm[worker][tlabel] = {}
                for label in self.label_set:
                    # case 1: diagonal elements
                    # initialize using the initial score (eg. 0.7)
                    if tlabel == label:
                        w2cm[worker][tlabel][label] = self.initalquality
                    # case 2: off-diagonal
                    # Divide the "remainder" of the quality score over
                    # the rest of the classes so everything adds to 1
                    else:
                        w2cm[worker][tlabel][label] = (
                            1 - self.initalquality) / (
                            len(self.label_set) - 1)

        return w2cm

    def Run(self, iterr=20, get_likelihood=False):
        """Run the EM algorithm.

        Parameters
        ----------
        iterr: int
            No of EM iterations
        get_likelihood: bool
            Get the likelihood progress?

        Returns
        -------
        dict
            Indexed by question, each value is a dict, where keys are
            the various possible labels, and the values are the float
            probabilities that this label is the correct one. These
            of couse add up to 1 for each question.
            For example, e2lpd['5']['3'] -> 0.99
            means that '3' is, with 99% probability, the correct answer to
            question '5'.
        dict
            Indexed by worker, this is just the confusion matrix (I checked).
            For example, w2cm['1']['3']['2'] -> 0.16
            means that, when looking at questions whose true answer is '3',
            worker '1' has incorrectly called them '2' 16% of the time.

        """
        self.l2pd = self.Init_l2pd()
        self.w2cm = self.Init_w2cm()
        if get_likelihood:
            self.likelihoods = []

        while iterr > 0:
            # E-step
            self.Update_e2lpd()

            # M-step
            self.Update_l2pd()
            self.Update_w2cm()

            # compute the likelihood
            if get_likelihood:
                self.likelihoods.append(self.computelikelihood())

            iterr -= 1

        return self.e2lpd, self.w2cm

    def computelikelihood(self):

        lh = 0

        for _, worker_label_set in self.e2wl.items():
            temp = 0
            for tlabel, prior in self.l2pd.items():
                inner = prior
                for worker, label in worker_label_set:
                    inner *= self.w2cm[worker][tlabel][label]
                temp += inner

            lh += math.log(temp)

        return lh


###################################
# The above is the EM method (a class)
# The following are several external functions
###################################


def gettruthfrompd(e2lpd):
    e2truth = {}
    for e in e2lpd:
        if isinstance(e2lpd[e], dict):
            temp = 0
            for label in e2lpd[e]:
                if temp < e2lpd[e][label]:
                    temp = e2lpd[e][label]

            candidate = []

            for label in e2lpd[e]:
                if temp == e2lpd[e][label]:
                    candidate.append(label)

            truth = random.choice(candidate)

        else:
            truth = e2lpd[e]

        e2truth[e] = truth
    return e2truth


def getaccuracy(truthfile, e2lpd):
    e2truth = {}
    f = open(truthfile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, truth = line
        e2truth[example] = truth

    tcount = 0
    count = 0

    for e in e2lpd:

        if e not in e2truth:
            continue

        temp = 0
        for label in e2lpd[e]:
            if temp < e2lpd[e][label]:
                temp = e2lpd[e][label]

        candidate = []

        for label in e2lpd[e]:
            if temp == e2lpd[e][label]:
                candidate.append(label)

        truth = random.choice(candidate)

        count += 1

        if truth == e2truth[e]:
            tcount += 1

    return tcount * 1.0 / count


def gete2wlandw2el(datafile):
    """Get dict inputs for the EM algorithm.

    Parameters
    ----------
    datafile: str
        path to datafile. The data file itself must be a csv file with
        3 columns: question, worker, answer
        and the first row is the column names.

    Returns
    -------
    dict
        Indexed by question
        Each value is a list of workers and labels assigned by them.
        For example, e2wl['1'][3] -> ['4', '3']
        means that questoin '1', was answered as '3' by worker '4'
    dict
        Indexed by worker name
        Each value is a list of questions and labels assigned by the worker.
        For example, w2el['4'][0] -> ['1', '3']
        means that worker '4', when asked question '1', assigned the label '3'.
    list
        list of unique labels in the dataset

    """
    e2wl = {}
    w2el = {}
    label_set = []

    f = open(datafile, 'r')
    reader = csv.reader(f)
    next(reader)

    for line in reader:
        example, worker, label = line
        if example not in e2wl:
            e2wl[example] = []
        e2wl[example].append([worker, label])

        if worker not in w2el:
            w2el[worker] = []
        w2el[worker].append([example, label])

        if label not in label_set:
            label_set.append(label)

    return e2wl, w2el, label_set


def gete2wlandw2el_fromdf(df, missingval='*'):
    """Get dict inputs for the EM algorithm from a pandas dataframe.

    Parameters
    ----------
    df: pandas.DataFrame
        dataframe, indexed by the question, and where column names
        represent the various workers. Entries are the labels assigned
        by workers. For example
        Q   W1  W2  W3  W4 . .
        Q1  "a" "a" "b" "a"
        Q2  "b" "a" "b" "b"
        Q3  "a" "b" "b" "a"
        .
        .
        Note that missing values should be coded as missingval (next param).
        The assumption here, if you'd like to use this function as opposed
        to reading the linear file as in gete2wlandw2el() is that each worker
        only answers the question once (if they do answer it).

    missingval: str or other
        value to denote that a worker sis not answer this question


    Returns
    -------
    dict
        Indexed by example (question)
        Each value is a list of workers and labels assigned by them.
        For example, e2wl['1'][3] -> ['4', '3']
        means that question '1', was answered as '3' by worker '4'
    dict
        Indexed by worker name
        Each value is a list of questions and labels assigned by the worker.
        For example, w2el['4'][0] -> ['1', '3']
        means that worker '4', when asked question '1', assigned the label '3'.
    list
        list of unique labels in the dataset

    """
    e2wl = dict()
    w2el = {w: [] for w in df.columns}
    label_set = []
    for example, row in df.iterrows():
        row = dict(row)
        e2wl[example] = []
        for w, l in row.items():
            if l != missingval:  # noqa
                e2wl[example].append([w, l])
                w2el[w].append([example, l])
                if l not in label_set:
                    label_set.append(l)

    return e2wl, w2el, label_set


if __name__ == "__main__":

    # datafile = sys.argv[1]
    DATAPATH = "/home/mtageld/Desktop/tmp/btz083_supplementary_information/datasets/s4_Dog data/"  # noqa
    datafile = opj(DATAPATH, 'answer.csv')

    # generate structures to pass into EM
    e2wl, w2el, label_set = gete2wlandw2el(datafile)

    # Run D&S EM algorithm
    iterations = 20  # EM iteration number
    initquality = 0.7
    em = EM(
        e2wl=e2wl, w2el=w2el, label_set=label_set, initquality=initquality)
    e2lpd, w2cm = em.Run(iterr=iterations)
    e2truth = gettruthfrompd(e2lpd)

    # print(w2cm)
    # print(e2lpd)
