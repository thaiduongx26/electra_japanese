import argparse
import multiprocessing
import os
import random
import tarfile
import time
import tensorflow.compat.v1 as tf

import build_pretraining_dataset
from util import utils

from distutils.dir_util import copy_tree


def write_examples(job_id, args):
    """A single process creating and writing out pre-processed examples."""
    job_tmp_dir = os.path.join(args.data_dir, "tmp", "job_" + str(job_id))
    owt_dir = os.path.join(args.data_dir, "wiki")

    def log(*args):
        msg = " ".join(map(str, args))
        print("Job {}:".format(job_id), msg)

    log("Creating example writer")
    example_writer = build_pretraining_dataset.ExampleWriter(
        job_id=job_id,
        model_file=os.path.join(args.model_dir, "wiki-ja.model"),
        vocab_file=os.path.join(args.model_dir, "wiki-ja.vocab"),
        output_dir=os.path.join(args.model_dir, "pretrain_tfrecords"),
        max_seq_length=args.max_seq_length,
        num_jobs=args.num_processes,
        blanks_separate_docs=False,
        do_lower_case=args.do_lower_case
    )
    log("Writing tf examples")
    fnames = tf.io.gfile.listdir(owt_dir)
    fnames = [f for f in fnames if '.' not in f]
    fnames = sorted(fnames)

    fnames = [f for (i, f) in enumerate(fnames)
              if i % args.num_processes == job_id]
    random.shuffle(fnames)
    for file_no, fname in enumerate(fnames):
        print('file number : {} of job_id: {}'.format(file_no, job_id))
        utils.rmkdir(job_tmp_dir)
        copy_tree(os.path.join(owt_dir, fname), job_tmp_dir)
        list_files = tf.io.gfile.listdir(job_tmp_dir)
        list_files = [fi for fi in list_files if fi != 'all.txt']
        for file_name in list_files:
            example_writer.write_examples(os.path.join(job_tmp_dir, file_name))
    example_writer.finish()
    log("Done!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Location of data (corpus, etc).")
    parser.add_argument("--model-dir", required=True,
                        help="Location of Sentence piece model, vocab file, etc")
    parser.add_argument("--max-seq-length", default=128, type=int,
                        help="Number of tokens per example.")
    parser.add_argument("--num-processes", default=1, type=int,
                        help="Parallelize across multiple processes.")
    parser.add_argument("--do-lower-case", dest='do_lower_case',
                        action='store_true', help="Lower case input text.")
    parser.add_argument("--no-lower-case", dest='do_lower_case',
                        action='store_false', help="Don't lower case input text.")
    parser.set_defaults(do_lower_case=True)
    args = parser.parse_args()

    utils.rmkdir(os.path.join(args.model_dir, "pretrain_tfrecords"))
    if args.num_processes == 1:
        write_examples(0, args)
    else:
        jobs = []
        for i in range(args.num_processes):
            job = multiprocessing.Process(
                target=write_examples, args=(i, args))
            jobs.append(job)
            job.start()
        for job in jobs:
            job.join()


if __name__ == "__main__":
    main()
