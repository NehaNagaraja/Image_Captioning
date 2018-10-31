from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import os
import tracemalloc
import resource
import time
import psutil
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from caption_generator import CaptionGenerator
from model import ShowAndTellModel
from vocabulary import Vocabulary

tracemalloc.start()
start_time = time.time()
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_path", "", "Model graph def path")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_files", "",
                       "File pattern or comma-separated list of file patterns of image files.")

logging.basicConfig(level=logging.INFO) #

logger = logging.getLogger(__name__)


def main(_):
    model = ShowAndTellModel(FLAGS.model_path)
    vocab = Vocabulary(FLAGS.vocab_file)
    filenames = _load_filenames()
    #print (filenames)

    generator = CaptionGenerator(model, vocab)

    for filename in filenames:
        with tf.gfile.GFile(filename, "rb") as f:
            image = f.read()
        captions = generator.beam_search(image)
        print("\nCaptions for image %s in decreasing order of their probabilities:" % os.path.basename(filename))
        for i, caption in enumerate(captions):
            # Ignore begin and end tokens <S> and </S>.
            sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
            sentence = " ".join(sentence)
            print("  %d) %s" % (i+1, sentence))
            print("\tLog probability of the above caption = %f" % math.exp(caption.logprob))

    #print("Current: %d, Peak %d" % tracemalloc.get_traced_memory())
    print ("\nMemory usage :")
    print (psutil.virtual_memory())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    """print("max cpu", "=>", resource.getrlimit(resource.RLIMIT_CPU))
    print("max data", "=>", resource.getrlimit(resource.RLIMIT_DATA))
    print("max processes", "=>", resource.getrlimit(resource.RLIMIT_NPROC))
    print("page size", "=>", resource.getpagesize())"""
    print ("\nResources used by the current process and its children:")
    for name, desc in [
        ('ru_utime', 'User time'),
        ('ru_stime', 'System time'),
        ('ru_maxrss', 'Max. Resident Set Size'),
        ('ru_ixrss', 'Shared Memory Size'),
        ('ru_idrss', 'Unshared Memory Size'),
        ('ru_isrss', 'Stack Size'),
        ('ru_inblock', 'Block inputs'),
        ('ru_oublock', 'Block outputs'),
        ]:
        print ('%-25s (%-10s) = %s' % (desc, name, getattr(usage, name)))
    end_time = time.time()
    print ("\nSystem CPU times:")
    print(psutil.cpu_times())
    print ("\nRun time of the application = %.2f ms" % (end_time-start_time))

def _load_filenames():
    filenames = []
    for file_pattern in FLAGS.input_files.split(","):
        filenames.extend(tf.gfile.Glob(file_pattern))
    #logger.info("Running caption generation on %d files matching %s",
                #len(filenames), FLAGS.input_files)
    print ("Running caption generation on %d files : %s" %
           (len(filenames), FLAGS.input_files))
    return filenames


if __name__ == "__main__":
    tf.app.run()