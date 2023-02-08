import os
import subprocess
import sys

CURDIR = os.path.dirname(os.path.abspath(__file__))


FILEPATH = "data/jawiki-20200601-pages-articles-multistream.xml.bz2"
EXTRACTDIR = "data/wiki/"


def extract():
    subprocess.call(['python3', 
                    os.path.join(CURDIR,
                                 'wikiextractor', 'WikiExtractor.py'), 
                    FILEPATH, "-o={}".format(EXTRACTDIR)])


def main():
    extract()


if __name__ == "__main__":
    main()