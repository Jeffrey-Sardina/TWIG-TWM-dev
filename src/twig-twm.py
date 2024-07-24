# TWIG imports
from load_data import do_load

# external imports
import sys

if __name__ == '__main__':
    do_load(
        datasets_to_load={
            "UMLS": ["2.1"]
        },
        test_ratio=0.1,
        valid_ratio=0.1
    )
