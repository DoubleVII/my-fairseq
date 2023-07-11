import torch

import argparse

dd = {"a":100, "b":23}
p = argparse.ArgumentParser()
n=argparse.Namespace(**dd)
print(n)