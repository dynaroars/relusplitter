import os 
import subprocess
import sys
import logging

from .neuralsat import Neuralsat
from .abcrown import AlphaBetaCrown    
from .marabou import Marabou


verifiers = {
    "neuralsat": Neuralsat,
    "abcrown": AlphaBetaCrown,
    "marabou": Marabou
}

def init_verifier(verifier_name):
    assert verifier_name in verifiers, f"Verifier {verifier_name} not exist"
    return verifiers[verifier_name]


