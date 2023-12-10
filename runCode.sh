#!/bin/bash

conda create -n qrl python=3.10

conda activate qrl

pip install -r requirements.txt

echo "finished running file dependencies"
