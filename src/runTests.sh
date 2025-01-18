#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath "$0"))
PYTHONPATH=$SCRIPT_DIR pytest test/test.py  --junitxml=reports/test_report.xml