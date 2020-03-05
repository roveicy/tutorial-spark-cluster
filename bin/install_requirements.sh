#!/bin/sh
python3 -m pip install -U pip setuptools wheel pip-tools --upgrade-strategy eager
pip3 install -r requirements.txt
