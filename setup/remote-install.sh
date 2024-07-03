#!/bin/bash

WD=`pwd`
SCRIPT="bash ${WD}/setup/install.sh;"
HOST="sh00"
ssh ${HOST} "${SCRIPT}"
