#!/bin/bash


ENV_NAME="flexselect"


if conda env list | grep -q "$ENV_NAME"; then
    echo "env '$ENV_NAME' already exists, skipping..."
else
    echo "creating env '$ENV_NAME'..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME" || {
    echo "cannot activate '$ENV_NAME', check your conda source!!!"
    exit 1
}
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
echo "virtual env setup successfully, activate it by:"
echo "conda activate $ENV_NAME"