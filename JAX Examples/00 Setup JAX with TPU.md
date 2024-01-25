# Setup a TPU
## Create TPU VM
```bash
gcloud config set project google.com:ml-baguette-demos
gcloud compute tpus tpu-vm create auv-tpu-vm  \
  --zone=us-central1-a \
  --accelerator-type=v3-8  \
  --version=tpu-ubuntu2204-base
```

## Install JAX
```bash
gcloud compute tpus tpu-vm ssh auv-tpu-vm \
  --zone=us-central1-a --worker=all --command="pip install \
  --upgrade 'jax[tpu]>0.3.0' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

## SSH into TPU VM and install Jupyter
```bash
gcloud compute firewall-rules create default-allow-ssh --allow tcp:22; \
gcloud compute tpus tpu-vm ssh auv-tpu-vm --zone us-central1-a -- -L 8080:localhost:8080
source ~/.profile
pip install jupyterlab matplotlib
python3 -m jupyter lab --allow-root --port=8080
```

## Connect to a localhost:8080 Jupyter session
Continue in JupyterLab Terminal
`git clone https://github.com/marsbroshok/TPU_examples.git`