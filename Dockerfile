# CPU image for the tests and for the scripts that do not need a GPU. Not tested by the author.
# For the detection on a GPU use a CUDA base image (for example pytorch/pytorch with the CUDA runtime) and run with: docker run --gpus all ...
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy opencv-python-headless scipy matplotlib
COPY . /app
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
