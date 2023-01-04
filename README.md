# Modular Component-based Quantum Circuit Synthesis

This is repository for implementation of paper "Modular Component-based Quantum Circuit Synthesis".
Our algorithm produce synthesis of quantum circuit, given by user-provided in/output spec and component gates.


## Setup

We present two option for setup : (i) by building Docker image or (ii) by installing required python libraries via \verb*|pip|. Among two options, we recommend using Docker so that user's local environment is separated. 


### Via Docker
We packaged our artifact in Docker.
Following command will build the Dockerfile, in few minutes ($<$10 min).

```
docker build -t qsyn -f Dockerfile .
```


Note that ``-t` option denotes tag name for build image.
After the build, run docker image with following command:

```
docker run -ti qsyn
```

### Via `pip` install


Environment for our program also can be prepared, only by installing required python libraries listed in \verb|requirements.txt| via \verb|pip|, instead using Docker.
Installation will be done by following command :

```
pip3 install -r requirements.txt
```



## Verifying Install

If the environment is properly prepared, under entry directory(`/workdir/` if using docker) we will be able to run following command (which synthesizes our working example `GHZ_from_100` appeared Section 3 in our paper).

```
python3 qsyn/run_single.py --benchmark GHZ_from_100  --mode Ours
```


## How to Run Our Program

### Running Synthesis Problem

### Reproducing Main Experiment Reuslt

