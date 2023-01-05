# Modular Component-based Quantum Circuit Synthesis

This is repository for implementation of paper "Modular Component-based Quantum Circuit Synthesis".
Our algorithm produce synthesis of quantum circuit, given by user-provided in/output spec and component gates.


## Setup

We present two option for setup : (i) by building Docker image or (ii) by installing required python libraries via \verb*|pip|. Among two options, we recommend using Docker so that user's local environment is separated. 


### Via Docker
We packaged our artifact in Docker.
Following command will build the Dockerfile, in few minutes (<10 min).

```
docker build -t qsyn -f Dockerfile .
```


Note that `-t` option denotes tag name for build image.
After the build, run docker image with following command:

```
docker run -ti qsyn
```

### Via `pip` install


Environment for our program also can be prepared, only by installing required python libraries listed in `requirements.txt` via `pip`, instead using Docker.
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

In this section, we give step-by-step instruction to reproduce our result of synthesis.
We also explain how to give new input to our program (i.e., new synthesis problem).



### Running Single Synthesis Problem


Each run of single benchmark is done by following command :

```
python3 qsyn/run_single.py --benchmark [Benchmark]  --mode [Mode]
```

Two option should be specified in the command:
- For `[Benchmark]`, insert name of one of `.json` files under `/benchmarks/` folder, which is one of following items
```
three_superpose , M_valued , GHZ_from_100 , GHZ_by_iSWAP, GHZ_by_QFT , GHZ_Game , W_orthog , W_phased , W_four ,  cluster , bit_measure, flip 	, teleportation 	, indexed_bell 	,toffoli_by_sqrt_X	, QFT , draper
```
Note that name is aligned with benchmarks in Table~3.
	
- For \verb|[Mode]|, insert name of  synthesis algorithm mode. The mode is one of following four (refer Section~5.1 of our paper for more detail) :
	- `Ours` :  Main setting of our modular synthesis
	- `Ours_no_prune` : Ours with no module-level pruning.
	- `Base`  : Naive-BFS-based synthesis algorithm with basic pruning.
	-  `Base_no_prune` : `Base`  wihtout pruning. This setting is included to check that  `Base` is not very weak.

For example, following command runs synthesis problem `GHZ_from_100` with our main algorithm mode `Ours`.

```
python3 qsyn/run_single.py --benchmark GHZ_from_100  --mode Ours 
```


### Reproducing Table~3 (in our paepr)



Whole benchmark run for reproduction of Table~3 is done by running `run_all.py` file.
Following command will run whole benchmarks under `/benchmarks` for each mode of synthesis:
```
python3 qsyn/run_all.py  --mode [Mode] 
```
where `[Mode]` specifies setting for synthesis, as one of `Ours`, `Ours_no_prune`, `Base`, `Base_no_prune`.

### Giving New Synthesis Problem


### Reproducing Qiskit's Transpile

Following command will show those transpiled circuits in printed text:
```
python3 transpile_qc.py
```

## Contact

Kang, Chan Gu (e-mail : changukang@korea.ac.kr)