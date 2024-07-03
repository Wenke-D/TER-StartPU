set -e

make -C build

DATA_SIZE=1024
FACTOR=8

export STARPU_PROFILE=1 STARPU_WORKER_STATS=1 STARPU_BUS_STATS=1
echo 'eager'
export STARPU_SCHED=eager
./build/gemm --m ${DATA_SIZE} --n ${DATA_SIZE} --k ${DATA_SIZE} --factor ${FACTOR}

echo 'random'
export STARPU_SCHED=random
./build/gemm --m ${DATA_SIZE} --n ${DATA_SIZE} --k ${DATA_SIZE} --factor ${FACTOR}

echo 'lws'
export STARPU_SCHED=lws
./build/gemm --m ${DATA_SIZE} --n ${DATA_SIZE} --k ${DATA_SIZE} --factor ${FACTOR}

echo 'ws'
export STARPU_SCHED=ws
./build/gemm --m ${DATA_SIZE} --n ${DATA_SIZE} --k ${DATA_SIZE} --factor ${FACTOR}

echo 'dm'
export STARPU_SCHED=dm
./build/gemm --m ${DATA_SIZE} --n ${DATA_SIZE} --k ${DATA_SIZE} --factor ${FACTOR}

echo 'dmda'
export STARPU_SCHED=dmda
./build/gemm --m ${DATA_SIZE} --n ${DATA_SIZE} --k ${DATA_SIZE} --factor ${FACTOR}