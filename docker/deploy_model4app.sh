

WORKDIR="$HOME/workspace/Perspective-Regressor"
DATADIR="$HOME/data/datasets"

docker run \
    -p 5000:5000 \
    --user $USER \
    --shm-size=8g \
    --gpus all \
    --mount type=bind,source=$WORKDIR,target=/workspace \
    --mount type=bind,source=$DATADIR,target=/workspace/data \
    -e IS_LOCAL_RUN=1 \
    -it \
    --name=perspective-regressor \
    perspective-regressor-12_4_1:v1.0