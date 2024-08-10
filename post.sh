

set -ex

: "${IMG_ID:="4d8d3780-e786-400f-b2fd-62eed728ba8c"}"

curl -X POST -H "Content-Type: application/json" -d '{"file_name": "'"$IMG_ID"'"}' http://localhost:5000/predict