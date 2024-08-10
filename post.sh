

set -ex

# : "${IMG_ID:="4d8d3780-e786-400f-b2fd-62eed728ba8c"}"
# : "${IMG_ID:="0e550d4a-8479-46c6-9f5a-e65d7642f3c8"}"
# : "${IMG_ID:="ca996e1f-cdca-47b5-84d4-9e33bcef1bf9"}"
: "${IMG_ID:="0e63cc86-cab1-4098-a80a-c94d90063955"}"


curl -X POST -H "Content-Type: application/json" -d '{"file_name": "'"$IMG_ID"'"}' http://localhost:5000/predict