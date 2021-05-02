#!/bin/bash
fileid="1x8eQttn7x2TtqMbKpAzuPc31-zY9sqZr"
filename="m39v1.tar"
curl -k -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -k -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm -r cookie