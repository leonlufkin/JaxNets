#!/bin/bash

PROJECT_NAME=$1
DIR_NAME=$2
TIME_MIN=$3

echo "Pulling recent weights from HPC"
echo "Project Name: $PROJECT_NAME"
echo "Directory Name: $DIR_NAME"
echo "Time in minutes: $TIME_MIN"

# Run the SSH and SCP commands
echo "find /ceph/scratch/leonl/results/weights -type f -mmin -$TIME_MIN | xargs tar -czvf /ceph/scratch/leonl/results/recent.tar.gz"
ssh leonl@hpc "find /ceph/scratch/leonl/results/weights -type f -mmin -$TIME_MIN | xargs tar -czvf /ceph/scratch/leonl/results/recent.tar.gz"
scp leonl@hpc:/ceph/scratch/leonl/results/recent.tar.gz /Users/leonlufkin/Documents/GitHub/$PROJECT_NAME/$DIR_NAME/results/weights
tar -xzvf /Users/leonlufkin/Documents/GitHub/$PROJECT_NAME/$DIR_NAME/results/weights/recent.tar.gz -C /Users/leonlufkin/Documents/GitHub/$PROJECT_NAME/$DIR_NAME/results/weights/
mv /Users/leonlufkin/Documents/GitHub/$PROJECT_NAME/$DIR_NAME/results/weights/ceph/scratch/leonl/results/weights/* /Users/leonlufkin/Documents/GitHub/$PROJECT_NAME/$DIR_NAME/results/weights/
rm -r /Users/leonlufkin/Documents/GitHub/$PROJECT_NAME/$DIR_NAME/results/weights/ceph
rm /Users/leonlufkin/Documents/GitHub/$PROJECT_NAME/$DIR_NAME/results/weights/recent.tar.gz

