#!/bin/bash

# Create a logs directory if it doesn't exist
mkdir -p logs

# Loop to run the command 10 times
for i in {1..10}
do
    echo "Running iteration $i..."
    # Run the command and store the output in a log file
    poetry run python src/baseline.py > logs/run_$i.log 2>&1
done

echo "All iterations completed. Logs are saved in the 'logs' directory."
