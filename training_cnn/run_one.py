import sys
from create_input_batches import run_generation

window_size = int(sys.argv[1])
cutoff = float(sys.argv[2])

print(f"Running window_size={window_size}, cutoff={cutoff}")

#this function generates the input batches
run_generation(
    window_size=window_size,
    methylation_cutoff=cutoff,
    random_sequences=False,
    supreme_test=False,
    normalize=False
)
