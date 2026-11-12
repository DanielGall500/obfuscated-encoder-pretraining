from pathlib import Path

from tmallet import TMallet

# -- Parameters --
OUTPUT_FOLDER = "..."
OBFUS_TYPE = "pmi-filter-obfuscation"

CHUNK_SIZE = 100_000
BATCH_SIZE = 16
START_INDEX = 0
NUM_SAMPLES = 1_000_000

dataset_repo = "DanielGallagherIRE/FineWeb-Edu-10B"
dataset_config = "E1"

algorithm = "shannon"
config = {
    "threshold": [2.5, 5, 7.5, 10, 12.5],
    "bound": "lower",  # remove low MI words first, preserve signal
    "replacement_mechanism": "delete",
    "max_context_length": 128,
}

def main():
    # -- Load Text Mallet and Obfuscate --
    lang = "en"
    prefer_gpu = True
    tmallet = TMallet(lang, prefer_gpu)

    print("====")
    print(f"Obfuscating {algorithm}")
    tmallet.load_obfuscator(algorithm, config)

    save_chunks_to = Path(OUTPUT_FOLDER) / f"{OBFUS_TYPE}-obfus" / algorithm
    tmallet.obfuscate_dataset_by_chunk(
        dataset_repo=dataset_repo,
        column="text",
        column_obfuscated=f"text_{algorithm}",
        save_chunks_to_folder=save_chunks_to,
        dataset_config=dataset_config,
        dataset_split="train",
        chunk_size=CHUNK_SIZE,
        batch_size=BATCH_SIZE,
        start_index=START_INDEX,
        num_samples=NUM_SAMPLES,
    )
    print(f"Finished obfuscation of {algorithm}")
    print("====")


if __name__ == "__main__":
    main()
