from pathlib import Path
import numpy as np
import pandas as pd


def convert_to_seconds(time_str):
    minutes, seconds = time_str.split("m")
    total = 60 * int(minutes) + int(seconds.split(".")[0])
    return total


def main():
    mean_wall_time = []
    best_wall_time = []

    file_list = ["1Lbb", "InclSS3L", "staus"]
    for filename in file_list:
        file_path = Path("data").joinpath("river", f"{filename}_times.txt")
        with open(file_path, "r") as readfile:
            lines = readfile.readlines()

        times = np.array([convert_to_seconds(line) for line in lines])
        mean_wall_time.append(f"${np.mean(times)}\pm{np.std(times):.1f}$")

    single_node_time = []
    file_list = ["1Lbb", "InclSS3L", "staus"]
    for filename in file_list:
        file_path = (
            Path("data").joinpath("river").joinpath(f"{filename}_single_node_time.txt")
        )
        with open(file_path, "r") as readfile:
            time = readfile.readlines()[0]
        single_node_time.append(convert_to_seconds(time))

    river_config = {"max_blocks": 4, "nodes_per_block": 1}

    table_data = pd.DataFrame(
        dict(
            # analysis=["ATLAS SUSY 1Lbb", "ATLAS SUSY SS3L", "ATLAS SUSY staus"],
            analysis=[
                "Eur. Phys. J. C 80 (2020) 691",
                "JHEP 06 (2020) 46",
                "Phys. Rev. D 101 (2020) 032009",
            ],
            patches=[125, 76, 57],
            _spacer_A="",
            nodes_per_block=[river_config["nodes_per_block"]] * 3,
            _spacer_B="",
            max_blocks=[river_config["max_blocks"]] * 3,
            _spacer_C="",
            mean_wall_time=mean_wall_time,
            _spacer_D="",
            single_node_time=single_node_time,
        )
    )

    caption = (
        "Fit times for analyses using \pyhf{}'s NumPy backend and SciPy optimizer orchestrated with \\funcX{} on RIVER"
        + f" over {len(times)} trials compared to a single RIVER node."
        + " The reported wall fit time is the mean wall fit time of the trials."
        + " The uncertainty on the mean wall time corresponds to the standard deviation of the wall fit times."
        + " The number of worker nodes used is approximate as per-run reporting is not available."
    )
    performance_table_markdown = table_data.to_markdown(
        headers=[
            "Analysis",
            "Patches",
            "",
            "Nodes per block",
            "",
            "Max blocks",
            "",
            "Wall time (sec)",
            "",
            "Single node (sec)",
        ],
        index=False,
    )

    with open("performance_table.md", "w") as table_file:
        table_file.write(performance_table_markdown)


if __name__ == "__main__":
    main()
