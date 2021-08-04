import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--k",
    type=int,
    default=0,
    help="Smoothing parameter for computing token counts"
)


if __name__ == "__main__":
    args = parser.parse_args()
