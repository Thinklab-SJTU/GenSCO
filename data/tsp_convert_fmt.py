import numpy as np
import argparse
import os


def convert(lines: list[str]) -> dict[str, np.ndarray]:
    def _convert_single_instance(line: str) -> dict[str, np.ndarray]:
        coord_str, tour_str = line.split('output')
        coords = np.array(
            list(
                map(float, filter(lambda x: x != '', coord_str.split(' ')))
            ), dtype='float64',
        ).reshape(-1, 2)
        tour = list(
            map(int, filter(lambda x: x != '' and x != '\n', tour_str.split(' ')))
        )
        assert tour[0] == tour[-1] == 1
        tour = np.array(tour[:-1], dtype='int32')
        tour -= 1       # turn to 0-based
        return {'coords': coords, 'opt_tours': tour}

    result = list(map(_convert_single_instance, lines))
    return {
        'coords': np.stack([it['coords'] for it in result]),
        'opt_tours': np.stack([it['opt_tours'] for it in result]),
    }


def main(src_path: str, tgt_path: str, overwrite: bool = False) -> None:
    assert os.path.exists(src_path)
    if not overwrite:
        assert not os.path.exists(tgt_path)

    lines: list[str]
    with open(src_path, 'r') as f:
        lines = f.readlines()
    
    np.savez_compressed(tgt_path, **convert(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, required=True)
    parser.add_argument('--tgt-path', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    main(**vars(args))
