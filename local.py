import gpudb

import bb_module_default
import kmllogger


def main():
    db = gpudb.GPUdb('159.69.39.8')
    logger = kmllogger.attach_log()
    in_map = {
        'city': 'New York',
        'country': 'us',
        'day_of_week': 6,
        'hour': 18,
        'group_events': 100,
        'group_members': 4000
    }
    out_map = bb_module_default.predict(in_map, db, logger)
    print(out_map)


if __name__ == '__main__':
    main()
