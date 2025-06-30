import numpy as np
import glob2
import datetime
from pathlib import Path
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
from utils.detection.association_geodesic import squarize
plt.rcParams.update({'font.size': 18})
import math

from utils.data_reading.sound_data.station import StationsCatalog
from utils.physics.sound_model.spherical_sound_model import GridSphericalSoundModel as GridSoundModel, MonthlyHomogeneousSphericalSoundModel as HomogeneousSoundModel
from utils.detection.association_geodesic import compute_candidates, update_valid_grid, update_results, load_detections, compute_grids

if __name__=="__main__":
    # paths
    CATALOG_PATH = "/media/plerolland/akoustik"
    for dataset in ["MAHY0","MAHY1","MAHY2","MAHY3","MAHY4"]:
        DETECTIONS_DIR = f"../../../../data/detection/i_TiSSNet_raw_OBS-fixed/{dataset}"
        SOUND_MODEL_PATH = f"../../../../data/sound_model"

        # Detections loading parameters
        MIN_P_TISSNET_PRIMARY = 0.5  # min probability of browsed detections
        MIN_P_TISSNET_SECONDARY = 0.3  # min probability of detections that can be associated with the browsed one
        MERGE_DELTA_S = 5  # threshold below which we consider two events should be merged
        MERGE_DELTA = datetime.timedelta(seconds=MERGE_DELTA_S)

        REQ_CLOSEST_STATIONS = 0  # The REQ_CLOSEST_STATIONS th closest stations will be required for an association to be valid

        # sound model definition
        STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated().by_dataset(dataset)
        mid_pos = np.mean([s.get_pos() for s in STATIONS], axis=0)
        SOUND_MODEL = HomogeneousSoundModel([f"{SOUND_MODEL_PATH}/min-velocities_month-{i:02d}.nc" for i in range(1, 13)],
                                            mid_pos)

        # association running parameters
        SAVE_PATH_ROOT = None  # change this to save the grids as figures, leave at None by default

        STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()
        det_files = [f for f in glob2.glob(DETECTIONS_DIR + "/*.pkl") if Path(f).is_file()]
        DETECTIONS = load_detections(det_files, STATIONS, 0.05, merge_delta=datetime.timedelta(seconds=MERGE_DELTA_S))

        DETECTIONS_DIR_NAME = DETECTIONS_DIR.split("/")[-1]

        Path(f"{DETECTIONS_DIR}/cache").mkdir(parents=True, exist_ok=True)
        DET_PATH = f"{DETECTIONS_DIR}/cache/detections_{MIN_P_TISSNET_SECONDARY}_{MERGE_DELTA_S}.pkl"
        if not Path(DET_PATH).exists():
            STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated()
            det_files = [f for f in glob2.glob(DETECTIONS_DIR + "/*.pkl") if Path(f).is_file()]
            DETECTIONS = load_detections(det_files, STATIONS, MIN_P_TISSNET_SECONDARY,
                                         merge_delta=datetime.timedelta(seconds=MERGE_DELTA_S))
            with open(DET_PATH, "wb") as f:
                pickle.dump((DETECTIONS), f)
        else:
            with open(DET_PATH, "rb") as f:
                DETECTIONS = pickle.load(f)

        idx_det = 0
        IDX_TO_DET = {}
        for idx, s in enumerate(DETECTIONS.keys()):
            s.idx = idx  # indexes to store efficiently the associations
            DETECTIONS[s] = list(DETECTIONS[s])
            for i in range(len(DETECTIONS[s])):
                DETECTIONS[s][i] = np.concatenate((DETECTIONS[s][i], [idx_det]))
                IDX_TO_DET[idx_det] = DETECTIONS[s][i]
                idx_det += 1
            DETECTIONS[s] = np.array(DETECTIONS[s])
        DETECTION_IDXS = np.array(list(range(idx_det)))

        STATIONS = [s for s in DETECTIONS.keys()]
        FIRSTS_DETECTIONS = {s: DETECTIONS[s][0, 0] for s in STATIONS}
        LASTS_DETECTIONS = {s: DETECTIONS[s][-1, 0] for s in STATIONS}

        DETECTIONS_MERGED = np.concatenate([[(det[0], det[1], det[2], s) for det in DETECTIONS[s]] for s in STATIONS])
        DETECTIONS_MERGED = DETECTIONS_MERGED[DETECTIONS_MERGED[:, 1] > MIN_P_TISSNET_PRIMARY]
        DETECTIONS_MERGED = DETECTIONS_MERGED[np.argsort(DETECTIONS_MERGED[:, 1])][::-1]

        LAT_BOUNDS = [-13.4, -12.4]
        LON_BOUNDS = [45.25, 46.25]
        GRID_SIZE = 150  # number of points along lat axis
        PICK_UNCERTAINTY = 1
        SOUND_SPEED_UNCERTAINTY = 0.25
        MAX_CLOCK_DRIFT = 0.25

        GRID_PATH = f"{DETECTIONS_DIR}/cache/grids_{LAT_BOUNDS[0]}_{LAT_BOUNDS[1]}_{LON_BOUNDS[0]}_{LON_BOUNDS[1]}_{GRID_SIZE}_{PICK_UNCERTAINTY}_{SOUND_SPEED_UNCERTAINTY}_{MAX_CLOCK_DRIFT}.pkl"

        if not Path(GRID_PATH).exists():
            GRID_TO_COORDS, TDoA, MAX_TDoA, TDoA_UNCERTAINTIES = compute_grids(LAT_BOUNDS, LON_BOUNDS, GRID_SIZE,
                                                                               SOUND_MODEL, STATIONS,
                                                                               pick_uncertainty=PICK_UNCERTAINTY,
                                                                               sound_speed_uncertainty=SOUND_SPEED_UNCERTAINTY,
                                                                               max_clock_drift=MAX_CLOCK_DRIFT)
            with open(GRID_PATH, "wb") as f:
                pickle.dump((GRID_TO_COORDS, TDoA, MAX_TDoA, TDoA_UNCERTAINTIES), f)
        else:
            with open(GRID_PATH, "rb") as f:
                GRID_TO_COORDS, TDoA, MAX_TDoA, TDoA_UNCERTAINTIES = pickle.load(f)
        GRID_TO_COORDS = np.array(GRID_TO_COORDS)

        print("starting association")
        MIN_ASSOCIATION_SIZE = 3
        ASSOCIATION_RECORD_TOLERANCE = 0
        max_reached_per_det = {det_idx: MIN_ASSOCIATION_SIZE + ASSOCIATION_RECORD_TOLERANCE for det_idx in
                               DETECTION_IDXS}

        already_examined = set()


        def process_detection(arg):
            detection, already_examined, max_reached_per_det = arg
            max_reached_per_det_modifications = {}
            local_association = []
            date1, p1, idx_det1, s1 = detection

            # list all other stations and sort them by distance from s1
            other_stations = np.array([s2 for s2 in STATIONS if s2 != s1
                                       and date1 + datetime.timedelta(days=1) > FIRSTS_DETECTIONS[s2]
                                       and date1 - datetime.timedelta(days=1) < LASTS_DETECTIONS[s2]])
            other_stations = other_stations[np.argsort([MAX_TDoA[s1][s2][date1.month - 1] for s2 in other_stations])]

            # given the detection date1 occurred on station s1, list all the detections of other stations that may be generated by the same source event
            current_association = {s1: (date1, idx_det1)}
            candidates = compute_candidates(other_stations, current_association, DETECTIONS, MAX_TDoA, MERGE_DELTA_S)

            # update the list of other stations to only include the ones having at least a candidate detection
            other_stations = [s for s in other_stations if len(candidates[s]) > 0]

            # define the recursive browsing function (that is responsible for browsing the search space of associations for s1-date1)
            def backtrack(station_index, current_association, valid_grid, associations):
                if station_index == len(other_stations):
                    return
                station = other_stations[station_index]

                candidates = compute_candidates([station], current_association, DETECTIONS, MAX_TDoA, MERGE_DELTA_S)
                probabilities = [DETECTIONS[station][idx][1] for idx in candidates[station]]
                candidates[station] = np.array(candidates[station])[np.argsort(probabilities)][::-1][:10]
                for idx in candidates[station]:
                    date, p, idx_det = DETECTIONS[station][idx]

                    if date in already_examined:
                        # the det was already browsed as main
                        continue
                    if len(other_stations) < max_reached_per_det[idx_det] - ASSOCIATION_RECORD_TOLERANCE - 1:
                        # the det already belongs to an association larger that what we could have here
                        continue

                    valid_grid_new, dg_new = update_valid_grid(current_association, valid_grid, station, date, TDoA,
                                                               TDoA_UNCERTAINTIES)

                    valid_points_new = np.argwhere(valid_grid_new)[:, 0]

                    if len(valid_points_new) > 0:
                        current_association[station] = (date, idx_det)

                        if np.all(
                                [len(current_association) >= max_reached_per_det[idx] - ASSOCIATION_RECORD_TOLERANCE for
                                 _, idx in
                                 current_association.values()]):
                            update_results(date1, current_association, valid_points_new, local_association, TDoA,
                                           TDoA_UNCERTAINTIES)
                            for _, idx in current_association.values():
                                if len(current_association) > max_reached_per_det[idx]:
                                    max_reached_per_det[idx] = len(current_association)
                                    max_reached_per_det_modifications[idx] = len(current_association)
                        backtrack(station_index + 1, current_association, valid_grid_new, associations)
                        del current_association[station]
                # also try without self
                if station_index >= REQ_CLOSEST_STATIONS:
                    backtrack(station_index + 1, current_association, valid_grid, associations)
                return

            if len(other_stations) >= max_reached_per_det[idx_det1] - ASSOCIATION_RECORD_TOLERANCE - 1:
                # we only browse other stations if we can make at least a trio
                backtrack(0, current_association, None, associations)
            return local_association, max_reached_per_det_modifications


        frac = 0.1
        n_chunks = math.ceil(1 / frac)
        chunk_size = len(DETECTIONS_MERGED) // n_chunks
        chunks = [DETECTIONS_MERGED[i * chunk_size: (i + 1) * chunk_size] for i in range(n_chunks - 1)]
        chunks.append(DETECTIONS_MERGED[9 * chunk_size:])

        # main part (note: process parallelization is a very efficient solution in case needed)
        for i in range(len(chunks)):
            fname = f"{DETECTIONS_DIR}/cache/associations_{MIN_ASSOCIATION_SIZE}_{i * frac:.02f}.pkl"
            if Path(fname).exists():
                continue
            associations = []
            for det in tqdm(chunks[i]):
                local_association, max_reached_per_det_modifications = process_detection(
                    (det, already_examined, max_reached_per_det))
                already_examined.add(det[0])
                associations.extend(local_association)
                for i, v in max_reached_per_det_modifications.items():
                    max_reached_per_det[i] = max(max_reached_per_det[i], max_reached_per_det_modifications[i])
            with open(fname, "wb") as f:
                pickle.dump(associations, f)