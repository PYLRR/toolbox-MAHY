import numpy as np
import glob2
import datetime
from pathlib import Path
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 18})
import math
import pandas as pd

from utils.data_reading.sound_data.station import StationsCatalog
from utils.detection.association_geodesic import squarize
from utils.detection.association_geodesic_3D import compute_candidates, update_valid_grid, update_results, load_detections, compute_grids

if __name__=="__main__":
    # paths
    COSTS = True
    CATALOG_PATH = "/media/plerolland/akoustik"
    for dataset in ["MAHY0","MAHY1","MAHY2","MAHY3","MAHY4"]:
        print(f"Processing {dataset}")
        DETECTIONS_DIR = f"../../../../data/detection/TiSSNet_Pn_raw_OBS-fixed/{dataset}"
        SOUND_MODEL_PATH = f"../../../../data/sound_model"

        # Detections loading parameters
        MIN_P_TISSNET_PRIMARY = 0.3  # min probability of browsed detections
        MIN_P_TISSNET_SECONDARY = 0.15  # min probability of detections that can be associated with the browsed one
        MERGE_DELTA_S = 5 # threshold below which we consider two events should be merged
        MERGE_DELTA = datetime.timedelta(seconds=MERGE_DELTA_S)

        REQ_CLOSEST_STATIONS = 0  # The REQ_CLOSEST_STATIONS th closest stations will be required for an association to be valid

        # sound model definition
        STATIONS = StationsCatalog(CATALOG_PATH).filter_out_undated().filter_out_unlocated().by_dataset(dataset)
        seismic_paths = glob2.glob("../../../../data/MAHY/loc_3D/*.npz")

        # association running parameters
        SAVE_PATH_ROOT = None  # change this to save the grids as figures, leave at None by default

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
        for i in range(len(STATIONS)):
            STATIONS[i].idx = i
        FIRSTS_DETECTIONS = {s: DETECTIONS[s][0, 0] for s in STATIONS}
        LASTS_DETECTIONS = {s: DETECTIONS[s][-1, 0] for s in STATIONS}

        DETECTIONS_MERGED = np.concatenate([[(det[0], det[1], det[2], s) for det in DETECTIONS[s]] for s in STATIONS])
        DETECTIONS_MERGED = DETECTIONS_MERGED[DETECTIONS_MERGED[:, 1] > MIN_P_TISSNET_PRIMARY]
        DETECTIONS_MERGED = DETECTIONS_MERGED[np.argsort(DETECTIONS_MERGED[:, 1])][::-1]

        BOUNDS = [(1_000, 100_000), (-13.4, -12.4), (45.25, 46.25)]
        GRID_SIZES = [100, 100]
        PICK_UNCERTAINTY = 1
        MAX_CLOCK_DRIFT = 0.1
        GEOMETRICAL = 0.1

        GRID_PATH = f"{DETECTIONS_DIR}/cache/grids_{BOUNDS[0][0]}_{BOUNDS[0][1]}_{BOUNDS[1][0]}_{BOUNDS[1][1]}_{BOUNDS[2][0]}_{BOUNDS[2][1]}_{GRID_SIZES[0]}_{GRID_SIZES[1]}_{PICK_UNCERTAINTY}_{MAX_CLOCK_DRIFT}.pkl"

        if not Path(GRID_PATH).exists():
            GRID_TO_COORDS, TDoA, MAX_TDoA, TDoA_UNCERTAINTY, LATS, DEPTHS, TRAVEL_TIMES = compute_grids(BOUNDS,
                                                                                                         GRID_SIZES,
                                                                                                         STATIONS,
                                                                                                         seismic_paths,
                                                                                                         pick_uncertainty=PICK_UNCERTAINTY,
                                                                                                         max_clock_drift=MAX_CLOCK_DRIFT,
                                                                                                         geometrical_uncertainty=GEOMETRICAL)
            with open(GRID_PATH, "wb") as f:
                pickle.dump((GRID_TO_COORDS, TDoA, MAX_TDoA, TDoA_UNCERTAINTY, LATS, DEPTHS, TRAVEL_TIMES), f)
        else:
            with open(GRID_PATH, "rb") as f:
                GRID_TO_COORDS, TDoA, MAX_TDoA, TDoA_UNCERTAINTY, LATS, DEPTHS, TRAVEL_TIMES = pickle.load(f)
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
            other_stations = other_stations[np.argsort([MAX_TDoA[s1][s2] for s2 in other_stations])]

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
                                                               TDoA_UNCERTAINTY)

                    valid_points_new = np.argwhere(valid_grid_new)[:, 0]

                    if len(valid_points_new) > 0:
                        current_association[station] = (date, idx_det)

                        if np.all(
                                [len(current_association) >= max_reached_per_det[idx] - ASSOCIATION_RECORD_TOLERANCE for
                                 _, idx in
                                 current_association.values()]):
                            if COSTS:
                                update_results(current_association, valid_points_new, local_association, TDoA,
                                               TDoA_UNCERTAINTY, IDX_TO_DET, STATIONS, compute_costs=True)
                            else:
                                update_results(current_association, valid_points_new, local_association, TDoA,
                                               TDoA_UNCERTAINTY)
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
            fname = f"{DETECTIONS_DIR}/cache/associations_{PICK_UNCERTAINTY}_{MIN_ASSOCIATION_SIZE}_{i * frac:.02f}.pkl"
            print(fname)
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

        DELTA = pd.Timedelta(seconds=15)
        S_df = pd.read_csv(
            "../../../../data/MAHY/lavayssiere_and_public.csv", header=None,
            names=["date", "lat", "lon", "depth", "mb"], parse_dates=["date"]
        )
        S_df = S_df[(S_df['date'] >= np.min(DETECTIONS_MERGED[:, 0]) - 5 * DELTA) & (
                    S_df['date'] <= np.max(DETECTIONS_MERGED[:, 0]))]
        pts_df = []

        matched = {}

        asso, matched_asso = 0, 0
        association_files = glob2.glob(f"{DETECTIONS_DIR}/cache/associations_{PICK_UNCERTAINTY}_{3}_*.pkl")
        for file in association_files:
            with open(file, "rb") as f:
                associations = pickle.load(f)
            for detections, valid_points in tqdm(associations):
                asso += 1

                if len(np.array(valid_points).shape) == 2:
                    cell = int(valid_points[np.argmin(np.array(valid_points)[:,2])][0])
                else:
                    coords = np.mean(GRID_TO_COORDS[valid_points],axis=0)
                    d_diff = ((GRID_TO_COORDS[:,1] - coords[1]) ** 2 +
                              (GRID_TO_COORDS[:,2] - coords[2]) ** 2 +
                              ((GRID_TO_COORDS[:,0] - coords[0]) / 111_000) ** 2)
                    cell = np.argmin(d_diff)
                coords = GRID_TO_COORDS[cell]
                try:
                    dates = np.array([IDX_TO_DET[d_i][0] - datetime.timedelta(seconds=TRAVEL_TIMES[STATIONS[s_i]][cell]) for s_i, d_i in detections])
                except ValueError:
                    continue
                date_h = dates[0] + np.mean(dates - dates[0])

                pts_df.append({"date": date_h, "depth": coords[0], "lat": coords[1], "lon": coords[2]})
                for si, di in detections:
                    s, (date, _, _) = STATIONS[si], IDX_TO_DET[di]
                    pts_df[-1][s.name] = date

                candidates = S_df[(S_df['date'] >= date_h - DELTA) & \
                                  (S_df['date'] <= date_h + DELTA)]
                if len(candidates) > 0:
                    matched_asso += 1
                for idx in candidates.index:
                    d_diff = np.sqrt(
                        (S_df['lat'][idx] - coords[1]) ** 2 +
                        (S_df['lon'][idx] - coords[2]) ** 2 +
                        ((S_df['depth'][idx] - coords[0]) / 111_000) ** 2)

                    t_diff = S_df['date'][idx] - date_h

                    matched.setdefault(idx, []).append((t_diff.total_seconds(), d_diff, detections, coords, date_h))
        print(asso, matched_asso)

        n = 0
        S_df_matched = S_df.loc[list(matched.keys())].copy()
        for idx in matched.keys():
            if len(matched[idx]) > 1:
                longest = np.argmax([len(d[2]) for d in matched[idx]])
                best = np.argmin([d[1] for d in matched[idx]])
                if len(matched[idx][longest][2]) == 4:
                    n += 1
                    best = longest
                matched[idx] = [matched[idx][best]]

            h_dates = []
            for si, di in matched[idx][0][2]:
                s, (date, _, _) = STATIONS[si], IDX_TO_DET[di]
                S_df_matched.loc[idx, s.name] = date
            S_df_matched.loc[idx, "h_date"] = matched[idx][0][4]
            S_df_matched.loc[idx, "h_depth"] = matched[idx][0][3][0] / 1_000
            S_df_matched.loc[idx, "h_lat"] = matched[idx][0][3][1]
            S_df_matched.loc[idx, "h_lon"] = matched[idx][0][3][2]

        print(n, len(S_df_matched), len(S_df))

        h = dataset[-1]

        if len(STATIONS) > 3:
            S_df_matched.to_csv(f'../../../../data/MAHY/loc_3D/twin-cat/{dataset}_fixed_OBS-fixed.csv', index=False,
                                columns=["date", "h_date", "lat", "h_lat", "lon", "h_lon", "depth", "h_depth", "mb",
                                         f"MAHY{h}1", f"MAHY{h}2", f"MAHY{h}3", f"MAHY{h}4"], float_format='%.3f')
            pts_df = pd.DataFrame(pts_df)
            pts_df.to_csv(f'../../../../data/MAHY/loc_3D/twin-cat/{dataset}_all_OBS-fixed.csv', index=False,
                          columns=["date", "lat", "lon", "depth",f"MAHY{h}1", f"MAHY{h}2", f"MAHY{h}3", f"MAHY{h}4"], float_format='%.3f')
        else:
            S_df_matched.to_csv(f'../../../../data/MAHY/loc_3D/twin-cat/{dataset}_fixed_OBS-fixed.csv', index=False,
                                columns=["date", "h_date", "lat", "h_lat", "lon", "h_lon", "depth", "h_depth", "mb",
                                         f"MAHY{h}1", f"MAHY{h}2", f"MAHY{h}3"], float_format='%.3f')
            pts_df = pd.DataFrame(pts_df)
            pts_df.to_csv(f'../../../../data/MAHY/loc_3D/twin-cat/{dataset}_all_OBS-fixed.csv', index=False,
                          columns=["date", "lat", "lon", "depth",f"MAHY{h}1", f"MAHY{h}2", f"MAHY{h}3"], float_format='%.3f')