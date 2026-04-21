from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap import bwm_query, load_good_units, load_trials_and_mask, bwm_units
from collections import defaultdict
import pandas as pd
from ibl_info.utils import check_config

config = check_config()
MY_REGIONS = config["stim_prior_regions"]


def savewifimice():
    one = ONE()
    sessions_all = one.search(datasets="widefieldU.images.npy")
    print(f"{len(sessions_all)} sessions with widefield data found")  # type: ignore
    data_folder = "../../public_hybrid_rnn/session_data/"

    # load each trial, and figure out timing data first

    time_dict = defaultdict(dict)
    for eid in sessions_all:  # type: ignore
        details = one.get_details(eid)
        subject = details["subject"]  # type: ignore
        timestamp = details["start_time"]  # type: ignore
        time_dict[subject][eid] = timestamp

    session_order_dict = defaultdict(dict)

    for subject, sessions in time_dict.items():
        sorted_eids = sorted(sessions.keys(), key=lambda eid: sessions[eid])
        for index, eid in enumerate(sorted_eids):
            session_order_dict[subject][eid] = index

    all_dfs = []
    for eid in sessions_all:
        details = one.get_details(eid)
        subject = details["subject"]
        sessionx = SessionLoader(one=one, eid=eid)
        sessionx.load_trials()
        trials_df = sessionx.trials.copy()
        sessionx = trials_df.assign(subject=subject)
        sessionx = sessionx.assign(session_number=session_order_dict[subject][eid])
        all_dfs.append(sessionx)

    uber_df = pd.concat(all_dfs)
    uber_df.to_csv(data_folder + "wifimice.csv")


def checkmiceresults():

    one = ONE(
        base_url="https://openalyx.internationalbrainlab.org",
        password="international",
        silent=True,
        username="intbrainlab",
    )
    print("Querying BWM Units...")

    units_df = bwm_units(one)
    relevant_pids = units_df[units_df["Beryl"].isin(MY_REGIONS)]["pid"].unique()

    bwm_df = bwm_query(one)
    subset_df = bwm_df[bwm_df["pid"].isin(relevant_pids)]

    task_list = [(row["pid"], row["eid"]) for _, row in subset_df.iterrows()]

    list_of_eids = subset_df["eid"].unique()

    all_dfs = []
    for idx, eid in enumerate(list_of_eids):

        details = one.get_details(eid)
        subject = details["subject"]
        sessionx = SessionLoader(one=one, eid=eid)
        sessionx.load_trials()
        trials_df = sessionx.trials.copy()
        sessionx = trials_df.assign(subject=subject)
        sessionx = sessionx.assign(session_number=idx)
        sessionx = sessionx.assign(session_id=eid)
        all_dfs.append(sessionx)

        if idx == 5:
            break

    uber_df = pd.concat(all_dfs)
    data_folder = "./data/"
    uber_df.to_csv(data_folder + "checkmiceresults.csv")


if __name__ == "__main__":
    checkmiceresults()
    # savewifimice()
