from one.api import ONE
from brainbox.io.one import SessionLoader
from brainwidemap.bwm_loading import load_trials_and_mask
from collections import defaultdict
import pandas as pd

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

wifi_sessions = []
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
