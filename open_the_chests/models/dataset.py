import torch
from torch.utils.data import Dataset

from open_the_chests.envs.sequence_generator import generate_sequence
from open_the_chests.envs.otc_registry import (
    all_event_types, all_event_attributes,
    all_noise_types, all_noise_attributes,
)

# merge types and attributes (same as Parser does internally)
ALL_TYPES = all_event_types + all_noise_types
ALL_BG    = all_event_attributes["bg"] + all_noise_attributes["bg"]
ALL_FG    = all_event_attributes["fg"] + all_noise_attributes["fg"]

NUM_CHESTS = 3
TIME_NORM  = 100.0  # divide start/end by this to keep values ~[0,1]


def _encode_event(event):
    e_type   = ALL_TYPES.index(event.type)
    bg       = ALL_BG.index(event.attributes["bg"])
    fg       = ALL_FG.index(event.attributes["fg"])
    start    = event.start    / TIME_NORM
    end      = event.end      / TIME_NORM
    duration = (event.end - event.start) / TIME_NORM
    return e_type, bg, fg, start, end, duration


def _signals_to_action(signals):
    """Returns a binary vector of size NUM_CHESTS: 1 if that pattern is satisfied."""
    action = [0.0] * NUM_CHESTS
    for pid, sigs in signals.items():
        if "satisfied" in sigs and pid < NUM_CHESTS:
            action[pid] = 1.0
    return action


def build_trajectory(n_events=200, env: str = "medium"):
    """
    Generates one trajectory and returns tensors ready for the model.

    Returns
    -------
    dict with keys:
        e_type, bg, fg : (T,) int
        start, end, duration, open_chests : (T,) or (T, NUM_CHESTS) float
        a              : (T, NUM_CHESTS) float  — correct action at each step
        R              : (T, 1) float           — return-to-go
        t              : (T,)  int              — timestep indices
    """
    events, signals_list = generate_sequence(n_events, env=env)
    T = len(events)
    raw_events = events  # keep reference for callers that need the Event objects

    e_types, bgs, fgs, starts, ends, durations = [], [], [], [], [], []
    actions = []
    opened = [False] * NUM_CHESTS  # each chest opens at most once

    for event, signals in zip(events, signals_list):
        e_type, bg, fg, start, end, duration = _encode_event(event)
        e_types.append(e_type)
        bgs.append(bg)
        fgs.append(fg)
        starts.append(start)
        ends.append(end)
        durations.append(duration)

        action = [0.0] * NUM_CHESTS
        for pid, sigs in signals.items():
            if "satisfied" in sigs and pid < NUM_CHESTS and not opened[pid]:
                action[pid] = 1.0
                opened[pid] = True
        actions.append(action)

    actions_t = torch.tensor(actions, dtype=torch.float32)  # (T, NUM_CHESTS)

    # open_chests: cumulative OR of past actions (what has been opened so far)
    open_chests = torch.zeros(T, NUM_CHESTS)
    for i in range(1, T):
        open_chests[i] = (open_chests[i - 1] + actions_t[i - 1]).clamp(max=1.0)

    # reward: 1 if any chest was correctly opened at this step
    rewards = actions_t.sum(dim=-1)  # (T,)

    # return-to-go: sum of future rewards
    R = torch.flip(torch.cumsum(torch.flip(rewards, [0]), dim=0), [0])  # (T,)

    return raw_events, {
        "e_type":      torch.tensor(e_types,    dtype=torch.long),
        "bg":          torch.tensor(bgs,         dtype=torch.long),
        "fg":          torch.tensor(fgs,         dtype=torch.long),
        "start":       torch.tensor(starts,      dtype=torch.float32),
        "end":         torch.tensor(ends,         dtype=torch.float32),
        "duration":    torch.tensor(durations,   dtype=torch.float32),
        "open_chests": open_chests,
        "a":           actions_t,
        "R":           R.unsqueeze(-1),           # (T, 1)
        "t":           torch.arange(T,           dtype=torch.long),
    }


class ChestDataset(Dataset):
    def __init__(self, num_sequences: int, n_events: int = 200, K: int = None, env: str = "medium"):
        """
        Parameters
        ----------
        num_sequences : number of trajectories to generate
        n_events      : events per trajectory
        K             : context window size (truncates each trajectory to last K steps).
                        If None, uses the full trajectory.
        env           : which environment to use ("easy", "medium", "hard")
        """
        self.K = K
        self.data = [traj for _, traj in (build_trajectory(n_events, env=env) for _ in range(num_sequences))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj = self.data[idx]
        if self.K is not None:
            traj = {k: v[-self.K:] for k, v in traj.items()}
            # re-index t to start from 0
            traj["t"] = torch.arange(len(traj["t"]), dtype=torch.long)
        return traj
