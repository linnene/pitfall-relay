#!/usr/bin/env python3
"""Navigation self-failure detection: collection + offline training + evaluation.

Semantics (frozen before collection):
  fixed goal (gx, gy, radius=0.05, deadline=64 steps, inclusive)
  success   : distance < radius at some t <= deadline-1 (entry; leaving later still success)
  in_progress: t < deadline-1 and never entered
  failure   : at t == deadline-1 and never entered
  censored  : episode cut before deadline without success (not produced here)

Execution input: local obs (18), action (2), goal vector, clock. Privileged
positions are used only for offline labels / counterfactual relabeling.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import vmas

DEADLINE = 64
RADIUS = 0.05
WINDOW = 8
NUM_ENVS = 8
CONTROLLERS = ["direct", "pause", "slow", "leave", "noisy"]


def sample_goal(rng, start, region):
    """Distance 0.7-1.5 from start, inside the requested x half."""
    for _ in range(200):
        ang = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(0.45, 1.1)
        g = start + d * np.array([np.cos(ang), np.sin(ang)])
        if np.abs(g).max() > 0.98:
            continue
        if region == "seen" and g[0] >= 0:
            continue
        if region == "holdout" and g[0] <= 0:
            continue
        return g
    return np.array([-0.7, 0.0]) if region == "seen" else np.array([0.7, 0.0])


def policy(env, t, ctrl, arrived_t, prev_actions):
    ag = env.scenario.world.agents[0]
    pos = ag.state.pos
    vel = ag.state.vel
    goal = ag.goal.state.pos
    delta = goal - pos
    n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    speed = 0.10 if ctrl == "slow" else 0.18
    v_des = delta / n * speed
    u = (v_des - vel) * 8.0
    if ctrl == "pause" and 15 <= t < 30:
        u = torch.zeros_like(u)
    if ctrl == "leave":
        arrived = (arrived_t >= 0).squeeze(-1)
        u = torch.where(arrived.unsqueeze(-1), -u, u)
    if ctrl == "noisy":
        phase = torch.arange(u.shape[0], dtype=u.dtype).unsqueeze(-1)
        noise = 0.35 * torch.sin(0.7 * t + phase)
        u = u + noise
        if t >= 2 and len(prev_actions) >= 2:
            u = prev_actions[-2]
    return [u.clamp(-1.0, 1.0)]


def collect(out_dir, n_episodes, base_seed):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(base_seed)
    store = {k: [] for k in ("obs", "act", "pos", "goal", "dist")}
    ep_ids = []
    steps_used = 0
    ep = 0
    while ep < n_episodes:
        batch = min(NUM_ENVS, n_episodes - ep)
        env = vmas.make_env("navigation", num_envs=batch, device="cpu",
                            continuous_actions=True, seed=base_seed + ep, n_agents=1,
                            collisions=False)
        env.reset(seed=base_seed + ep)
        ag = env.scenario.world.agents[0]
        goals = []
        ctrls = []
        for i in range(batch):
            idx = ep + i
            region = "seen" if idx < int(0.4 * n_episodes) else "holdout"
            start = ag.state.pos[i].numpy()
            g = sample_goal(rng, start, region)
            goals.append(g)
            ctrls.append(CONTROLLERS[idx % len(CONTROLLERS)])
        g_t = torch.tensor(np.stack(goals), dtype=torch.float32)
        ag.goal.set_pos(g_t, None)
        ag.pos_shaping = (ag.state.pos - ag.goal.state.pos).norm(dim=-1)

        arrived_t = torch.full((batch,), -1, dtype=torch.long)
        prev_actions = []
        rec = {k: [] for k in ("obs", "act", "pos", "goal", "dist", "arrived")}
        for t in range(DEADLINE):
            acts = []
            # (per-env controller branch is applied vectorized below)
            pos = ag.state.pos; vel = ag.state.vel
            delta = ag.goal.state.pos - pos
            n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            speed = torch.where(torch.tensor([c == "slow" for c in ctrls]),
                                torch.tensor(0.10), torch.tensor(0.18)).unsqueeze(-1)
            u = (delta / n * speed - vel) * 8.0
            pause_mask = torch.tensor([(c == "pause" and 15 <= t < 30) for c in ctrls])
            u = torch.where(pause_mask.unsqueeze(-1), torch.zeros_like(u), u)
            leave_mask = (torch.tensor([c == "leave" for c in ctrls]) & (arrived_t >= 0))
            u = torch.where(leave_mask.unsqueeze(-1), -u, u)
            noisy_mask = torch.tensor([c == "noisy" for c in ctrls])
            phase = torch.arange(batch, dtype=u.dtype).unsqueeze(-1)
            u = torch.where(noisy_mask.unsqueeze(-1),
                            u + 0.35 * torch.sin(0.7 * t + phase), u)
            if t >= 2 and len(prev_actions) >= 2:
                u = torch.where(noisy_mask.unsqueeze(-1), prev_actions[-2], u)
            u = u.clamp(-1.0, 1.0)
            prev_actions.append(u.clone())

            obs, rews, dones, infos = env.step([u])
            d = (ag.state.pos - ag.goal.state.pos).norm(dim=-1)
            newly = (d < RADIUS) & (arrived_t < 0)
            arrived_t = torch.where(newly, torch.full_like(arrived_t, t), arrived_t)
            rec["obs"].append(obs[0].numpy())
            rec["act"].append(u.numpy())
            rec["pos"].append(ag.state.pos.numpy())
            rec["goal"].append(ag.goal.state.pos.numpy())
            rec["dist"].append(d.numpy())
            rec["arrived"].append(arrived_t.numpy())
        steps_used += batch * DEADLINE
        for i in range(batch):
            ep_ids.append({
                "episode": ep + i, "seed": base_seed + ep,
                "goal": goals[i].tolist(),
                "region": "seen" if (ep + i) < int(0.4 * n_episodes) else "holdout",
                "controller": ctrls[i],
                "arrived_t": int(arrived_t[i]),
                "min_dist": float(np.min([rec["dist"][t][i] for t in range(DEADLINE)])),
            })
        for k in rec:
            arr = np.stack(rec[k])  # [T, batch, ...]
            for i in range(batch):
                if k in store:
                    store[k].append(arr[:, i])
        ep += batch
    arrays = {k: np.stack(v) for k, v in store.items()}
    np.savez_compressed(os.path.join(out_dir, "episodes.npz"), **arrays)
    with open(os.path.join(out_dir, "episodes.json"), "w") as f:
        json.dump({"meta": {"n_episodes": n_episodes, "deadline": DEADLINE,
                            "radius": RADIUS, "steps_used": steps_used,
                            "num_envs": NUM_ENVS}, "episodes": ep_ids}, f)
    print(json.dumps({"collection": {"episodes": n_episodes, "steps": steps_used}}))


def load(out_dir):
    d = np.load(os.path.join(out_dir, "episodes.npz"))
    meta = json.load(open(os.path.join(out_dir, "episodes.json")))
    # stored [E, T, ...]; convert to [T, E, ...]
    out = {}
    for k in d.files:
        arr = d[k]
        out[k] = np.swapaxes(arr, 0, 1)
    return out, meta


def split_episodes(meta):
    train, dev, test_seen, test_hold = [], [], [], []
    seen = [e for e in meta["episodes"] if e["region"] == "seen"]
    hold = [e for e in meta["episodes"] if e["region"] == "holdout"]
    n_tr = int(0.67 * len(seen)); n_dv = int(0.17 * len(seen))
    for i, e in enumerate(seen):
        if i < n_tr:
            train.append(e["episode"])
        elif i < n_tr + n_dv:
            dev.append(e["episode"])
        else:
            test_seen.append(e["episode"])
    test_hold = [e["episode"] for e in hold]
    return {"train": train, "dev": dev, "test_seen": test_seen, "test_holdout": test_hold}


def build_features(data, meta, history):
    obs = data["obs"]        # [T, E, 18]
    act = data["act"]        # [T, E, 2]
    goal = data["goal"]      # [T, E, 2]
    pos = data["pos"]        # [T, E, 2]
    T, E, _ = obs.shape
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    xs, ys, keys = [], [], []
    for e in range(E):
        info = ep_info[e]
        g = np.array(info["goal"], dtype=np.float32)
        arrived_t = info["arrived_t"]
        # recompute goal-relative channel to guarantee consistency (pos - goal)
        goal_rel = pos[:, e] - g[None, :]
        for t in range(WINDOW - 1, T):
            win_obs = obs[t - WINDOW + 1:t + 1, e].copy()
            win_obs[:, 4:6] = goal_rel[t - WINDOW + 1:t + 1]
            win_act = act[t - WINDOW + 1:t + 1, e]
            flat = np.concatenate([win_obs.reshape(-1), win_act.reshape(-1)])
            clock = np.array([t / (DEADLINE - 1)], dtype=np.float32)
            gvec = np.array([g[0], g[1], RADIUS, DEADLINE], dtype=np.float32)
            hist = np.array([np.linalg.norm(goal_rel[:t + 1], axis=-1).min()], dtype=np.float32)
            x = np.concatenate([flat, gvec, clock]) if not history else \
                np.concatenate([flat, gvec, clock, hist])
            y = 1.0 if (t == T - 1 and arrived_t < 0) else 0.0
            xs.append(x); ys.append(y)
            keys.append((e, t))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32), keys


def build_features_nogoal(data, meta):
    obs = data["obs"]; act = data["act"]; pos = data["pos"]
    T, E, _ = obs.shape
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    xs, ys, keys = [], [], []
    for e in range(E):
        info = ep_info[e]; g = np.array(info["goal"], dtype=np.float32)
        goal_rel = pos[:, e] - g[None, :]
        for t in range(WINDOW - 1, T):
            win_obs = obs[t - WINDOW + 1:t + 1, e].copy()
            win_obs[:, 4:6] = 0.0  # remove goal-relative channel
            win_act = act[t - WINDOW + 1:t + 1, e]
            flat = np.concatenate([win_obs.reshape(-1), win_act.reshape(-1)])
            clock = np.array([t / (DEADLINE - 1)], dtype=np.float32)
            x = np.concatenate([flat, clock])
            y = 1.0 if (t == T - 1 and info["arrived_t"] < 0) else 0.0
            xs.append(x); ys.append(y); keys.append((e, t))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32), keys


class MLP(nn.Module):
    def __init__(self, dim, hidden=(32, 16)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden[0]), nn.ReLU(),
                                 nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
                                 nn.Linear(hidden[1], 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(x, y, seed, epochs=12, bs=512):
    torch.manual_seed(seed)
    m = MLP(x.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pos = float(y.sum()); neg = len(y) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0)))
    w = np.where(y > 0.5, 0.5 / max(pos, 1.0), 0.5 / max(neg, 1.0))
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                                     num_samples=len(y), replacement=True)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler)
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(m(xb), yb).backward(); opt.step()
    return m


def predict(m, x):
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(torch.from_numpy(x))).numpy()


def episode_outcome(info):
    return "failure" if info["arrived_t"] < 0 else "success"


def evaluate(pred_by_key, meta, episodes, threshold=0.5):
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    deadline = DEADLINE - 1
    by_ep = {}
    for (e, t), p in pred_by_key.items():
        by_ep.setdefault(e, {})[t] = p
    tp = fp = fn = tn = 0
    alarm_eps = 0; eval_eps = 0
    first_fire = {}
    for e in episodes:
        info = ep_info[e]
        y = 1 if info["arrived_t"] < 0 else 0
        pe = by_ep.get(e, {})
        p_dead = pe.get(deadline, 0.0) >= threshold
        if y:
            tp += int(p_dead); fn += int(not p_dead)
        else:
            fp += int(p_dead); tn += int(not p_dead)
        pre = [t for t, p in pe.items() if t < deadline and p >= threshold]
        eval_eps += 1
        if pre:
            alarm_eps += 1
            first_fire[e] = min(pre)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "failure_miss_rate": fn / max(tp + fn, 1),
            "predeadline_alarm_rate": alarm_eps / max(eval_eps, 1),
            "n_deadline": tp + fp + fn + tn, "n_episodes": eval_eps,
            "first_fire": first_fire}


def choose_threshold(pred_by_key, meta, episodes):
    """Pick the decision threshold on dev only: minimise miss + pre-deadline alarm."""
    best_th, best_score = 0.5, 1e9
    for th in np.arange(0.10, 0.91, 0.05):
        r = evaluate(pred_by_key, meta, episodes, threshold=float(th))
        score = r["failure_miss_rate"] + r["predeadline_alarm_rate"]
        if score < best_score:
            best_th, best_score = float(th), score
    return best_th, best_score


def label3(info, t):
    """0 success (already arrived), 1 in-progress, 2 failure confirmed."""
    if info["arrived_t"] >= 0 and t >= info["arrived_t"]:
        return 0
    if info["arrived_t"] < 0 and t == DEADLINE - 1:
        return 2
    return 1


def build_features3(data, meta, history):
    obs = data["obs"]; act = data["act"]; goal = data["goal"]; pos = data["pos"]
    T, E, _ = obs.shape
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    xs, ys, keys = [], [], []
    for e in range(E):
        info = ep_info[e]; g = np.array(info["goal"], dtype=np.float32)
        goal_rel = pos[:, e] - g[None, :]
        for t in range(WINDOW - 1, T):
            win = obs[t - WINDOW + 1:t + 1, e].copy()
            win[:, 4:6] = goal_rel[t - WINDOW + 1:t + 1]
            flat = np.concatenate([win.reshape(-1), act[t - WINDOW + 1:t + 1, e].reshape(-1)])
            clock = np.array([t / (DEADLINE - 1)], dtype=np.float32)
            gv = np.array([g[0], g[1], RADIUS, DEADLINE], dtype=np.float32)
            hist = np.array([np.linalg.norm(goal_rel[:t + 1], axis=-1).min()], dtype=np.float32)
            x = np.concatenate([flat, gv, clock, hist]) if history else np.concatenate([flat, gv, clock])
            xs.append(x); ys.append(label3(info, t)); keys.append((e, t))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64), keys


def build_features3_nogoal(data, meta):
    obs = data["obs"]; act = data["act"]; pos = data["pos"]
    T, E, _ = obs.shape
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    xs, ys, keys = [], [], []
    for e in range(E):
        info = ep_info[e]; g = np.array(info["goal"], dtype=np.float32)
        goal_rel = pos[:, e] - g[None, :]
        for t in range(WINDOW - 1, T):
            win = obs[t - WINDOW + 1:t + 1, e].copy()
            win[:, 4:6] = 0.0
            flat = np.concatenate([win.reshape(-1), act[t - WINDOW + 1:t + 1, e].reshape(-1)])
            clock = np.array([t / (DEADLINE - 1)], dtype=np.float32)
            xs.append(np.concatenate([flat, clock])); ys.append(label3(info, t)); keys.append((e, t))
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64), keys


class MLP3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(),
                                 nn.Linear(32, 16), nn.ReLU(),
                                 nn.Linear(16, 3))

    def forward(self, x):
        return self.net(x)


def train_mlp3(x, y, seed, epochs=12, bs=512):
    torch.manual_seed(seed)
    m = MLP3(x.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    counts = np.bincount(y, minlength=3).astype(np.float64)
    w = 1.0 / np.maximum(counts, 1)
    w = w / w.sum()
    sw = w[y]
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(sw, dtype=torch.double),
                                                     num_samples=len(y), replacement=True)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, sampler=sampler)
    loss_fn = nn.CrossEntropyLoss()
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(m(xb), yb).backward(); opt.step()
    return m


def predict3(m, x):
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(x)).argmax(-1).numpy()


def evaluate3(pred_by_key, meta, episodes):
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    deadline = DEADLINE - 1
    by_ep = {}
    for (e, t), p in pred_by_key.items():
        by_ep.setdefault(e, {})[t] = int(p)
    conf = np.zeros((3, 3), dtype=np.int64)
    tp = fp = fn = tn = 0
    alarm_eps = inprog_ok = inprog_n = 0
    for e in episodes:
        info = ep_info[e]
        y = 2 if info["arrived_t"] < 0 else 0
        pe = by_ep.get(e, {})
        p_dead = pe.get(deadline, 1)
        conf[y if y == 0 else 2, p_dead] += 1
        if y == 2:
            tp += int(p_dead == 2); fn += int(p_dead != 2)
        else:
            fp += int(p_dead == 2); tn += int(p_dead != 2)
        pre = [t for t, p in pe.items() if t < deadline]
        if pre and any(pe[t] == 2 for t in pre):
            alarm_eps += 1
        for t in pre:
            if label3(info, t) == 1:
                inprog_n += 1
                inprog_ok += int(pe[t] == 1)
    return {"confusion_deadline": conf.tolist(), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "failure_miss_rate": fn / max(tp + fn, 1),
            "predeadline_alarm_rate": alarm_eps / max(len(episodes), 1),
            "inprogress_recall": inprog_ok / max(inprog_n, 1),
            "n_episodes": len(episodes)}


def rule_predictions(data, meta, episodes):
    """distance + clock + full local arrival memory rule."""
    pos = data["pos"]; goal = data["goal"]
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    out = {}
    for e in episodes:
        g = np.array(ep_info[e]["goal"])
        dist = np.linalg.norm(pos[:, e] - g, axis=-1)
        for t in range(WINDOW - 1, DEADLINE):
            arrived = dist[:t + 1].min() < RADIUS
            if arrived:
                out[(e, t)] = 0
            elif t == DEADLINE - 1:
                out[(e, t)] = 2
            else:
                out[(e, t)] = 1
    return out


def counterfactual_eval(model, data, meta, episodes):
    """Swap the goal on the same physical trajectory, rebuild goal-relative obs."""
    pos = data["pos"]; obs = data["obs"]; act = data["act"]
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    xs, keys, labels = [], [], []
    for e in episodes:
        g0 = np.array(ep_info[e]["goal"], dtype=np.float32)
        # counterfactual goal: reflect across y axis, same distance from start
        g1 = np.array([-g0[0], g0[1]], dtype=np.float32)
        rel0 = pos[:, e] - g0[None, :]
        rel1 = pos[:, e] - g1[None, :]
        arr0 = int(np.argmax(np.linalg.norm(rel0, axis=-1) < RADIUS)) if (np.linalg.norm(rel0, axis=-1) < RADIUS).any() else -1
        arr1 = int(np.argmax(np.linalg.norm(rel1, axis=-1) < RADIUS)) if (np.linalg.norm(rel1, axis=-1) < RADIUS).any() else -1
        for t in range(WINDOW - 1, DEADLINE):
            win = obs[t - WINDOW + 1:t + 1, e].copy()
            win[:, 4:6] = rel1[t - WINDOW + 1:t + 1]
            flat = np.concatenate([win.reshape(-1), act[t - WINDOW + 1:t + 1, e].reshape(-1)])
            clock = np.array([t / (DEADLINE - 1)], dtype=np.float32)
            gv = np.array([g1[0], g1[1], RADIUS, DEADLINE], dtype=np.float32)
            hist = np.array([np.linalg.norm(rel1[:t + 1], axis=-1).min()], dtype=np.float32)
            xs.append(np.concatenate([flat, gv, clock, hist]))
            keys.append((e, t))
            labels.append(1.0 if (t == DEADLINE - 1 and arr1 < 0) else 0.0)
    x = np.stack(xs).astype(np.float32)
    pred = predict(model, x) >= 0.5
    cf = {}
    for k, l, p in zip(keys, labels, pred):
        cf[k] = (l, p)
    flips = 0; flip_ok = 0
    for e in episodes:
        y0 = 1 if ep_info[e]["arrived_t"] < 0 else 0
        key = (e, DEADLINE - 1)
        if key not in cf:
            continue
        y1, p1 = cf[key]
        if y0 != int(y1):
            flips += 1
            flip_ok += int(p1 == bool(y1))
    return {"counterfactual_flips": flips,
            "counterfactual_flip_accuracy": flip_ok / max(flips, 1)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["collect", "train"], required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--episodes", type=int, default=1496)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    if args.mode == "collect":
        collect(args.dir, args.episodes, args.seed)
        return

    data, meta = load(args.dir)
    splits = split_episodes(meta)

    feats = {}
    for history in (False, True):
        x, y, keys = build_features(data, meta, history)
        feats["hist" if history else "win"] = (x, y, keys)
    x_ng, y_ng, keys_ng = build_features_nogoal(data, meta)

    def subset(x, keys, eps):
        idx = [i for i, (e, t) in enumerate(keys) if e in eps]
        return x[idx]

    results = {}
    models = {}
    for seed in (0, 1, 2):
        for name, hist in (("win", False), ("hist", True)):
            x, y, keys = feats["hist" if hist else "win"]
            tr = subset(x, keys, set(splits["train"]))
            ytr = np.array([y[i] for i, (e, t) in enumerate(keys) if e in set(splits["train"])])
            mu = tr.mean(0); sd = tr.std(0); sd[sd < 1e-6] = 1.0
            model = train_mlp(((tr - mu) / sd).astype(np.float32), ytr, seed)
            models[f"{name}_{seed}"] = (model, mu, sd)
            test_eps = set(splits["test_seen"] + splits["test_holdout"])
            xs = ((subset(x, keys, test_eps) - mu) / sd).astype(np.float32)
            ks = [k for k in keys if k[0] in test_eps]
            pr = predict(model, xs)
            pred_by_key = dict(zip(ks, pr))
            xd = ((subset(x, keys, set(splits["dev"])) - mu) / sd).astype(np.float32)
            kd = [k for k in keys if k[0] in set(splits["dev"])]
            pdv = predict(model, xd)
            pred_dev = dict(zip(kd, pdv))
            th, th_score = choose_threshold(pred_dev, meta, splits["dev"])
            results[f"{name}_{seed}"] = {
                "dev_threshold": th,
                "test_all": evaluate(pred_by_key, meta, list(test_eps), threshold=th),
                "test_seen": evaluate(pred_by_key, meta, splits["test_seen"], threshold=th),
                "test_holdout": evaluate(pred_by_key, meta, splits["test_holdout"], threshold=th),
            }

    # rule baseline
    test_eps = splits["test_seen"] + splits["test_holdout"]
    rule = rule_predictions(data, meta, test_eps)
    results["rule"] = {"test_all": evaluate(rule, meta, test_eps),
                       "test_seen": evaluate(rule, meta, splits["test_seen"]),
                       "test_holdout": evaluate(rule, meta, splits["test_holdout"])}
    # goal+clock only: predict failure iff at deadline (never use history)
    gcpred = {}
    for e in test_eps:
        for t in range(WINDOW - 1, DEADLINE):
            gcpred[(e, t)] = 1.0 if t == DEADLINE - 1 else 0.0
    results["deadline_only"] = {"test_all": evaluate(gcpred, meta, test_eps),
                                "test_seen": evaluate(gcpred, meta, splits["test_seen"]),
                                "test_holdout": evaluate(gcpred, meta, splits["test_holdout"])}

    # no-goal ablation
    tr_idx = [i for i, (e, t) in enumerate(keys_ng) if e in set(splits["train"])]
    mu2 = x_ng[tr_idx].mean(0); sd2 = x_ng[tr_idx].std(0); sd2[sd2 < 1e-6] = 1.0
    ytr2 = y_ng[tr_idx]
    m2 = train_mlp(((x_ng[tr_idx] - mu2) / sd2).astype(np.float32), ytr2, 0)
    te_idx = [i for i, (e, t) in enumerate(keys_ng) if e in set(test_eps)]
    p2 = predict(m2, ((x_ng[te_idx] - mu2) / sd2).astype(np.float32))
    results["nogoal_0"] = {"test_all": evaluate(dict(zip([keys_ng[i] for i in te_idx], p2)),
                                                meta, test_eps)}

    # counterfactual on hist model seed 0
    model, mu, sd = models["hist_0"]
    results["counterfactual"] = counterfactual_eval(model, data, meta, test_eps)

    # stratified counts
    ep_info = {e["episode"]: e for e in meta["episodes"]}
    strat = {"late_arrival": 0, "leave_after": 0, "near_boundary_fail": 0, "flip_goal": 0}
    for e in test_eps:
        info = ep_info[e]
        if info["arrived_t"] >= DEADLINE - 10 and info["arrived_t"] >= 0:
            strat["late_arrival"] += 1
        if info["arrived_t"] >= 0 and info["min_dist"] < RADIUS:
            g = np.array(info["goal"])
            d_end = float(np.linalg.norm(data["pos"][-1, e] - g))
            if d_end > 0.2:
                strat["leave_after"] += 1
        if info["arrived_t"] < 0 and info["min_dist"] <= RADIUS + 0.10:
            strat["near_boundary_fail"] += 1
    results["strata"] = strat
    results["splits"] = {k: len(v) for k, v in splits.items()}
    with open(args.json_out or os.path.join(args.dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "first_fire"}, indent=2,
                     default=str))


if __name__ == "__main__":
    main()
