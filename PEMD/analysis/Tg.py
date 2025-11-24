import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from panedr import edr_to_df as edrdf


class TgAnanysis:
    def __init__(
        self,
        work_dir,
        edr_file="npt_tg.edr",
        T_start=600.0,
        T_end=100.0,
        dT=20.0,
        eq_time=2000.0,
        anneal_rate=5.0,  # 退火速度（K/ps）
        window_ps=1000.0,
    ):

        self.work_dir = work_dir
        self.edr_file =  edr_file
        self.T_start = float(T_start)
        self.T_end = float(T_end)
        self.dT = float(dT)
        self.plateau_ps = float(eq_time)
        self.window_ps = float(window_ps)
        self.ramp_ps = self.dT / anneal_rate  # = dT / (K/ps)
        self.anneal_times = None  # List[int]
        self.anneal_temps = None  # List[float]
        self.edr_df = None        # DataFrame
        self.tg_data = None       # DataFrame

        self._generate_annealing_schedule()

    # ------------------------------------------------------------------
    # 1. 生成 annealing-time / annealing-temp
    # ------------------------------------------------------------------
    def _generate_annealing_schedule(self):
        """根据 T_start / T_end / dT / plateau_ps / ramp_ps 生成退火 schedule。"""
        temps_levels = []
        T = self.T_start
        while True:
            temps_levels.append(T)
            T_next = T - self.dT
            if T_next < self.T_end:
                break
            T = T_next

        times = [0.0]
        temps = [self.T_start]
        t = 0.0

        for i, T in enumerate(temps_levels):
            # plateau 结束
            t += self.plateau_ps
            times.append(t)
            temps.append(T)

            if i == len(temps_levels) - 1:
                break  # 最后一个温度后面不再降温

            # ramp 结束，温度跳到下一个水平
            T_next = temps_levels[i + 1]
            t += self.ramp_ps
            times.append(t)
            temps.append(T_next)

        self.anneal_times = [int(x) for x in times]
        self.anneal_temps = temps

        print("[TgAnnealAnalyzer] Generated annealing schedule:")
        print("  npoints =", len(self.anneal_times))
        print("  time[0:6] =", self.anneal_times[:6])
        print("  temp[0:6] =", self.anneal_temps[:6])

    def print_mdp_block(self):
        """打印可以直接复制到 .mdp 的退火设置片段。"""
        if self.anneal_times is None or self.anneal_temps is None:
            raise RuntimeError("Annealing schedule 未生成。")

        print("annealing             = single")
        print(f"annealing-npoints     = {len(self.anneal_times)}")
        print("annealing-time        = " + " ".join(str(t) for t in self.anneal_times))
        print("annealing-temp        = " + " ".join(str(int(T)) for T in self.anneal_temps))

    # ------------------------------------------------------------------
    # 2. 读取 EDR
    # ------------------------------------------------------------------
    def load_edr(self):

        edr_filepath = os.path.join(self.work_dir, self.edr_file)
        df = edrdf(edr_filepath)
        self.edr_df = df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. 基于恒温平台尾部窗口构建 tg_data
    # ------------------------------------------------------------------
    def build_tg_data_from_plateaus(self, min_frames=10):
        """
        基于退火 schedule 中的恒温平台，在每个平台尾部取 window_ps 的时间窗，
        对 Temperature / Density 做平均，构建 tg_data。
        """
        if self.edr_df is None:
            raise RuntimeError("edr_df is None. 请先调用 load_edr()。")

        if self.anneal_times is None or self.anneal_temps is None:
            raise RuntimeError("Annealing schedule 未生成。")

        df = self.edr_df
        time = df["Time"].values  # ps

        # 找出所有恒温平台：相邻两个点温度相同的区间
        plateaus = []  # 每个元素：(T, t_start, t_end)
        for i in range(len(self.anneal_temps) - 1):
            if self.anneal_temps[i] == self.anneal_temps[i + 1]:
                T = self.anneal_temps[i]
                t0 = self.anneal_times[i]
                t1 = self.anneal_times[i + 1]
                plateaus.append((T, t0, t1))

        print(f"[TgAnnealAnalyzer] Found {len(plateaus)} plateaus (constant-T segments).")

        rows = []
        for T, t0, t1 in plateaus:
            # 平台尾部窗口 [t1 - window_ps, t1]
            t_start_win = max(t0, t1 - self.window_ps)
            t_end_win = t1

            mask = (time >= t_start_win) & (time <= t_end_win)
            window = df.loc[mask]

            if len(window) < min_frames:
                print(f"  Warning: T={T} K plateau has only {len(window)} frames, skip.")
                continue

            T_mean = window["Temperature"].mean()
            den_mean = window["Density"].mean()
            den_std = window["Density"].std()

            rows.append(
                {
                    "T_K": np.round(T_mean),  # 也可以直接用 T
                    "den_mean": float(den_mean),
                    "den_std": float(den_std),
                }
            )

        self.tg_data = pd.DataFrame(rows, columns=["T_K", "den_mean", "den_std"])
        print(f"[TgAnnealAnalyzer] tg_data built with {len(self.tg_data)} points.")

    # ------------------------------------------------------------------
    # 4. 保存 / 读取 tg_data
    # ------------------------------------------------------------------
    def save_tg_data(self, csv_path="tg_data.csv"):
        if self.tg_data is None:
            raise RuntimeError("tg_data is None. 请先调用 build_tg_data_from_plateaus()。")
        self.tg_data.to_csv(csv_path, index=False)
        print(f"[TgAnnealAnalyzer] tg_data saved to {csv_path}")

    def load_tg_data(self, csv_path="tg_data.csv"):
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"{csv_path} not found.")
        self.tg_data = pd.read_csv(csv_path)
        print(f"[TgAnnealAnalyzer] tg_data loaded from {csv_path} ({len(self.tg_data)} points).")

    # ------------------------------------------------------------------
    # 5. 拟合 Tg 并画图
    # ------------------------------------------------------------------
    def fit_tg(
        self,
        lt_Tmin,
        lt_Tmax,
        ht_Tmin,
        ht_Tmax,
        label="PEO_LiTFSI",
        color="tab:blue",
        marker="o",
        show_points=True,
        figsize=(6, 4),
    ):
        if self.tg_data is None:
            raise RuntimeError("tg_data is None. 请先 build 或 load tg_data。")

        temp = self.tg_data["T_K"].values
        dens = self.tg_data["den_mean"].values
        den_err = self.tg_data.get(
            "den_std", pd.Series([np.nan] * len(self.tg_data))
        ).values

        temp = np.array(temp)
        dens = np.array(dens)

        high_mask = (temp >= ht_Tmin) & (temp <= ht_Tmax)
        low_mask = (temp >= lt_Tmin) & (temp <= lt_Tmax)

        high_t, high_d = temp[high_mask], dens[high_mask]
        low_t, low_d = temp[low_mask], dens[low_mask]

        if len(high_t) < 2 or len(low_t) < 2:
            raise ValueError(
                f"拟合点太少：low-T 点数={len(low_t)}, high-T 点数={len(high_t)}"
            )

        # 线性拟合：ρ = m T + c
        m_high, c_high = np.polyfit(high_t, high_d, 1)
        m_low, c_low = np.polyfit(low_t, low_d, 1)

        # 交点温度 Tg
        tg_cal = (c_low - c_high) / (m_high - m_low)
        den_tg = m_low * tg_cal + c_low

        # 画图
        T_fit = np.linspace(temp.min(), temp.max(), 200)
        plt.figure(figsize=figsize)

        if show_points:
            plt.errorbar(
                temp,
                dens,
                yerr=den_err,
                fmt=marker + "--",
                lw=1,
                label=label,
                color=color,
                capsize=3,
            )

        plt.plot(
            T_fit, m_high * T_fit + c_high,
            ls="-.", c="red", alpha=0.7, lw=1, label="high-T fit",
        )
        plt.plot(
            T_fit, m_low * T_fit + c_low,
            ls="-.", c="blue", alpha=0.7, lw=1, label="low-T fit",
        )
        plt.scatter(
            tg_cal, den_tg,
            marker="X", c="k", zorder=5,
            label=f"Tg ≈ {tg_cal:.1f} K",
        )

        plt.xlabel("T [K]")
        plt.ylabel(r"$\rho$ [kg m$^{-3}$]")
        plt.legend()
        plt.tight_layout()
        plt.show()

        print(f"[TgAnnealAnalyzer] Estimated Tg = {tg_cal:.2f} K")
        return tg_cal

