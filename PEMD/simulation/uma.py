from pathlib import Path
from typing import Union
import os, re
from ase import Atoms
from ase.io import read
from ase.optimize import LBFGS
import torch

from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def _load_uma_predictor(
    *, model_name: str, device: str,
    use_local_uma: bool = False,
    local_checkpoint: Union[Path, str, None] = None,
    hf_cache_dir: Union[Path, str, None] = None,
    hf_offline: bool = False,
):
    if use_local_uma and local_checkpoint is not None:
        ckpt_path = os.getenv("UMA_CKPT", str(local_checkpoint))
        ckpt_file = Path(ckpt_path).expanduser().resolve()
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"[conf_uma] 本地检查点未找到：{ckpt_file}\n"
                f"可设置 UMA_CKPT 或 local_checkpoint 指向 {model_name}.pt"
            )
        return load_predict_unit(path=str(ckpt_file), device=device)

    if hf_cache_dir is not None:
        hf_home = Path(hf_cache_dir).expanduser().resolve()
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"]
    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        return pretrained_mlip.get_predict_unit(model_name, device=device, cache_dir=os.environ.get("HF_HOME"))
    except TypeError:
        return pretrained_mlip.get_predict_unit(model_name, device=device)

# ---------- 注释解析 & Success 规范化 ----------
def _normalize_success(v):
    if isinstance(v, bool): return v
    if v is None: return None
    s = str(v).strip().lower()
    if s in {"true","t","1","yes","y","ok","success","s"}: return True
    if s in {"false","f","0","no","n","fail","fails","failed","error"}: return False
    return None

def _parse_comment_meta(line: str) -> dict:
    meta = {}
    for k, v in re.findall(r'([A-Za-z_]\w*)\s*=\s*([^;]+)', line):
        key = k.strip(); val = v.strip()
        low = val.lower()
        if low in {"true","false","t","f","yes","no","y","n"}:
            meta[key] = _normalize_success(val)
        else:
            try:
                fv = float(val)
                meta[key] = int(fv) if fv.is_integer() else fv
            except ValueError:
                meta[key] = val
    return meta

def _attach_meta_from_xyz(xyz_path: Path, atoms_list: list[Atoms]) -> None:
    lines = Path(xyz_path).read_text().splitlines()
    i = 0; idx = 0; nlines = len(lines)
    while i < nlines and idx < len(atoms_list):
        if not lines[i].strip():
            i += 1; continue
        try:
            nat = int(lines[i].strip())
        except ValueError:
            i += 1; continue
        if i + 1 < nlines:
            meta = _parse_comment_meta(lines[i + 1])
            a = atoms_list[idx]
            if "id" in meta and "id" not in a.info:
                a.info["id"] = meta["id"]               # —— 保持输入 id 不变
            if "energy" in meta and "energy" not in a.info:
                a.info["energy"] = meta["energy"]
            suc = meta.get("Success", meta.get("success", None))
            suc = _normalize_success(suc)
            if suc is not None and "Success" not in a.info:
                a.info["Success"] = suc
            idx += 1
        i += 2 + nat
# ------------------------------------------------

def uma(
    work_dir: Union[Path, str],
    xyz_in: Union[Path, str],
    *,
    top_n_uma: int | None = None,
    charge: float = 0,
    mult: int = 1,
    model_name: str = "uma-m-1p1",
    task_name: str = "omol",
    max_steps: int = 500,
    fmax: float = 0.03,
    device: Union[str, None] = None,
    opt_log: bool = False,
    use_local_uma: bool = False,
    local_checkpoint: Union[Path, str, None] = None,
    hf_cache_dir: Union[Path, str, None] = None,
    hf_offline: bool = False,
    continue_on_fail: bool = False,
    continue_on_success: bool = False,
) -> str:
    """
    使用 UMA 对多构象优化，并以
    'id = ...; energy = ...; success = True/False'
    的注释行格式保存；id 保持与输入一致。
    """
    if continue_on_success and continue_on_fail:
        raise ValueError("continue_on_success 与 continue_on_fail 不能同时为 True")

    work_path = Path(work_dir)
    if opt_log:
        uma_dir = work_path / "UMA_dir"
        uma_dir.mkdir(parents=True, exist_ok=True)
    else:
        uma_dir = None

    xyz_in = Path(xyz_in)
    atoms_list: list[Atoms] = read(xyz_in.as_posix(), index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if len(atoms_list) == 0:
        raise ValueError(f"No conformers found in {xyz_in}")

    # 解析输入注释并写回 atoms.info，保证拿到 id/Success
    _attach_meta_from_xyz(xyz_in, atoms_list)

    # 根据 Success 做筛选（如果指定）
    if continue_on_success or continue_on_fail:
        filtered = []
        for a in atoms_list:
            flag = _normalize_success(a.info.get("Success", a.info.get("success")))
            if continue_on_success and flag is True:
                filtered.append(a)
            elif continue_on_fail and flag is False:
                filtered.append(a)
        if not filtered:
            raise ValueError("按 Success 条件筛选后没有可计算的构象；请检查输入 xyz 注释。")
        atoms_list = filtered

    dev = device or _pick_device()
    predictor = _load_uma_predictor(
        model_name=model_name, device=dev,
        use_local_uma=use_local_uma, local_checkpoint=local_checkpoint,
        hf_cache_dir=hf_cache_dir, hf_offline=hf_offline,
    )
    calc = FAIRChemCalculator(predictor, task_name=task_name)

    # 计算
    results: list[dict] = []
    for i, atoms in enumerate(atoms_list, start=1):
        # 沿用原始 id（若无则用顺序号占位，但不会改写已有 id）
        orig_id = atoms.info.get("id", i-1)
        atoms.info["id"] = orig_id

        atoms.info.update({"charge": int(charge), "spin": int(mult)})
        atoms.calc = calc

        # 运行优化；记录是否收敛
        success_flag = True
        try:
            dyn = LBFGS(atoms, logfile=None if not opt_log else str((uma_dir / f"uma_opt_{i}.log")))
            dyn.run(fmax=fmax, steps=max_steps)
            # 如果 ASE 版本支持 .converged 属性，就更严格判断
            if hasattr(dyn, "converged"):
                success_flag = bool(dyn.converged)
        except Exception:
            success_flag = False

        # 能量（eV，FAIRChemCalculator/ASE 约定）
        try:
            e = float(atoms.get_potential_energy())
        except Exception:
            e = float("nan"); success_flag = False

        atoms.info["E_UMA_eV"] = e
        results.append({
            "energy": e,
            "atoms": atoms.copy(),
            "id": orig_id,                 # —— 保持 id 不变
            "success": bool(success_flag),
        })

    # 按能量升序（NaN 视为 +inf 放末尾）
    results.sort(key=lambda d: (float("inf") if not (d["energy"] == d["energy"]) else d["energy"]))

    # 选 top_n
    selected = results[:top_n_uma]  # top_n_uma=None 时等于全部

    # NEW: 写出前按 id 升序排（0,1,2,4,7...）
    selected.sort(key=lambda d: _id_sort_key(d["id"]))

    fname = f"uma_top{top_n_uma}.xyz" if top_n_uma else "uma_all.xyz"
    out_path = work_path / fname

    # —— 手写 XYZ，确保注释格式与 id 保持 —— #
    with open(out_path, "w") as f:
        for item in selected:
            a: Atoms = item["atoms"]
            nat = len(a)
            f.write(f"{nat}\n")
            # 注释行：严格 'id = ...; energy = ...; success = True/False'
            eid = str(item["id"]).strip().strip(";")  # 防御：去掉尾随分号
            e = item["energy"]
            e_str = f"{e:.10f}" if (e == e) else "None"  # NaN 检查：NaN != NaN
            suc_str = "True" if item["success"] else "False"
            # 用 join 保证只有单一分号分隔
            f.write("; ".join([f"id = {eid}", f"energy = {e_str}", f"success = {suc_str}"]) + "\n")
            # 坐标行
            symbols = a.get_chemical_symbols()
            pos = a.get_positions()
            for s, (x, y, z) in zip(symbols, pos):
                f.write(f"{s:<2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    print(f"[UMA] Optimized {len(results)} conformers. Saved {len(selected)} to '{fname}'.")
    return out_path.as_posix()

def _id_sort_key(v):
    """优先按数值排序；若无法转为整数，则按字符串排序。"""
    try:
        return (0, int(v))
    except Exception:
        return (1, str(v))