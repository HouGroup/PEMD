from pathlib import Path
from typing import Union
import os
from ase import Atoms
from ase.io import read, write
from ase.optimize import LBFGS
import torch

from fairchem.core import pretrained_mlip, FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _load_uma_predictor(
    *,
    model_name: str,
    device: str,
    use_local_uma: bool = False,
    local_checkpoint: Union[Path, str, None] = None,
    hf_cache_dir: Union[Path, str, None] = None,
    hf_offline: bool = False,
):
    """
    返回 UMA predictor 对象。
    优先：本地 TorchScript/pt；否则：在线/缓存加载官方预训练权重。
    """
    # 1) 本地权重优先
    if use_local_uma and local_checkpoint is not None:
        ckpt_path = os.getenv("UMA_CKPT", str(local_checkpoint))
        ckpt_file = Path(ckpt_path).expanduser().resolve()
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"[conf_uma] 本地检查点未找到：{ckpt_file}\n"
                f"可设置环境变量 UMA_CKPT 或参数 local_checkpoint 指向对应模型文件（例如 {model_name}.pt）。"
            )
        predictor = load_predict_unit(path=str(ckpt_file), device=device)
        return predictor

    # 2) 官方预训练（可指定缓存/离线）
    if hf_cache_dir is not None:
        hf_home = Path(hf_cache_dir).expanduser().resolve()
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HOME"]

    if hf_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        predictor = pretrained_mlip.get_predict_unit(
            model_name, device=device, cache_dir=os.environ.get("HF_HOME", None)
        )
    except TypeError:
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)

    return predictor

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
) -> str:
    """
    使用 UMA 机器学习势对多构象进行几何优化与能量筛选。
    - top_n_uma=None（默认）：保留全部构象
    - top_n_uma>0：保留能量最低的 N 个构象
    """
    work_path = Path(work_dir)

    uma_dir = None
    if opt_log:
        uma_dir = work_path / "UMA_dir"
        uma_dir.mkdir(parents=True, exist_ok=True)
    xyz_in = Path(xyz_in)

    atoms_list: list[Atoms] = read(xyz_in.as_posix(), index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if len(atoms_list) == 0:
        raise ValueError(f"No conformers found in {xyz_in}")

    dev = device or _pick_device()
    predictor = _load_uma_predictor(
        model_name=model_name,
        device=dev,
        use_local_uma=use_local_uma,
        local_checkpoint=local_checkpoint,
        hf_cache_dir=hf_cache_dir,
        hf_offline=hf_offline,
    )
    calc = FAIRChemCalculator(predictor, task_name=task_name)

    results: list[tuple[float, Atoms]] = []
    for i, atoms in enumerate(atoms_list, start=1):
        info = atoms.info.copy()
        info.update({"charge": int(charge), "spin": int(mult)})
        atoms.info = info
        atoms.calc = calc

        dyn = LBFGS(atoms, logfile=None if not opt_log else f"{uma_dir}/uma_opt_{i}.log")
        dyn.run(fmax=fmax, steps=max_steps)

        e = atoms.get_potential_energy()
        # 可选：把能量写进 info，便于后续追踪
        atoms.info["E_UMA_eV"] = float(e)
        results.append((e, atoms.copy()))

    results.sort(key=lambda t: t[0])

    n_total = len(results)
    if (top_n_uma is None) or (top_n_uma in (0, -1)) or (top_n_uma >= n_total):
        n_keep = n_total
        fname = "uma_all.xyz"
    else:
        n_keep = max(1, int(top_n_uma))
        fname = f"uma_top{n_keep}.xyz"

    top = results[:n_keep]
    out_path = work_path / fname
    write(out_path.as_posix(), [a for (_, a) in top], format="xyz")

    print(f"[3] Saved {n_keep}/{n_total} lowest-energy UMA structures to '{fname}'.")
    return out_path.as_posix()

