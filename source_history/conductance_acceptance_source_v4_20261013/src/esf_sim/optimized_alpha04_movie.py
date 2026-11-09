"""Side-by-side movie of baseline and optimized alpha=0.4 EPB evolution."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


BASELINE_ROOT = Path("output/best_baseline/figure3_v2_timeseries")
OPTIMIZED_ROOT = Path("output/optimization/alpha04_to_5011")
OUTPUT_ROOT = Path("output/optimization/alpha04_baseline_vs_optimized_movie")
END_TIME = 5011


class ArchiveReader:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/"manifest.json").read_text(encoding="utf-8"))
        self.chunks = self.manifest["chunks"]
        self.first = int(round(self.chunks[0]["first_time_s"]))
        self.last = int(round(self.chunks[-1]["last_time_s"]))
        self._index = -1; self._first = self._last = None; self._density = None
        expected = self.first
        for entry in self.chunks:
            first, last = int(entry["first_time_s"]), int(entry["last_time_s"])
            if first != expected or int(entry["frame_count"]) != last-first+1:
                raise ValueError(f"non-contiguous archive: {self.directory}")
            expected = last+1

    def frame(self, time_s):
        if not self.first <= time_s <= self.last:
            raise KeyError((self.directory,time_s))
        if self._density is None or not self._first <= time_s <= self._last:
            for index, entry in enumerate(self.chunks):
                first, last = int(entry["first_time_s"]), int(entry["last_time_s"])
                if first <= time_s <= last:
                    with np.load(self.directory/entry["file"],allow_pickle=False) as z:
                        density=np.asarray(z["density_m3"],dtype=np.float32)
                        times=np.asarray(z["time_s"])
                    if not np.array_equal(times,np.arange(first,last+1)):
                        raise ValueError(f"bad time vector in {entry['file']}")
                    self._index=index; self._first=first; self._last=last; self._density=density
                    break
        return self._density[time_s-self._first]


class BaselineTimeline:
    def __init__(self, root=BASELINE_ROOT):
        root=Path(root)
        self.parts=[
            ArchiveReader(root/"time_series"/"alpha_0p4"),
            ArchiveReader(root/"extended_1000s"/"time_series"/"alpha_0p4"),
            ArchiveReader(root/"extended_1000s_round2"/"time_series"/"alpha_0p4"),
        ]
        expected=((0,3150),(3150,4150),(4150,5011))
        for part,bounds in zip(self.parts,expected):
            if (part.first,part.last) != bounds: raise ValueError((part.directory,part.first,part.last))
        self.seam_3150=bool(np.array_equal(self.parts[0].frame(3150),self.parts[1].frame(3150)))
        self.seam_4150=bool(np.array_equal(self.parts[1].frame(4150),self.parts[2].frame(4150)))
        if not self.seam_3150 or not self.seam_4150: raise ValueError("baseline seam mismatch")

    def frame(self,time_s):
        if time_s <= 3150: return self.parts[0].frame(time_s)
        if time_s <= 4150: return self.parts[1].frame(time_s)
        return self.parts[2].frame(time_s)


def _movie_writer(path, fps=30):
    writer=imageio_ffmpeg.write_frames(
        str(path),(1920,1080),fps=fps,codec="libx264",pix_fmt_in="rgb24",
        output_params=["-pix_fmt","yuv420p","-crf","18","-preset","medium","-movflags","+faststart"],
    )
    writer.send(None)
    return writer


def _build_figure():
    fig,axes=plt.subplots(1,2,figsize=(16,9),dpi=120,sharey=True)
    fig.subplots_adjust(left=.07,right=.90,bottom=.10,top=.88,wspace=.09)
    blank=np.zeros((401,201))
    images=[]
    for ax,title in zip(axes,("Formal v2 baseline","Optimized production solver")):
        image=ax.imshow(blank,origin="lower",extent=(-125,125,250,650),aspect="auto",
            cmap="plasma",norm=Normalize(0,11),interpolation="nearest")
        images.append(image); ax.set_title(title,fontsize=19,weight="bold")
        ax.set_xlabel("East–West (km)",fontsize=16); ax.tick_params(labelsize=13)
    axes[0].set_ylabel("Altitude (km)",fontsize=16)
    cax=fig.add_axes([.92,.18,.018,.60]); cb=fig.colorbar(images[0],cax=cax)
    cb.set_label(r"Plasma density ($10^5$ cm$^{-3}$)",fontsize=14)
    cb.ax.tick_params(labelsize=12)
    time_text=fig.text(.5,.945,"",ha="center",fontsize=20,weight="bold")
    mode_text=axes[1].text(.03,.97,"",transform=axes[1].transAxes,va="top",color="white",
        fontsize=13,weight="bold",bbox=dict(facecolor="black",alpha=.35,edgecolor="none"))
    fig.canvas.draw()
    return fig,images,time_text,mode_text


def create_movie(baseline_root=BASELINE_ROOT,optimized_root=OPTIMIZED_ROOT,output=OUTPUT_ROOT):
    optimized_root=Path(optimized_root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    status=json.loads((optimized_root/"status.json").read_text(encoding="utf-8"))
    report=json.loads((optimized_root/"optimized_run_report.json").read_text(encoding="utf-8"))
    if status.get("state") != "completed" or int(status.get("frame_count",-1)) != END_TIME+1:
        raise RuntimeError("optimized simulation is not complete")
    switch_time=float(report["operator_switch_time_s"])
    baseline=BaselineTimeline(baseline_root)
    optimized=ArchiveReader(optimized_root/"time_series")
    if (optimized.first,optimized.last)!=(0,END_TIME): raise ValueError("optimized archive bounds")

    full_path=output/"alpha04_baseline_vs_optimized_0_5011s.mp4"
    preview_path=output/"alpha04_baseline_vs_optimized_preview_10s.mp4"
    full=_movie_writer(full_path); preview=_movie_writer(preview_path)
    fig,images,time_text,mode_text=_build_figure()
    final_rgb=None
    try:
        for t in range(END_TIME+1):
            images[0].set_data(baseline.frame(t)/1e11)
            images[1].set_data(optimized.frame(t)/1e11)
            time_text.set_text(f"PDM7 α = 0.4    t = {t:,d} s")
            if t < switch_time:
                mode="point-centred potential"
            else:
                mode=f"monotone exponential potential\nswitched at t ≈ {switch_time:.1f} s"
            mode_text.set_text(mode)
            fig.canvas.draw()
            rgb=np.asarray(fig.canvas.buffer_rgba())[...,:3].copy()
            full.send(rgb)
            if t%10==0: preview.send(rgb)
            if t==END_TIME: final_rgb=rgb
            if t and t%500==0: print(f"[comparison-movie] encoded t={t} s",flush=True)
    finally:
        full.close(); preview.close(); plt.close(fig)
    assert final_rgb is not None
    from PIL import Image
    poster=output/"alpha04_baseline_vs_optimized_final.png"
    Image.fromarray(final_rgb).save(poster)
    full_count,full_duration=imageio_ffmpeg.count_frames_and_secs(str(full_path))
    preview_count,preview_duration=imageio_ffmpeg.count_frames_and_secs(str(preview_path))
    checks={
        "baseline_seams_exact":baseline.seam_3150 and baseline.seam_4150,
        "optimized_frame_count":int(optimized.manifest["frame_count"])==5012,
        "full_movie_frame_count":full_count==5012,
        "preview_movie_frame_count":preview_count==502,
        "full_duration":math.isclose(full_duration,5012/30,abs_tol=.05),
        "preview_duration":math.isclose(preview_duration,502/30,abs_tol=.05),
    }
    payload={
        "format":"esf_alpha04_baseline_optimized_comparison_v1",
        "simulation_end_s":END_TIME,"fps":30,"resolution":[1920,1080],
        "operator_switch_time_s":switch_time,
        "baseline_root":str(Path(baseline_root).resolve()),
        "optimized_root":str(optimized_root.resolve()),
        "full_movie":{"path":str(full_path.resolve()),"frames":full_count,"duration_s":full_duration},
        "preview_movie":{"path":str(preview_path.resolve()),"frames":preview_count,"duration_s":preview_duration},
        "poster":str(poster.resolve()),"checks":checks,"all_checks_pass":all(checks.values()),
    }
    (output/"comparison_movie_report.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    if not payload["all_checks_pass"]: raise RuntimeError(checks)
    return payload


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline-root",default=str(BASELINE_ROOT))
    parser.add_argument("--optimized-root",default=str(OPTIMIZED_ROOT))
    parser.add_argument("--output",default=str(OUTPUT_ROOT))
    args=parser.parse_args()
    print(json.dumps(create_movie(args.baseline_root,args.optimized_root,args.output),indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
