"""Fast exact-1080p baseline/optimized comparison movie for alpha=0.2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from .optimized_alpha04_movie import ArchiveReader
from .optimized_alpha04_movie_fast import (
    FPS, HEIGHT, PANEL_H, PANEL_W, PANEL_X, PANEL_Y, WIDTH,
    _center, _font, _panel, _static_canvas,
)


BASELINE_ROOT=Path("output/best_baseline/figure3_v2_timeseries")
OPTIMIZED_ROOT=Path("output/optimization/alpha02_to_5050_corrected")
OUTPUT_ROOT=Path("output/optimization/alpha02_baseline_vs_optimized_movie_corrected")
END_TIME=5050


class Alpha02BaselineTimeline:
    def __init__(self, root=BASELINE_ROOT):
        root=Path(root)
        self.parts=[
            ArchiveReader(root/"time_series"/"alpha_0p2_recovered"),
            ArchiveReader(root/"extended_1000s"/"time_series"/"alpha_0p2"),
            ArchiveReader(root/"extended_1000s_round2"/"time_series"/"alpha_0p2"),
        ]
        expected=((0,3150),(3150,4150),(4150,5150))
        for part,bounds in zip(self.parts,expected):
            if (part.first,part.last)!=bounds: raise ValueError((part.directory,part.first,part.last))
        self.seam_3150=bool(np.array_equal(self.parts[0].frame(3150),self.parts[1].frame(3150)))
        self.seam_4150=bool(np.array_equal(self.parts[1].frame(4150),self.parts[2].frame(4150)))
        if not self.seam_3150 or not self.seam_4150: raise ValueError("alpha=0.2 baseline seam mismatch")

    def frame(self,t):
        if t<=3150:return self.parts[0].frame(t)
        if t<=4150:return self.parts[1].frame(t)
        return self.parts[2].frame(t)


def _writer(path):
    writer=imageio_ffmpeg.write_frames(
        str(path),(WIDTH,HEIGHT),fps=FPS,codec="libx264",pix_fmt_in="rgb24",
        macro_block_size=1,
        output_params=["-pix_fmt","yuv420p","-crf","18","-preset","fast","-movflags","+faststart"],
    );writer.send(None);return writer


def _metadata(path):
    reader=imageio_ffmpeg.read_frames(str(path),pix_fmt="rgb24")
    try: meta=next(reader)
    finally: reader.close()
    return {"size_pixels":list(meta["size"]),"fps":float(meta["fps"]),"duration_s":float(meta["duration"])}


def create_movie(baseline_root=BASELINE_ROOT,optimized_root=OPTIMIZED_ROOT,output=OUTPUT_ROOT):
    optimized_root=Path(optimized_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    run_report=json.loads((optimized_root/"optimized_run_report.json").read_text(encoding="utf-8"))
    switch=float(run_report["operator_switch_time_s"])
    baseline=Alpha02BaselineTimeline(baseline_root);optimized=ArchiveReader(optimized_root/"time_series")
    if (optimized.first,optimized.last)!=(0,END_TIME):raise ValueError("optimized archive bounds")
    initial_exact=bool(np.array_equal(baseline.frame(0),optimized.frame(0)))
    if not initial_exact:raise ValueError("baseline and optimized initial frames differ")
    full_path=output/"alpha02_baseline_vs_optimized_0_5050s.mp4"
    preview_path=output/"alpha02_baseline_vs_optimized_preview_10s.mp4"
    full=_writer(full_path);preview=_writer(preview_path)
    static,lut=_static_canvas();time_font=_font(31,True);mode_font=_font(20,True);final=None
    try:
        for t in range(END_TIME+1):
            frame=static.copy();frame.paste(_panel(baseline.frame(t),lut),(PANEL_X[0],PANEL_Y));frame.paste(_panel(optimized.frame(t),lut),(PANEL_X[1],PANEL_Y))
            draw=ImageDraw.Draw(frame);_center(draw,(WIDTH/2,25),f"PDM7 α = 0.2    t = {t:,d} s",time_font)
            mode="point-centred potential" if t<switch else f"monotone exponential potential · switched at t ≈ {switch:.1f} s"
            _center(draw,(PANEL_X[1]+PANEL_W/2,112),mode,mode_font,fill=(40,40,40))
            array=np.asarray(frame);full.send(array)
            if t%10==0:preview.send(array)
            if t==END_TIME:final=frame
            if t and t%500==0:print(f"[alpha02-comparison-movie] encoded t={t} s",flush=True)
    finally:full.close();preview.close()
    poster=output/"alpha02_baseline_vs_optimized_final.png";final.save(poster)
    fc,fd=imageio_ffmpeg.count_frames_and_secs(str(full_path));pc,pd=imageio_ffmpeg.count_frames_and_secs(str(preview_path))
    fm,pm=_metadata(full_path),_metadata(preview_path)
    checks={"initial_frames_exact":initial_exact,"baseline_seams_exact":baseline.seam_3150 and baseline.seam_4150,
        "optimized_archive_5051_frames":int(optimized.manifest["frame_count"])==5051,
        "full_movie_5051_frames":fc==5051,"preview_movie_506_frames":pc==506,
        "full_movie_exact_1920x1080":fm["size_pixels"]==[1920,1080],"preview_movie_exact_1920x1080":pm["size_pixels"]==[1920,1080],
        "metadata_fps_30":fm["fps"]==30 and pm["fps"]==30,"full_duration_correct":math.isclose(fd,5051/30,abs_tol=.05)}
    payload={"format":"esf_alpha02_baseline_optimized_comparison_v1","end_time_s":END_TIME,"operator_switch_time_s":switch,
        "full_movie":{"path":str(full_path.resolve()),"frames":fc,"duration_s":fd,**fm},
        "preview_movie":{"path":str(preview_path.resolve()),"frames":pc,"duration_s":pd,**pm},
        "poster":str(poster.resolve()),"checks":checks,"all_checks_pass":all(checks.values())}
    (output/"comparison_movie_report.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    if not payload["all_checks_pass"]:raise RuntimeError(checks)
    return payload


def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline-root",default=str(BASELINE_ROOT));p.add_argument("--optimized-root",default=str(OPTIMIZED_ROOT));p.add_argument("--output",default=str(OUTPUT_ROOT));a=p.parse_args();print(json.dumps(create_movie(a.baseline_root,a.optimized_root,a.output),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
