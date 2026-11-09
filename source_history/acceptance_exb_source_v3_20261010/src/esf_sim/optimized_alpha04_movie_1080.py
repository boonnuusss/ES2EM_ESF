"""Exact-1920x1080 wrapper around the fast alpha=0.4 comparison encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio_ffmpeg

from . import optimized_alpha04_movie_fast as workflow


def _exact_writer(path):
    writer=imageio_ffmpeg.write_frames(
        str(path),(workflow.WIDTH,workflow.HEIGHT),fps=workflow.FPS,
        codec="libx264",pix_fmt_in="rgb24",macro_block_size=1,
        output_params=["-pix_fmt","yuv420p","-crf","18","-preset","fast","-movflags","+faststart"],
    )
    writer.send(None)
    return writer


def _metadata(path):
    reader=imageio_ffmpeg.read_frames(str(path),pix_fmt="rgb24")
    try:
        meta=next(reader)
    finally:
        reader.close()
    return {"size_pixels":list(meta["size"]),"fps":float(meta["fps"]),
            "duration_s":float(meta["duration"])}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--baseline-root",default=str(workflow.BASELINE_ROOT))
    parser.add_argument("--optimized-root",default=str(workflow.OPTIMIZED_ROOT))
    parser.add_argument("--output",default=str(workflow.OUTPUT_ROOT))
    args=parser.parse_args()
    workflow._writer=_exact_writer
    report=workflow.create_fast_movie(args.baseline_root,args.optimized_root,args.output)
    full_meta=_metadata(report["full_movie"]["path"])
    preview_meta=_metadata(report["preview_movie"]["path"])
    report["full_movie"].update(full_meta); report["preview_movie"].update(preview_meta)
    report["checks"]["full_movie_exact_1920x1080"]=full_meta["size_pixels"]==[1920,1080]
    report["checks"]["preview_movie_exact_1920x1080"]=preview_meta["size_pixels"]==[1920,1080]
    report["checks"]["metadata_fps_30"]=abs(full_meta["fps"]-30)<1e-9 and abs(preview_meta["fps"]-30)<1e-9
    report["all_checks_pass"]=all(report["checks"].values())
    path=Path(args.output)/"comparison_movie_report.json"
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    if not report["all_checks_pass"]: raise RuntimeError(report["checks"])
    print(json.dumps(report,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
