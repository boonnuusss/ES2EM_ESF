"""Fast PIL/LUT encoder for the alpha=0.4 baseline comparison movie."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .optimized_alpha04_movie import (
    ArchiveReader, BaselineTimeline, BASELINE_ROOT, END_TIME,
    OPTIMIZED_ROOT, OUTPUT_ROOT,
)


WIDTH, HEIGHT, FPS = 1920, 1080, 30
PANEL_W, PANEL_H = 500, 800
PANEL_Y = 145
PANEL_X = (225, 900)


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(mpl.get_data_path())/"fonts"/"ttf"/name),size)


def _center(draw, xy, text, font, fill="black"):
    box=draw.textbbox((0,0),text,font=font)
    draw.text((xy[0]-(box[2]-box[0])/2,xy[1]),text,font=font,fill=fill)


def _static_canvas():
    image=Image.new("RGB",(WIDTH,HEIGHT),"white"); draw=ImageDraw.Draw(image)
    title=_font(34,True); label=_font(27); tick=_font(21); small=_font(20)
    for x,name in zip(PANEL_X,("Formal v2 baseline","Optimized production solver")):
        draw.rectangle((x-2,PANEL_Y-2,x+PANEL_W+2,PANEL_Y+PANEL_H+2),outline="black",width=3)
        _center(draw,(x+PANEL_W/2,82),name,title)
        _center(draw,(x+PANEL_W/2,970),"East–West (km)",label)
        for value in (-100,-50,0,50,100):
            px=x+(value+125)/250*PANEL_W
            draw.line((px,PANEL_Y+PANEL_H,px,PANEL_Y+PANEL_H+9),fill="black",width=2)
            _center(draw,(px,PANEL_Y+PANEL_H+12),str(value),tick)
        for value in (250,350,450,550,650):
            py=PANEL_Y+PANEL_H-(value-250)/400*PANEL_H
            draw.line((x-9,py,x,py),fill="black",width=2)
            if x==PANEL_X[0]:
                text=str(value); box=draw.textbbox((0,0),text,font=tick)
                draw.text((x-17-(box[2]-box[0]),py-(box[3]-box[1])/2),text,font=tick,fill="black")
    # Rotated y label.
    ylabel=Image.new("RGBA",(300,50),(255,255,255,0)); yd=ImageDraw.Draw(ylabel)
    yd.text((0,5),"Altitude (km)",font=label,fill="black")
    ylabel=ylabel.rotate(90,expand=True,resample=Image.Resampling.BICUBIC)
    image.paste(ylabel,(35,390),ylabel)
    # Plasma colourbar.
    cmap=mpl.colormaps["plasma"]
    lut=(cmap(np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
    gradient=np.tile(lut[::-1,None,:],(1,42,1))
    bar=Image.fromarray(gradient).resize((42,650),Image.Resampling.BILINEAR)
    bx,by=1550,220; image.paste(bar,(bx,by)); draw.rectangle((bx-1,by-1,bx+43,by+651),outline="black",width=2)
    for value in range(12):
        py=by+650-value/11*650
        draw.line((bx+43,py,bx+52,py),fill="black",width=2)
        draw.text((bx+58,py-12),str(value),font=tick,fill="black")
    draw.text((1490,165),"Plasma density",font=small,fill="black")
    draw.text((1488,190),"(10⁵ cm⁻³)",font=small,fill="black")
    return image,lut


def _panel(density,lut):
    index=np.clip(np.asarray(density,dtype=np.float64)/1.1e12*255,0,255).astype(np.uint8)
    rgb=lut[index]
    return Image.fromarray(rgb[::-1]).resize((PANEL_W,PANEL_H),Image.Resampling.BILINEAR)


def _writer(path):
    writer=imageio_ffmpeg.write_frames(
        str(path),(WIDTH,HEIGHT),fps=FPS,codec="libx264",pix_fmt_in="rgb24",
        output_params=["-pix_fmt","yuv420p","-crf","18","-preset","fast","-movflags","+faststart"],
    ); writer.send(None); return writer


def create_fast_movie(baseline_root=BASELINE_ROOT,optimized_root=OPTIMIZED_ROOT,output=OUTPUT_ROOT):
    optimized_root=Path(optimized_root); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    run_report=json.loads((optimized_root/"optimized_run_report.json").read_text(encoding="utf-8"))
    switch_time=float(run_report["operator_switch_time_s"])
    baseline=BaselineTimeline(baseline_root); optimized=ArchiveReader(optimized_root/"time_series")
    if (optimized.first,optimized.last)!=(0,END_TIME): raise ValueError("optimized archive bounds")
    full_path=output/"alpha04_baseline_vs_optimized_0_5011s.mp4"
    preview_path=output/"alpha04_baseline_vs_optimized_preview_10s.mp4"
    full=_writer(full_path); preview=_writer(preview_path)
    static,lut=_static_canvas(); time_font=_font(31,True); mode_font=_font(20,True)
    final=None
    try:
        for t in range(END_TIME+1):
            frame=static.copy(); frame.paste(_panel(baseline.frame(t),lut),(PANEL_X[0],PANEL_Y))
            frame.paste(_panel(optimized.frame(t),lut),(PANEL_X[1],PANEL_Y))
            draw=ImageDraw.Draw(frame)
            _center(draw,(WIDTH/2,25),f"PDM7 α = 0.4    t = {t:,d} s",time_font)
            if t < switch_time: mode="point-centred potential"
            else: mode=f"monotone exponential potential · switched at t ≈ {switch_time:.0f} s"
            _center(draw,(PANEL_X[1]+PANEL_W/2,112),mode,mode_font,fill=(40,40,40))
            array=np.asarray(frame)
            full.send(array)
            if t%10==0: preview.send(array)
            if t==END_TIME: final=frame
            if t and t%500==0: print(f"[fast-comparison-movie] encoded t={t} s",flush=True)
    finally:
        full.close(); preview.close()
    poster=output/"alpha04_baseline_vs_optimized_final.png"; final.save(poster)
    full_count,full_duration=imageio_ffmpeg.count_frames_and_secs(str(full_path))
    preview_count,preview_duration=imageio_ffmpeg.count_frames_and_secs(str(preview_path))
    checks={
        "baseline_seams_exact":baseline.seam_3150 and baseline.seam_4150,
        "optimized_archive_5012_frames":int(optimized.manifest["frame_count"])==5012,
        "full_movie_5012_frames":full_count==5012,
        "preview_movie_502_frames":preview_count==502,
        "full_duration_correct":math.isclose(full_duration,5012/FPS,abs_tol=.05),
        "preview_duration_correct":math.isclose(preview_duration,502/FPS,abs_tol=.05),
    }
    payload={"format":"esf_alpha04_baseline_optimized_comparison_v2_fast",
        "end_time_s":END_TIME,"fps":FPS,"resolution":[WIDTH,HEIGHT],
        "operator_switch_time_s":switch_time,
        "full_movie":{"path":str(full_path.resolve()),"frames":full_count,"duration_s":full_duration},
        "preview_movie":{"path":str(preview_path.resolve()),"frames":preview_count,"duration_s":preview_duration},
        "poster":str(poster.resolve()),"checks":checks,"all_checks_pass":all(checks.values())}
    (output/"comparison_movie_report.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    if not payload["all_checks_pass"]: raise RuntimeError(checks)
    return payload


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--baseline-root",default=str(BASELINE_ROOT))
    parser.add_argument("--optimized-root",default=str(OPTIMIZED_ROOT)); parser.add_argument("--output",default=str(OUTPUT_ROOT))
    a=parser.parse_args(); print(json.dumps(create_fast_movie(a.baseline_root,a.optimized_root,a.output),indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
