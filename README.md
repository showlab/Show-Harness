
# Show-Pro: Embodied Harness Enables VLMs to Play Robots

<p align="center">
  <em>A compact semantic interface that lets vision–language models act on the physical world.</em>
</p>

<p align="center">
  Yanzhe Chen*, Zechen Bai*, Zhijun Cao*, Wenzheng Zeng*, Kevin Qinghong Lin, <br>
  Yiqi Lin, Guoqiang Liang, Kevin Yuchen Ma, Qiming Huang, Mike Zheng Shou† <br>
  Show Lab @ National University of Singapore
</p>

<p align="center">
  <sup>*</sup>Equal contribution &nbsp;&middot;&nbsp; <sup>†</sup>Corresponding author
</p>

---

This branch hosts the **project page** for Show-Pro. It is a static site — no build
step and no package manager — so it can be served directly by GitHub Pages.

```
index.html            the whole page
assets/dist/          Bootstrap 5.3 (vendored)
assets/images/        figures, all vector SVG
assets/videos/        the overview clip, the GUMI rollout and 32 real-robot demos
assets/videos/posters small JPEG covers so a tile shows content before its clip loads
.nojekyll             keeps GitHub Pages from running Jekyll over assets/
```

### Local preview

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Opening `index.html` straight from disk also works, but a local server is closer to
production: `file://` does not support HTTP range requests, so the demo videos have to
download in full before they start.

### Editing the demo gallery

The gallery is generated from a manifest inside `index.html` — the `CLIPS` array holds
one `{file, dur, tag}` entry per clip. Adding a demo means dropping the `.mp4` into
`assets/videos/gallery/`, a matching `.jpg` cover into `assets/videos/posters/`, and one
line into that array. Clips are grouped by `tag` and load lazily as they scroll into
view, so the page stays responsive even with dozens of them.
