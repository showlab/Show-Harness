
# Show-Harness: Just a VLM Agent Can Play Robots

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

This branch hosts the **project page** for Show-Harness. It is a static site — no build
step and no package manager — so GitHub Pages can serve it as it stands.

```
index.html             the whole page, including the inlined experiment figures
assets/dist/           Bootstrap 5.3 (vendored, minified build only)
assets/images/         overview, method and teaser figures as SVG
assets/videos/         the narrated overview clip and the GUMI rollout
assets/videos/gallery_2x/  35 real-robot demos, sped up for playback
assets/videos/posters/     JPEG covers so a tile shows content before its clip loads
.nojekyll              keeps GitHub Pages from running Jekyll over assets/
```

### What is not in this branch

`.gitignore` keeps out everything the page does not actually load: the 1× gallery
masters and their covers (~270 MB), the standalone copies of the figures that are
inlined into `index.html`, Bootstrap's source maps and RTL build, and the internal
preview page. Nothing here is generated at deploy time — what is committed is what
is served.

### Local preview

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

Opening `index.html` straight from disk also works, but a local server is closer to
production: `file://` does not support HTTP range requests, so the demo videos have to
download in full before they start.

### Where the page comes from

The page is authored in the main repository's working tree, not here, and three tools
own the parts that would otherwise drift:

- `page/tools/gallery.py` encodes a demo clip, cuts its cover and rewrites the `CLIPS`
  manifest inside `index.html`. `gallery.py check` reports anything out of sync.
- `page/tools/figures.py` inlines the experiment figures so the page can animate their
  bars and switches, namespacing every id on the way in.
- `page/tools/illustrations.py` shrinks the hand-drawn figures for the web.

Inside `index.html`, `CAT_CLIPS` decides which shelf each demo sits on and in what
order; it is keyed by capture id so it survives a manifest rebuild.
